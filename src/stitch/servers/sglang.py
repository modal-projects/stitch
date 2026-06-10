"""HTTP sidecar that adds weight-version protocol semantics to SGLang."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import uuid
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

from stitch.bulletin import FilesystemBulletinBoard
from stitch.engines.sglang import SGLangDiskDeltaAdapter
from stitch.protocol import WeightVersionPolicy
from stitch.sync import PolicyViolation, WeightSyncManager


logger = logging.getLogger(__name__)

# Engine control routes that must not be reachable through the body-blind
# gateway: a stray call would mutate engine state behind the sync manager's
# back and silently break its version bookkeeping. The sidecar itself calls
# these on the upstream directly, not through the proxy.
BLOCKED_ROUTES = frozenset(
    {
        "update_weights_from_disk",
        "update_weights_from_distributed",
        "update_weights_from_tensor",
        "update_weight_version",
        "init_weights_update_group",
        "destroy_weights_update_group",
        "flush_cache",
        "pause_generation",
        "continue_generation",
        "abort_request",
        "release_memory_occupation",
        "resume_memory_occupation",
        "post_process_weights",
    }
)


def create_app(
    manager: WeightSyncManager,
    *,
    upstream_url: str,
    versioned_routes: Iterable[str] = ("generate", "v1/chat/completions"),
):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    import httpx

    globals()["Request"] = Request
    globals()["Response"] = Response
    upstream_url = upstream_url.rstrip("/")
    versioned_route_set = {route.strip("/") for route in versioned_routes}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await manager.startup_sync()
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "current_version": manager.current_version}

    @app.get("/server_info")
    async def server_info() -> dict[str, Any]:
        return await manager.server_info()

    @app.get("/get_weight_version")
    async def get_weight_version() -> dict[str, str]:
        return {"weight_version": str(manager.current_version)}

    @app.post("/rpc_sync_from_bulletin_board")
    async def rpc_sync_from_bulletin_board(request: Request) -> dict[str, Any]:
        payload = await request.json()
        target = payload.get("target_version")
        manager.queue_sync(int(target) if target is not None else None)
        return {
            "accepted": True,
            "current_version": manager.current_version,
            "queued_target_version": manager.queued_target_version,
            "sync_state": manager.sync_state.value,
        }

    async def _watch_disconnect(request: Request) -> None:
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    async def _abort_upstream(rid: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
                await client.post(f"{upstream_url}/abort_request", json={"rid": rid})
        except Exception:  # noqa: BLE001
            logger.warning("sidecar_proxy failed to abort upstream rid=%s", rid, exc_info=True)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(path: str, request: Request) -> Response:
        route = path.strip("/")
        if route in BLOCKED_ROUTES:
            return JSONResponse(
                {
                    "error": {
                        "type": "RouteBlocked",
                        "message": f"/{route} is managed by the weight-sync sidecar and is not proxied",
                    }
                },
                status_code=403,
            )

        body = await request.body()
        payload: dict[str, Any] = {}
        if body and request.headers.get("content-type", "").startswith("application/json"):
            parsed = await request.json()
            if isinstance(parsed, dict):
                payload = parsed

        versioned_route = route in versioned_route_set
        policy = WeightVersionPolicy.from_payload(payload) if versioned_route else WeightVersionPolicy()
        request_id = request.headers.get("x-slime-request-id", "-")

        rid: str | None = None
        if versioned_route and payload:
            payload.pop("weight_version", None)
            # Inject a request id so the upstream generation can be aborted if
            # the client disconnects; otherwise abandoned requests keep
            # generating, holding the commit quiesce point for their full
            # remaining length.
            rid = payload.get("rid")
            if rid is None:
                rid = uuid.uuid4().hex
                payload["rid"] = rid

        try:
            ctx = manager.request_context(policy if versioned_route else None)
            async with ctx as start_version:
                started = asyncio.get_running_loop().time()
                if versioned_route and manager.debug_requests:
                    logger.info(
                        "sidecar_proxy start request_id=%s path=%s exact=%s current=%s active=%s",
                        request_id,
                        path,
                        policy.exact_version,
                        start_version,
                        manager.active_requests,
                    )

                async def _upstream_call() -> Any:
                    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
                        return await client.request(
                            request.method,
                            f"{upstream_url}/{path}",
                            params=request.query_params,
                            json=payload if payload else None,
                            content=None if payload else body,
                            headers=_forward_headers(request.headers),
                        )

                upstream_task = asyncio.ensure_future(_upstream_call())
                disconnect_task = asyncio.ensure_future(_watch_disconnect(request))
                try:
                    await asyncio.wait({upstream_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
                    if not upstream_task.done():
                        upstream_task.cancel()
                        with contextlib.suppress(BaseException):
                            await upstream_task
                        if rid is not None:
                            await _abort_upstream(rid)
                        logger.info(
                            "sidecar_proxy client_disconnect request_id=%s path=%s rid=%s elapsed=%.2fs",
                            request_id,
                            path,
                            rid,
                            asyncio.get_running_loop().time() - started,
                        )
                        return Response(status_code=499)
                finally:
                    disconnect_task.cancel()
                    with contextlib.suppress(BaseException):
                        await disconnect_task

                try:
                    resp = upstream_task.result()
                    content_type = resp.headers.get("content-type", "")
                    if "application/json" not in content_type:
                        return Response(
                            content=resp.content,
                            status_code=resp.status_code,
                            media_type=content_type or None,
                        )
                    data = resp.json()
                except Exception:
                    if versioned_route:
                        logger.exception(
                            "sidecar_proxy error request_id=%s path=%s elapsed=%.2fs current=%s active=%s",
                            request_id,
                            path,
                            asyncio.get_running_loop().time() - started,
                            manager.current_version,
                            manager.active_requests,
                        )
                    raise
                if versioned_route and manager.debug_requests:
                    logger.info(
                        "sidecar_proxy end request_id=%s path=%s status=%s elapsed=%.2fs current=%s active=%s",
                        request_id,
                        path,
                        resp.status_code,
                        asyncio.get_running_loop().time() - started,
                        manager.current_version,
                        manager.active_requests,
                    )
                # Captured while the request is still pinned, so a commit
                # cannot advance the version between serving and reporting.
                end_version = manager.current_version
        except PolicyViolation as exc:
            logger.info(
                "sidecar_proxy reject request_id=%s path=%s current=%s error=%s",
                request_id,
                path,
                manager.current_version,
                exc.error,
            )
            return JSONResponse(exc.error, status_code=409)

        if path == "generate" and isinstance(data, dict):
            meta = data.setdefault("meta_info", {})
            meta["weight_version"] = str(start_version)
            meta["weight_version_start"] = start_version
            meta["weight_version_end"] = end_version
        elif path == "v1/chat/completions" and isinstance(data, dict):
            data["weight_version_start"] = start_version
            data["weight_version_end"] = end_version
        return JSONResponse(data, status_code=resp.status_code)

    return app


def build_manager(
    *,
    upstream_url: str,
    bulletin_root: str,
    volume_name: str = "",
    run_id: str | None = None,
    debug_requests: bool = False,
) -> WeightSyncManager:
    refresh = None
    if volume_name:
        from stitch.providers.modal import volume_reloader

        refresh = volume_reloader(volume_name)
    board = FilesystemBulletinBoard(bulletin_root, refresh=refresh)
    engine = SGLangDiskDeltaAdapter(upstream_url=upstream_url)
    return WeightSyncManager(board=board, engine=engine, run_id=run_id, debug_requests=debug_requests)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--bulletin-root", default=os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin"))
    parser.add_argument("--volume-name", default=os.environ.get("DELTA_VOLUME_NAME", ""))
    parser.add_argument("--run-id", default=os.environ.get("DISAGG_RUN_ID"))
    parser.add_argument(
        "--debug-requests",
        action="store_true",
        default=os.environ.get("SIDECAR_DEBUG_REQUESTS", "").lower() in {"1", "true", "yes"},
        help="Log every versioned sidecar proxy request at INFO level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    import uvicorn

    manager = build_manager(
        upstream_url=args.upstream_url,
        bulletin_root=args.bulletin_root,
        volume_name=args.volume_name,
        run_id=args.run_id,
        debug_requests=args.debug_requests,
    )
    uvicorn.run(
        create_app(manager, upstream_url=args.upstream_url),
        host=args.host,
        port=args.port,
        log_level="info",
    )


def _forward_headers(headers: Any) -> dict[str, str]:
    blocked = {"host", "content-length", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in blocked}


if __name__ == "__main__":
    main()
