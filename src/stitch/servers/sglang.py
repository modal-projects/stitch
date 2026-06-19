"""HTTP sidecar that adds weight-version protocol semantics to SGLang."""

import asyncio
import contextlib
import inspect
import logging
import uuid
from collections.abc import Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any

from stitch.engines.sglang import compose_extra_key
from stitch.protocol import WeightVersionPolicy
from stitch.sync import PolicyViolation, RolloutSyncManager


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
    manager: RolloutSyncManager,
    *,
    upstream_url: str,
    versioned_routes: Iterable[str] = ("generate", "v1/chat/completions"),
    register_routes: Callable[[Any], None] | None = None,
    include_sync_routes: bool = True,
    upstream_timeout: float | None = 3600.0,
    background_sync_interval: float | None = None,
):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    import httpx

    upstream_url = upstream_url.rstrip("/")
    # Bound the wait for an upstream (SGLang) response. A generation that
    # finishes but never delivers its HTTP body would otherwise hang this proxy
    # forever (timeout=None), holding the request open and wedging the client
    # awaiting it — exactly the failure mode that stalled a rollout for hours.
    # On timeout the upstream call raises, surfacing as a 5xx the client can
    # retry, instead of an infinite hold. connect stays short (upstream is
    # localhost); pass None to opt out.
    upstream_request_timeout = httpx.Timeout(upstream_timeout, connect=10.0)
    versioned_route_set = {route.strip("/") for route in versioned_routes}

    # One pooled upstream client for the whole process. A rollout proxy that
    # reconnects to the engine on every request pays a TCP/pool setup per hop;
    # reusing the client keeps connections warm across the sidecar->engine hop
    # that carries every rollout. Created lazily on the first proxied request
    # (so request-time test patching of httpx.AsyncClient still wins) and closed
    # on shutdown.
    pooled: dict[str, Any] = {}

    def upstream_client() -> Any:
        client = pooled.get("client")
        if client is None:
            client = httpx.AsyncClient(timeout=upstream_request_timeout, trust_env=False)
            pooled["client"] = client
        return client

    async def _reconcile_loop(interval: float) -> None:
        # Pull-based reconcile against the bulletin board's `latest` pointer.
        # In the log-as-truth deployment the front door advances `latest` and
        # the pool catches up here, so a replica that missed a wake (or scaled
        # up after one) still converges without any request-version pin.
        board = getattr(manager, "board", None)
        queue_sync = getattr(manager, "queue_sync", None)
        if board is None or queue_sync is None:
            return
        while True:
            await asyncio.sleep(interval)
            try:
                await board.refresh()
                queue_sync()
            except Exception:  # noqa: BLE001
                logger.warning("background reconcile failed", exc_info=True)

    reconcile: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await manager.startup_sync()
        if background_sync_interval and background_sync_interval > 0:
            reconcile["task"] = asyncio.ensure_future(_reconcile_loop(background_sync_interval))
        try:
            yield
        finally:
            task = reconcile.pop("task", None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
            client = pooled.pop("client", None)
            if client is not None:
                await client.aclose()
            shutdown_sync = getattr(manager, "shutdown_sync", None)
            if shutdown_sync is not None:
                result = shutdown_sync()
                if inspect.isawaitable(result):
                    await result

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

    if include_sync_routes:

        @app.post("/rpc_sync_from_bulletin_board")
        async def rpc_sync_from_bulletin_board(request: Request) -> dict[str, Any]:
            payload = await request.json()
            target = payload.get("target_version")
            manager.queue_sync(int(target) if target is not None else None)
            return {
                "accepted": True,
                "current_version": manager.current_version,
                "queued_target_version": getattr(
                    manager, "queued_target_version", None
                ),
                "sync_state": _sync_state_value(getattr(manager, "sync_state", None)),
            }

    if register_routes is not None:
        register_routes(app)

    async def _watch_disconnect(request: Request) -> None:
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return

    async def _abort_upstream(rid: str) -> None:
        try:
            await upstream_client().request(
                "POST", f"{upstream_url}/abort_request", json={"rid": rid}, timeout=10.0
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "sidecar_proxy failed to abort upstream rid=%s", rid, exc_info=True
            )

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
        payload: dict[str, Any] | None = None
        if body and request.headers.get("content-type", "").startswith(
            "application/json"
        ):
            parsed = await request.json()
            if isinstance(parsed, dict):
                payload = parsed

        versioned_route = route in versioned_route_set
        policy = (
            WeightVersionPolicy.from_payload(payload)
            if versioned_route
            else WeightVersionPolicy()
        )
        request_id = request.headers.get("x-slime-request-id", "-")

        forward_headers = _forward_headers(request.headers)

        rid: str | None = None
        if versioned_route and payload is not None:
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
                if versioned_route and payload is not None:
                    # Stamp the serving version into the engine's KV cache
                    # namespace: requests admitted under different versions
                    # structurally cannot share radix-tree prefixes.
                    user_key = payload.get("extra_key")
                    if isinstance(user_key, list):
                        payload["extra_key"] = [
                            compose_extra_key(start_version, k) for k in user_key
                        ]
                    else:
                        payload["extra_key"] = compose_extra_key(
                            start_version, user_key
                        )
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
                    request_kwargs: dict[str, Any] = {
                        "params": request.query_params,
                        "headers": forward_headers,
                    }
                    if payload is not None:
                        request_kwargs["json"] = payload
                    else:
                        request_kwargs["content"] = body
                    return await upstream_client().request(
                        request.method,
                        f"{upstream_url}/{path}",
                        **request_kwargs,
                    )

                upstream_task = asyncio.ensure_future(_upstream_call())
                disconnect_task = asyncio.ensure_future(_watch_disconnect(request))
                try:
                    await asyncio.wait(
                        {upstream_task, disconnect_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
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

        if versioned_route and isinstance(data, dict):
            # Driven by the same `versioned_route` flag that gated and stamped the
            # request, so injection can't diverge from gating (previously a fixed
            # path list here meant /v1/completions got version metadata while
            # going ungated, and a custom versioned route got gated but no
            # metadata). /generate carries it in meta_info; OpenAI-style routes
            # at the top level.
            if route == "generate":
                meta = data.setdefault("meta_info", {})
                meta["weight_version"] = str(start_version)
                meta["weight_version_start"] = start_version
                meta["weight_version_end"] = end_version
            else:
                data["weight_version_start"] = start_version
                data["weight_version_end"] = end_version
        return JSONResponse(data, status_code=resp.status_code)

    return app


def _forward_headers(headers: Any) -> dict[str, str]:
    blocked = {"host", "content-length", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in blocked}


def _sync_state_value(sync_state: Any) -> Any:
    return getattr(sync_state, "value", sync_state)
