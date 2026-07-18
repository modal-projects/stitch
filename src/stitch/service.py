"""The rollout-service runtime: the versioned proxy (``create_app``), the sidecar
entrypoint (``serve``), and cross-replica readiness aggregation (``readiness``).

Engine- and provider-agnostic: request/response version stamping is delegated to the
Engine, and the proxy forwards everything else to the engine's own HTTP surface.

No ``from __future__ import annotations`` here: the FastAPI route handlers below are
introspected at runtime, and their ``Request`` type is a create_app-local import — under
stringized annotations FastAPI can't resolve it (it looks only in module globals) and
demotes ``request`` to a required query param, 422-ing every call.
"""

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

from stitch.engines.base import Engine
from stitch.pools.base import Pool
from stitch.stores.base import Store
from stitch.sync import CommitMode, ConstraintUnmet, Reconciler
from stitch.types import PoolState, ReplicaState, VersionConstraint

logger = logging.getLogger(__name__)

VERSIONED_ROUTES = ("generate", "v1/chat/completions", "v1/completions")

# Hop-by-hop / rewritten headers the proxy never forwards upstream.
_DROP_HEADERS = {"host", "content-length", "connection"}


def create_app(
    reconciler: Reconciler,
    engine: Engine,
    *,
    versioned_routes: Iterable[str] = VERSIONED_ROUTES,
    upstream_timeout: float | None = 3600.0,
):
    """The versioned rollout proxy. Versioned routes are admitted through the gate
    (constraint enforced, serving version captured), stamped by the engine, forwarded,
    and the response stamped with the served version. A rejected constraint returns a
    retryable 409; a client disconnect aborts the upstream generation."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    import httpx

    engine_url = engine.base_url().rstrip("/")
    blocked = engine.blocked_routes()
    timeout = httpx.Timeout(upstream_timeout, connect=10.0)
    versioned = {r.strip("/") for r in versioned_routes}
    pooled: dict[str, Any] = {}

    def client() -> Any:
        c = pooled.get("client")
        if c is None:
            c = httpx.AsyncClient(timeout=timeout, trust_env=False)
            pooled["client"] = c
        return c

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await reconciler.startup()
        try:
            yield
        finally:
            await reconciler.shutdown()
            c = pooled.pop("client", None)
            if c is not None:
                await c.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/server_info")
    async def server_info() -> dict[str, Any]:
        return reconciler.server_info()

    @app.post("/wake")
    async def wake() -> dict[str, Any]:
        reconciler.wake()
        return reconciler.server_info()

    async def _watch_disconnect(request: Request) -> None:
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(path: str, request: Request) -> Response:
        route = path.strip("/")
        if route in blocked:
            return JSONResponse(
                {"error": {"type": "RouteBlocked", "message": f"/{route} is managed by the sidecar"}},
                status_code=403,
            )

        body = await request.body()
        payload: dict[str, Any] | None = None
        if body and request.headers.get("content-type", "").startswith("application/json"):
            parsed = await request.json()
            payload = parsed if isinstance(parsed, dict) else None

        is_versioned = route in versioned
        constraint = VersionConstraint.from_payload(payload) if is_versioned else VersionConstraint()

        # rid lets us abort the upstream generation on client disconnect, else it holds the quiesce point.
        rid = None
        if is_versioned and payload is not None:
            payload.pop("weight_version", None)
            rid = payload.setdefault("rid", uuid.uuid4().hex)

        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP_HEADERS}

        try:
            async with reconciler.admit(constraint if is_versioned else None) as served:
                if is_versioned and payload is not None and served is not None:
                    engine.stamp_request(payload, served)
                kwargs: dict[str, Any] = {"params": request.query_params, "headers": headers}
                kwargs["json" if payload is not None else "content"] = payload if payload is not None else body

                upstream_task = asyncio.ensure_future(client().request(request.method, f"{engine_url}/{path}", **kwargs))
                disconnect_task = asyncio.ensure_future(_watch_disconnect(request))
                try:
                    await asyncio.wait({upstream_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
                    if not upstream_task.done():
                        upstream_task.cancel()
                        with contextlib.suppress(BaseException):
                            await upstream_task
                        if rid is not None:
                            await _abort(client(), engine_url, rid)
                        return Response(status_code=499)
                finally:
                    disconnect_task.cancel()
                    with contextlib.suppress(BaseException):
                        await disconnect_task

                resp = upstream_task.result()
                if "application/json" not in resp.headers.get("content-type", ""):
                    return Response(content=resp.content, status_code=resp.status_code,
                                    media_type=resp.headers.get("content-type") or None)
                data = resp.json()
                current = reconciler.applied  # capture while still pinned, before a commit advances it
        except ConstraintUnmet as exc:
            return JSONResponse(exc.error, status_code=409)

        if is_versioned and isinstance(data, dict) and served is not None and current is not None:
            engine.stamp_response(data, served, current)
        return JSONResponse(data, status_code=resp.status_code)

    return app


async def _abort(client: Any, engine_url: str, rid: str) -> None:
    try:
        await client.request("POST", f"{engine_url}/abort_request", json={"rid": rid}, timeout=10.0)
    except Exception:  # noqa: BLE001
        logger.warning("failed to abort upstream rid=%s", rid, exc_info=True)


def serve(
    store: Store,
    engine: Engine,
    *,
    run_id: str | None = None,
    commit_mode: CommitMode = "in_place",
    flush_cache_on_commit: bool = False,
    host: str = "0.0.0.0",
    port: int = 8000,
    debug_requests: bool = False,
    reconcile_interval: float = 5.0,
) -> None:
    """Run one replica's sidecar: build the Reconciler over the given store+engine
    and serve the versioned proxy. The deployment supplies the concrete instances."""
    import uvicorn

    reconciler = Reconciler(
        store=store, engine=engine, run_id=run_id, commit_mode=commit_mode,
        flush_cache_on_commit=flush_cache_on_commit,
        debug_requests=debug_requests, reconcile_interval=reconcile_interval,
    )
    uvicorn.run(create_app(reconciler, engine), host=host, port=port, log_level="info")


async def readiness(pool: Pool, *, timeout: float = 15.0) -> PoolState:
    """Aggregate every replica's ``/server_info`` into a PoolState (drives the readiness
    poll and the smoke check). A replica that fails to answer counts as not ready."""
    import httpx

    async def probe(c: Any, url: str) -> ReplicaState:
        try:
            resp = await c.get(f"{url.rstrip('/')}/server_info", timeout=timeout)
            return ReplicaState.from_dict(resp.json())
        except Exception as exc:  # noqa: BLE001
            return ReplicaState(ready=False, reason=str(exc)[:80])

    async with httpx.AsyncClient(trust_env=False) as c:
        states = await asyncio.gather(*(probe(c, url) for url in pool.discover_replicas()))
    return PoolState(list(states))
