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
import json
import logging
import time
import urllib.request
import uuid
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

from stitch.engines.base import Engine
from stitch.pools.base import Pool
from stitch.stores.base import Store
from stitch.sync import CommitMode, ConstraintUnmet, Reconciler
from stitch.types import PoolState, ReplicaState, SyncState, VersionConstraint

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
    retryable 409; a client disconnect aborts the upstream generation. A local-engine
    transport failure returns a retryable 503 instead of escaping as a sidecar 500."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

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
        # Background reconcile so uvicorn answers /health (503 until the first catch-up) while it runs.
        syncing = asyncio.create_task(reconciler.startup())
        try:
            yield
        finally:
            syncing.cancel()
            with contextlib.suppress(BaseException):
                await syncing
            await reconciler.shutdown()
            c = pooled.pop("client", None)
            if c is not None:
                await c.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> Response:
        # 503 until the reconciler's first catch-up, so a deployment that gates routing on readiness
        # keeps a not-yet-synced replica out of rotation (else it's routed to and 409s the whole
        # catch-up). A fresh boot clears at once; a mid-run joiner waits until it has applied the
        # live version. Liveness/boot checks use /server_info instead.
        if not reconciler.ready:
            return JSONResponse(
                {"ready": False, "reason": reconciler.readiness_reason()},
                status_code=503,
            )
        return JSONResponse({"ready": True})

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
                {
                    "error": {
                        "type": "RouteBlocked",
                        "message": f"/{route} is managed by the sidecar",
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
            payload = parsed if isinstance(parsed, dict) else None

        is_versioned = route in versioned
        constraint = (
            VersionConstraint.from_payload(payload)
            if is_versioned
            else VersionConstraint()
        )

        # rid lets us abort the upstream generation on client disconnect, else it holds the quiesce point.
        rid = None
        if is_versioned and payload is not None:
            payload.pop("weight_version", None)
            rid = payload.setdefault("rid", uuid.uuid4().hex)

        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _DROP_HEADERS
        }

        # Metrics may be scraped before the first pointer exists. Keep the exporter
        # available without making scrapes participate in weight commits.
        try:
            async with (
                contextlib.nullcontext()
                if request.method == "GET" and route == "metrics"
                else reconciler.admit(constraint if is_versioned else None)
            ) as served:
                if is_versioned and payload is not None and served is not None:
                    engine.stamp_request(payload, served)
                kwargs: dict[str, Any] = {
                    "params": request.query_params,
                    "headers": headers,
                }
                kwargs["json" if payload is not None else "content"] = (
                    payload if payload is not None else body
                )

                upstream_task = asyncio.ensure_future(
                    client().request(request.method, f"{engine_url}/{path}", **kwargs)
                )
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
                            await _abort(client(), engine_url, rid)
                        return Response(status_code=499)
                finally:
                    disconnect_task.cancel()
                    with contextlib.suppress(BaseException):
                        await disconnect_task

                try:
                    resp = upstream_task.result()
                except httpx.RequestError as exc:
                    # The sidecar is still healthy when its colocated engine exits, wedges, or
                    # drops a connection. Surface that distinction to the pool so another replica
                    # can take the retry. A failed read can leave generation alive upstream, so
                    # retain the admission lease until the best-effort abort has completed.
                    logger.warning(
                        "local engine request failed method=%s route=/%s rid=%s error=%s: %s",
                        request.method,
                        route,
                        rid,
                        type(exc).__name__,
                        exc,
                    )
                    if rid is not None:
                        await _abort(client(), engine_url, rid)
                    return JSONResponse(
                        {
                            "error": {
                                "type": "EngineUnavailable",
                                "message": "the local inference engine is unavailable",
                                "retryable": True,
                            }
                        },
                        status_code=503,
                        headers={"Retry-After": "1"},
                    )
                if "application/json" not in resp.headers.get("content-type", ""):
                    return Response(
                        content=resp.content,
                        status_code=resp.status_code,
                        media_type=resp.headers.get("content-type") or None,
                    )
                data = resp.json()
                current = (
                    reconciler.applied
                )  # capture while still pinned, before a commit advances it
        except ConstraintUnmet as exc:
            return JSONResponse(exc.error, status_code=409)

        if (
            is_versioned
            and isinstance(data, dict)
            and served is not None
            and current is not None
        ):
            engine.stamp_response(data, served, current)
        return JSONResponse(data, status_code=resp.status_code)

    return app


async def _abort(client: Any, engine_url: str, rid: str) -> None:
    try:
        await client.request(
            "POST", f"{engine_url}/abort_request", json={"rid": rid}, timeout=10.0
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "failed to abort upstream rid=%s error=%s: %s",
            rid,
            type(exc).__name__,
            exc,
        )


def serve(
    store: Store,
    engine: Engine,
    *,
    run_id: str,
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
        store=store,
        engine=engine,
        run_id=run_id,
        commit_mode=commit_mode,
        flush_cache_on_commit=flush_cache_on_commit,
        debug_requests=debug_requests,
        reconcile_interval=reconcile_interval,
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
            return ReplicaState(
                reason=str(exc)[:80]
            )  # applied=None => counts as not at any version

    async with httpx.AsyncClient(trust_env=False) as c:
        # the async variant keeps pool-client I/O off this event loop (native or threaded per pool)
        replicas = await pool.discover_replicas_async()
        states = await asyncio.gather(*(probe(c, url) for url in replicas))
    return PoolState(list(states))


# The reconciler states in which the engine is legitimately unresponsive: it is
# staging or loading weights, so a stale health heartbeat is expected.
_SYNCING_STATES = {
    SyncState.STAGING.value,
    SyncState.COMMITTING.value,
}


def sync_in_progress(server_info_url: str, *, timeout: float = 5.0) -> bool:
    """Whether the replica is initializing, staging, or committing weights.

    A deployment's engine-health probe uses this to suppress expected health
    blips during weight work. An unreachable sidecar returns ``False`` so the
    caller still reports an actual failure.
    """
    try:
        with urllib.request.urlopen(server_info_url, timeout=timeout) as resp:
            info = json.loads(resp.read())
    except Exception:  # noqa: BLE001
        return False
    initializing = not info.get("update_destination_ready", True) and not info.get(
        "update_destination_error"
    )
    return bool(initializing or info.get("sync_state") in _SYNCING_STATES)


def await_pool_ready(
    pool: Pool, *, timeout: float = 20 * 60, interval: float = 30.0
) -> bool:
    """Block until the pool's gateway answers /health 200 — Flash holds requests through a
    cold-starting pool, so the first rollout/hot-load meets a ready pool instead of a 5xx storm
    while engines load. Returns True when ready; on timeout, warns and returns False (the caller
    proceeds anyway — the trainer retries). A launch-script helper: synchronous, unlike ``readiness``."""
    import httpx

    gateway = pool.gateway_url().rstrip("/")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{gateway}/health", timeout=10).status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(interval)
    print(
        f"WARNING: pool at {gateway} not ready after {timeout:.0f}s; proceeding (the trainer retries)"
    )
    return False
