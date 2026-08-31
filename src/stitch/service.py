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
import time
import uuid
from collections.abc import Iterable
from contextlib import asynccontextmanager
from math import ceil
from typing import Any, Protocol

from stitch.engines.base import Engine
from stitch.pools.base import Pool
from stitch.stores.base import Store
from stitch.sync import AdmissionGate, CommitMode, ConstraintUnmet, Reconciler
from stitch.types import PoolState, ReplicaState, VersionConstraint
from stitch.watchdog import (
    EngineWatchdog,
    SidecarWatchdog,
    TerminalFailureMonitor,
    run_server_with_watchdog,
)

logger = logging.getLogger(__name__)

VERSIONED_ROUTES = ("generate", "v1/chat/completions", "v1/completions")

# Temporary launch threshold; tune as we collect fleet startup and throughput data.
POOL_READY_FRACTION = 0.75

# Hop-by-hop / rewritten headers the proxy never forwards upstream.
_DROP_HEADERS = {"host", "content-length", "connection"}


class SidecarStatus(Protocol):
    """The status/control surface the proxy consumes besides admission — the reconciler
    implements it; tests stand it in with a stub. Admission flows through the
    ``AdmissionGate`` separately, so the data plane never needs the control loop."""

    @property
    def ready(self) -> bool: ...

    @property
    def applied(self) -> Any: ...

    def readiness_reason(self) -> str: ...

    def server_info(self) -> dict[str, Any]: ...

    def wake(self) -> None: ...

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...


def create_app(
    gate: AdmissionGate,
    status: SidecarStatus,
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
        # Run startup beside the HTTP loop so /health explains why the replica is
        # not yet admitted while destination initialization and catch-up run.
        syncing = asyncio.create_task(status.startup())
        try:
            yield
        finally:
            syncing.cancel()
            with contextlib.suppress(BaseException):
                await syncing
            await status.shutdown()
            c = pooled.pop("client", None)
            if c is not None:
                await c.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> Response:
        # 503 until destination initialization and first catch-up complete. This
        # is the routing-readiness contract; liveness and details use /server_info.
        if not status.ready:
            return JSONResponse(
                {"ready": False, "reason": status.readiness_reason()},
                status_code=503,
            )
        return JSONResponse({"ready": True})

    @app.get("/server_info")
    async def server_info() -> dict[str, Any]:
        return status.server_info()

    @app.post("/wake")
    async def wake() -> dict[str, Any]:
        status.wake()
        return status.server_info()

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
                else gate.admit(constraint if is_versioned else None)
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
                    status.applied
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
    boot_version: int = 0,
    commit_mode: CommitMode = "in_place",
    flush_cache_on_commit: bool = False,
    host: str = "0.0.0.0",
    port: int = 8000,
    debug_requests: bool = False,
    reconcile_interval: float = 5.0,
    watchdog_interval: float = 5.0,
    watchdog_failure_threshold: int = 3,
) -> None:
    """Run one replica's sidecar: build the Reconciler over the given store+engine
    and serve the versioned proxy. The deployment supplies the concrete instances."""
    import uvicorn

    reconciler = Reconciler(
        store=store,
        engine=engine,
        run_id=run_id,
        boot_version=boot_version,
        commit_mode=commit_mode,
        flush_cache_on_commit=flush_cache_on_commit,
        debug_requests=debug_requests,
        reconcile_interval=reconcile_interval,
    )
    watchdog = SidecarWatchdog(
        EngineWatchdog(
            engine,
            expects_engine_progress=reconciler.expects_engine_progress,
            interval=watchdog_interval,
            failure_threshold=watchdog_failure_threshold,
        ),
        TerminalFailureMonitor(reconciler.wait_for_terminal_error),
    )
    config = uvicorn.Config(
        create_app(reconciler.gate, reconciler, engine),
        host=host,
        port=port,
        log_level="info",
    )
    asyncio.run(run_server_with_watchdog(uvicorn.Server(config), watchdog))


async def readiness(pool: Pool, *, timeout: float = 15.0) -> PoolState:
    """Aggregate every replica's ``/server_info`` into a PoolState (drives the readiness
    poll and the smoke check). A replica that fails to answer counts as not ready."""
    import httpx

    async def probe(c: Any, url: str) -> ReplicaState:
        try:
            target, headers = await asyncio.to_thread(
                pool.replica_request, url, "/server_info"
            )
            resp = await c.get(target, headers=headers, timeout=timeout)
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


def await_pool_ready(
    pool: Pool,
    *,
    replica_floor: int,
    timeout: float = 60 * 60,
    interval: float = 30.0,
) -> bool:
    """Block until the configured fraction of ``replica_floor`` reports routing readiness.

    Readiness is counted from each replica's ``/server_info`` rather than inferred from the
    pool gateway: one healthy replica makes a gateway probe succeed, but is not enough capacity
    for a large rollout launch. Once the floor is met, the gateway is checked as well so the
    trainer sees a working traffic path. Timeout is terminal so training never starts below the
    requested floor. This launch-script helper is synchronous, unlike :func:`readiness`.
    """
    if replica_floor < 1:
        raise ValueError(f"replica_floor must be positive, got {replica_floor}")
    min_ready = ceil(POOL_READY_FRACTION * replica_floor)

    async def wait() -> bool:
        import httpx

        deadline = time.monotonic() + timeout
        last_counts: tuple[int, int] | None = None
        while time.monotonic() < deadline:
            state = await readiness(pool)
            counts = (
                sum(replica.ready for replica in state.replicas),
                len(state.replicas),
            )
            if counts != last_counts:
                print(
                    f"Rollout fleet readiness: {counts[0]}/{min_ready} required "
                    f"({counts[1]} discovered)",
                    flush=True,
                )
                last_counts = counts
            if counts[0] >= min_ready:
                try:
                    gateway = (await pool.gateway_url_async()).rstrip("/")
                    async with httpx.AsyncClient(trust_env=False) as client:
                        response = await client.get(f"{gateway}/health", timeout=10)
                    if response.status_code == 200:
                        return True
                except Exception:  # noqa: BLE001
                    pass
            await asyncio.sleep(interval)

        ready, discovered = last_counts or (0, 0)
        raise TimeoutError(
            f"rollout pool not ready after {timeout:.0f}s: "
            f"{ready}/{min_ready} required ({discovered} discovered)"
        )

    return asyncio.run(wait())
