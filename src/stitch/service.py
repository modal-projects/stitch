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
import os
import re
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
from stitch.types import PoolState, ReplicaState, VersionConstraint, VersionRef
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

# Set to trace each versioned request through the proxy by its rid (see create_app).
TRACE_REQUESTS_ENV = "STITCH_TRACE_REQUESTS"
# Request headers a trace logs the values of: ids a gateway sets to name a request.
_TRACE_ID_HEADER = re.compile(
    r"(request|trace|correlation|span|call)[-_]?u?u?id$|^traceparent$"
)
_SECRET_HEADER = re.compile(r"key|secret|token|auth|cookie")


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


def _render_json_response(
    content: bytes,
    engine: Engine,
    served: VersionRef | None,
    current: VersionRef | None,
) -> bytes:
    data = json.loads(content)
    if isinstance(data, dict) and served is not None and current is not None:
        engine.stamp_response(data, served, current)
    return json.dumps(
        data,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def create_app(
    gate: AdmissionGate,
    status: SidecarStatus,
    engine: Engine,
    *,
    versioned_routes: Iterable[str] = VERSIONED_ROUTES,
    upstream_timeout: float | None = 3600.0,
    proxy_max_connections: int = 100,
    proxy_max_keepalive_connections: int = 20,
    response_keepalive_after: float | None = None,
    response_keepalive_interval: float = 20.0,
    trace_requests: bool | None = None,
):
    """The versioned rollout proxy. Versioned routes are admitted through the gate
    (constraint enforced, serving version captured), stamped by the engine, forwarded,
    and the response stamped with the served version. A rejected constraint returns a
    retryable 409; a client disconnect aborts the upstream generation. A local-engine
    transport failure returns a retryable 503 instead of escaping as a sidecar 500.

    With ``response_keepalive_after`` set, a request the engine has not answered by then
    gets its 200 at once and a whitespace byte every ``response_keepalive_interval``
    seconds until the stamped body follows, so gateways that drop silent responses
    deliver it.

    ``trace_requests`` (default: whether ``STITCH_TRACE_REQUESTS`` is set) logs one
    ``trace`` line per hop of each versioned request, keyed by its rid: received,
    admitted, kept alive, upstream done or failed, client disconnected, responded."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    engine_url = engine.base_url().rstrip("/")
    blocked = engine.blocked_routes()
    timeout = httpx.Timeout(upstream_timeout, connect=10.0)
    versioned = {r.strip("/") for r in versioned_routes}
    pooled: dict[str, Any] = {}
    releasing: set[asyncio.Task] = set()
    if trace_requests is None:
        trace_requests = os.environ.get(TRACE_REQUESTS_ENV, "") not in ("", "0")
    traced_header_names: set[str] = set()

    def trace(event: str, rid: str, start: float, detail: str = "") -> None:
        logger.info(
            "trace %s rid=%s after=%.1f%s",
            event,
            rid,
            time.monotonic() - start,
            f" {detail}" if detail else "",
        )

    def trace_received(rid: str, route: str, size: int, headers: Any) -> None:
        names = sorted(name.lower() for name in headers.keys())
        if set(names) - traced_header_names:
            # Names only, once per new set: values may carry credentials.
            traced_header_names.update(names)
            logger.info("trace headers names=%s", ",".join(names))
        ids = {
            name: value
            for name, value in headers.items()
            if _TRACE_ID_HEADER.search(name.lower())
            and not _SECRET_HEADER.search(name.lower())
        }
        logger.info("trace recv rid=%s route=%s bytes=%d ids=%s", rid, route, size, ids)

    def client() -> Any:
        c = pooled.get("client")
        if c is None:
            c = httpx.AsyncClient(
                timeout=timeout,
                trust_env=False,
                limits=httpx.Limits(
                    max_connections=proxy_max_connections,
                    max_keepalive_connections=proxy_max_keepalive_connections,
                ),
            )
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
        traced = trace_requests and isinstance(rid, str)
        start = time.monotonic()
        if traced:
            trace_received(rid, route, len(body), request.headers)

        headers = {
            k: v for k, v in request.headers.items() if k.lower() not in _DROP_HEADERS
        }

        # Metrics may be scraped before the first pointer exists. Keep the exporter
        # available without making scrapes participate in weight commits. The lease is
        # held on a stack so a kept-alive response can carry it past this handler.
        lease = contextlib.AsyncExitStack()
        try:
            served = await lease.enter_async_context(
                contextlib.nullcontext()
                if request.method == "GET" and route == "metrics"
                else gate.admit(constraint if is_versioned else None)
            )
            if traced:
                trace("admitted", rid, start)
        except ConstraintUnmet as exc:
            if traced:
                trace("rejected", rid, start)
            return JSONResponse(exc.error, status_code=409)
        handed_off = False
        try:
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
                    timeout=response_keepalive_after if is_versioned else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task.done() and not upstream_task.done():
                    upstream_task.cancel()
                    with contextlib.suppress(BaseException):
                        await upstream_task
                    if traced:
                        trace("disconnect", rid, start)
                    if rid is not None:
                        await _abort(client(), engine_url, rid)
                    return Response(status_code=499)
            finally:
                disconnect_task.cancel()
                with contextlib.suppress(BaseException):
                    await disconnect_task

            if not upstream_task.done():
                # A long generation: start the response now and keep bytes flowing, or
                # a gateway on the way back drops it (Modal Flash drops a silent POST
                # response after ~6 min, and a Modal web endpoint answers 303 after
                # 150 s). The stream owns the admission lease until the engine answers.
                handed_off = True
                if traced:
                    trace("keepalive", rid, start)
                release = _releaser(lease, upstream_task, rid)
                return _KeptAliveResponse(
                    _kept_alive(release, upstream_task, route, rid, served),
                    release,
                )

            outcome = await _engine_outcome(upstream_task, request.method, route, rid)
            if isinstance(outcome, Response):
                if traced:
                    trace("upstream_error", rid, start)
                return outcome
            resp = outcome
            if traced:
                trace(
                    "upstream",
                    rid,
                    start,
                    f"status={resp.status_code} bytes={len(resp.content)}",
                )
            if "application/json" not in resp.headers.get("content-type", ""):
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type") or None,
                )
            current = (
                status.applied
            )  # capture while still pinned, before a commit advances it
        finally:
            if not handed_off:
                await lease.aclose()

        body = await asyncio.to_thread(
            _render_json_response,
            resp.content,
            engine,
            served if is_versioned else None,
            current if is_versioned else None,
        )
        if traced:
            trace("respond", rid, start, f"status={resp.status_code} bytes={len(body)}")
        return Response(
            content=body,
            status_code=resp.status_code,
            media_type="application/json",
        )

    async def _engine_outcome(
        upstream_task: asyncio.Future, method: str, route: str, rid: str | None
    ) -> Any:
        """The engine's response, or the retryable 503 for a failed local request."""
        try:
            return upstream_task.result()
        except httpx.RequestError as exc:
            # The sidecar is still healthy when its colocated engine exits, wedges, or
            # drops a connection. Surface that distinction to the pool so another replica
            # can take the retry. A failed read can leave generation alive upstream, so
            # retain the admission lease until the best-effort abort has completed.
            logger.warning(
                "local engine request failed method=%s route=/%s rid=%s error=%s: %s",
                method,
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

    def _releaser(
        lease: contextlib.AsyncExitStack, upstream_task: asyncio.Future, rid: str | None
    ) -> Any:
        """Ends a kept-alive request once, in its own task: aborts the generation if the
        engine has not answered, then releases the admission lease. A disconnecting client
        cancels the stream's task, so cleanup awaited there could be cut short and leak
        the lease; a separate task cannot be."""
        started: list[asyncio.Task] = []

        async def cleanup() -> None:
            if not upstream_task.done():
                upstream_task.cancel()
                with contextlib.suppress(BaseException):
                    await upstream_task
                if rid is not None:
                    await _abort(client(), engine_url, rid)
            await lease.aclose()

        async def release() -> None:
            if not started:
                task = asyncio.ensure_future(cleanup())
                releasing.add(task)
                task.add_done_callback(releasing.discard)
                started.append(task)
            await asyncio.shield(started[0])

        return release

    class _KeptAliveResponse(StreamingResponse):
        """A 200 whose body arrives later. Releases the request however the stream
        ends: completion, client disconnect, or a failed send."""

        def __init__(self, content: Any, release: Any) -> None:
            super().__init__(content, status_code=200, media_type="application/json")
            self._release = release

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                await self._release()

    async def _kept_alive(
        release: Any,
        upstream_task: asyncio.Future,
        route: str,
        rid: str | None,
        served: Any,
    ) -> Any:
        """Whitespace every ``response_keepalive_interval`` seconds until the engine
        answers, then its body, stamped as an immediate response would be. JSON allows
        leading whitespace, so a client parses the same document. The status is already
        200, so an engine error arrives as its error body (the session server rejects a
        body without choices)."""
        yield b" "
        while not upstream_task.done():
            await asyncio.wait({upstream_task}, timeout=response_keepalive_interval)
            if not upstream_task.done():
                yield b" "
        outcome = await _engine_outcome(upstream_task, "POST", route, rid)
        if isinstance(outcome, Response):
            await release()
            yield outcome.body
            return
        if outcome.status_code != 200:
            logger.warning(
                "local engine answered %d after the response started route=/%s rid=%s",
                outcome.status_code,
                route,
                rid,
            )
        current = (
            status.applied
        )  # capture while still pinned, before a commit advances it
        await release()
        if "application/json" not in outcome.headers.get("content-type", ""):
            yield outcome.content
            return
        yield await asyncio.to_thread(
            _render_json_response, outcome.content, engine, served, current
        )

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
    proxy_max_connections: int = 100,
    proxy_max_keepalive_connections: int = 20,
    response_keepalive_after: float | None = None,
    response_keepalive_interval: float = 20.0,
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
            health_observable=reconciler.engine_health_observable,
            interval=watchdog_interval,
            failure_threshold=watchdog_failure_threshold,
        ),
        TerminalFailureMonitor(reconciler.wait_for_terminal_error),
    )
    config = uvicorn.Config(
        create_app(
            reconciler.gate,
            reconciler,
            engine,
            proxy_max_connections=proxy_max_connections,
            proxy_max_keepalive_connections=proxy_max_keepalive_connections,
            response_keepalive_after=response_keepalive_after,
            response_keepalive_interval=response_keepalive_interval,
        ),
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
    latest: VersionRef | None = None,
) -> bool:
    """Block until the configured fraction of ``replica_floor`` reports routing readiness.

    Readiness is counted from each replica's ``/server_info`` rather than inferred from the
    pool gateway: one healthy replica makes a gateway probe succeed, but is not enough capacity
    for a large rollout launch. Once the floor is met, the gateway is checked as well so the
    trainer sees a working traffic path. Timeout is terminal so training never starts below the
    requested floor. This launch-script helper is synchronous, unlike :func:`readiness`.

    ``latest`` is the pointer the caller has just claimed: a replica applied past it on
    the same run is exiting for replacement, so it is not capacity even while its own
    reconcile pass has yet to notice the rewind.
    """
    if replica_floor < 1:
        raise ValueError(f"replica_floor must be positive, got {replica_floor}")
    min_ready = ceil(POOL_READY_FRACTION * replica_floor)

    def is_capacity(replica: ReplicaState) -> bool:
        if (
            latest is not None
            and replica.applied is not None
            and replica.applied.run_id == latest.run_id
            and replica.applied.version > latest.version
        ):
            return False
        return replica.ready

    async def wait() -> bool:
        import httpx

        deadline = time.monotonic() + timeout
        last_counts: tuple[int, int] | None = None
        while time.monotonic() < deadline:
            state = await readiness(pool)
            tally = (
                sum(is_capacity(replica) for replica in state.replicas),
                len(state.replicas),
            )
            if tally != last_counts:
                print(
                    f"Rollout fleet readiness: {tally[0]}/{min_ready} required "
                    f"({tally[1]} discovered)",
                    flush=True,
                )
                last_counts = tally
            if tally[0] >= min_ready:
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
