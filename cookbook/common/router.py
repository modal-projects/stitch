"""Session-affinity load-balancing router for the rollout pool.

A run-scoped fork of the flash-smart-router (github.com/modal-labs/flash-smart-router),
folded into the run's own Modal app: each recipe deploys a ``RouterRegistry`` and a
``Router`` Flash server class next to its GPU ``Server`` (thin module-level classes whose
lifecycle delegates here, exactly like ``common.server`` does for sglang). One app, one
deploy, one stop — the router scales and dies with the pool.

Why it exists: with a per-container queue bound (sglang ``--max-queued-requests``), Flash's
own sticky routing turns the first saturated replicas into 503 attractors — sessions stuck
on a full replica keep retrying it while the rest of the pool starves. Here, the registry
polls every replica's live queue depth (``/v1/loads`` + a ``/health`` zombie check); the
router pins each ``x-session-affinity`` key to a healthy replica with spare capacity via the
``modal-flash-upstream`` header, and a 503 from a pinned replica evicts it from rotation and
retries on a healthier one, so load spreads instead of sticking.

Deltas from the upstream router: no proxy-auth/api-key secret plumbing (cookbook pools are
``unauthenticated``), replica discovery reuses Stitch's ``ModalFlashPool`` adapter, stdlib
logging, and startup tolerates the sibling classes still cold-booting (single-app deploys
have no boot ordering) instead of dying on the first failed poll. The legacy-tuple
session-route migration is dropped — per-run route dicts start empty.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

import httpx
import modal
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, TypeAdapter

from stitch.pools.modal_flash import list_flash_containers_async

logger = logging.getLogger(__name__)

SESSION_ROUTE_TTL_SECONDS = 4 * 60 * 60
SESSION_ROUTE_MAX_UPSTREAMS = 10
# Flash consumes its reserved Modal-Session-ID before the ASGI app sees the request.
SESSION_AFFINITY_HEADER = "x-session-affinity"

CONTAINER_POLL_INTERVAL_SECONDS = 1.0
CONTAINER_POLL_TIMEOUT_SECONDS = 3.0

MAX_SESSION_RETRIES = 3

_COOKBOOK_DIR = Path(__file__).resolve().parent.parent


def build_router_image(experiment: str, run_id: str) -> modal.Image:
    """CPU image for the router classes. Like the serving image, it bakes the run
    coordinates and carries the cookbook + stitch sources, so a container's re-import of
    the recipe app module rebuilds the same app name and class wiring."""
    return (
        modal.Image.debian_slim()
        .pip_install("fastapi", "httpx", "pydantic", "uvicorn")
        .env({"EXPERIMENT_CONFIG": experiment, "RUN_ID": run_id})
        .add_local_python_source("stitch")
        .add_local_dir(
            str(_COOKBOOK_DIR), remote_path="/root/cookbook", ignore=["**/__pycache__"]
        )
    )


def session_routes_dict(app_name: str) -> modal.Dict:
    """The run's session→replica affinity store, namespaced to the run app."""
    return modal.Dict.from_name(f"{app_name}-session-routes", create_if_missing=True)


# ── routing state ────────────────────────────────────────────────────────────


class ContainerInfo(BaseModel):
    task_id: str
    upstream: str  # host:port of the replica
    load: int  # queued + running requests, last successful poll


ContainerInfoList = TypeAdapter(list[ContainerInfo])


class RouteEntry(BaseModel):
    task_id: str
    last_sent: float


RouteEntryList = TypeAdapter(list[RouteEntry])


def _container_addr(container: Any) -> str | None:
    """``host:port`` for a flash container (proto or dict), tolerating a host that
    already carries its port (the shape ``ModalFlashPool`` reads)."""
    if isinstance(container, dict):
        host, port = container.get("host"), container.get("port")
    else:
        host, port = getattr(container, "host", None), getattr(container, "port", None)
    if not host:
        return None
    host = str(host).rstrip("/")
    if ":" in host or not port:
        return host
    return f"{host}:{port}"


def filter_headers(headers: dict[str, str]) -> dict[str, str]:
    """Drop hop-by-hop and Modal-routing headers before forwarding upstream."""
    removed = {
        "content-length",
        "host",
        "modal-flash-upstream",
        "modal-key",
        "modal-secret",
        "modal-session-id",
        SESSION_AFFINITY_HEADER,
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-prefix",
        "x-forwarded-proto",
        "x-forwarded-server",
    }
    return {k: v for k, v in headers.items() if k.lower() not in removed}


def select_underloaded_container(
    containers: dict[str, ContainerInfo], overload_threshold: int
) -> ContainerInfo:
    """Spread new sessions across replicas with headroom.

    Registry loads are snapshots shared by multiple router replicas. Selecting the exact
    minimum makes every router converge on a newly ready zero-load replica before the next
    snapshot, creating a thundering herd. Random choice from the healthy underloaded set
    preserves load shedding without requiring distributed per-request reservations.
    """
    candidates = [
        container
        for container in containers.values()
        if container.load < overload_threshold
    ]
    return random.choice(candidates or list(containers.values()))


async def route_session(
    session_routes: modal.Dict,
    session_id: str,
    containers: dict[str, ContainerInfo],
    overload_threshold: int,
) -> ContainerInfo:
    """Pick the replica for one session: the most-recently-used of its known replicas
    that has room below the pool's configured soft capacity, else a random replica with
    headroom. Mutates and persists the session's routes."""
    current_time = time.time()

    routes: list[RouteEntry] = RouteEntryList.validate_python(
        await session_routes.get.aio(session_id, [])
    )
    # Drop expired routes and replicas this pod hasn't discovered (they may just be
    # new; every session lands on some pod's view, so this only trims the dead).
    routes = [
        entry
        for entry in routes
        if current_time - entry.last_sent <= SESSION_ROUTE_TTL_SECONDS
        and entry.task_id in containers
    ]
    routes.sort(key=lambda entry: entry.last_sent, reverse=True)

    async def save_routes() -> None:
        await session_routes.put.aio(
            session_id,
            RouteEntryList.dump_python(
                routes[:SESSION_ROUTE_MAX_UPSTREAMS], mode="json"
            ),
        )

    for entry in routes:
        container = containers[entry.task_id]
        if container.load < overload_threshold:
            entry.last_sent = current_time
            await save_routes()
            return container

    selected = select_underloaded_container(containers, overload_threshold)
    reason = (
        f"previous upstreams [{', '.join(entry.task_id for entry in routes)}] overloaded"
        f" (load >= {overload_threshold})"
        if routes
        else "no previous upstream"
    )
    logger.info(
        "session %s: %s; pinning new upstream %s (load %d)",
        session_id,
        reason,
        selected.task_id,
        selected.load,
    )
    routes.insert(0, RouteEntry(task_id=selected.task_id, last_sent=current_time))
    await save_routes()
    return selected


# ── serving plumbing ─────────────────────────────────────────────────────────
class _UvicornApp:
    """Runs a FastAPI app (from ``build_app``) on a uvicorn server in a daemon thread."""

    def build_app(self) -> Any:
        raise NotImplementedError

    def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self.build_app(), host="0.0.0.0", port=8000, timeout_keep_alive=300
        )
        self.uvicorn_server = uvicorn.Server(config)
        self.server_thread = threading.Thread(
            target=self.uvicorn_server.run, daemon=True
        )
        self.server_thread.start()

    def stop(self) -> None:
        self.uvicorn_server.should_exit = True
        self.server_thread.join()


class _RequestCancelledMiddleware:
    """Cancels the request handler (and so the upstream stream) on client disconnect."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> Any:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        queue: asyncio.Queue = asyncio.Queue()

        async def message_poller(handler_task: asyncio.Task) -> None:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    handler_task.cancel()
                    return
                await queue.put(message)

        handler_task = asyncio.create_task(self.app(scope, queue.get, send))
        asyncio.create_task(message_poller(handler_task))  # noqa: RUF006

        try:
            return await handler_task
        except asyncio.CancelledError:
            logger.info("cancelled request: client disconnected")
            return None


def serve_registry(replica: Any, *, app_name: str, upstream_cls: str) -> None:
    """Start the load-registry server on a ``RouterRegistry`` container (@modal.enter)."""
    registry = _RegistryApp(app_name=app_name, upstream_cls=upstream_cls)
    registry.start()
    replica._router_server = registry


def serve_router(
    replica: Any,
    *,
    registry_url: str,
    upstream_url: str,
    session_routes: modal.Dict,
    overload_threshold: int,
) -> None:
    """Start the session-routing proxy on a ``Router`` container (@modal.enter)."""
    router = _ProxyApp(
        registry_url=registry_url,
        upstream_url=upstream_url,
        session_routes=session_routes,
        overload_threshold=overload_threshold,
    )
    router.start()
    replica._router_server = router


def stop_server(replica: Any) -> None:
    """@modal.exit for either router class."""
    server = getattr(replica, "_router_server", None)
    if server is not None:
        server.stop()


class _RegistryApp(_UvicornApp):
    """Polls upstream replicas for load and serves the snapshot at ``/loads``."""

    def __init__(self, *, app_name: str, upstream_cls: str) -> None:
        self.app_name = app_name
        self.upstream_cls = upstream_cls
        self.containers: list[ContainerInfo] = []

    async def _container_load(self, upstream: str) -> int:
        loads_response, health_response = await asyncio.gather(
            self.client.get(f"https://{upstream}/v1/loads", params={"include": "core"}),
            self.client.get(f"https://{upstream}/health"),
        )
        loads_response.raise_for_status()
        health_response.raise_for_status()

        loads = loads_response.json()["loads"]
        queued = sum(int(load["num_waiting_reqs"]) for load in loads)
        running = sum(int(load["num_running_reqs"]) for load in loads)
        return queued + running

    async def _poll_once(self) -> list[ContainerInfo]:
        discovered = await list_flash_containers_async(self.app_name, self.upstream_cls)
        upstreams = {}
        for container in discovered:
            if isinstance(container, dict):
                task_id = container.get("task_id")
            else:
                task_id = getattr(container, "task_id", None)
            addr = _container_addr(container)
            if task_id and addr:
                upstreams[task_id] = addr
        results = await asyncio.gather(
            *(self._container_load(upstream) for upstream in upstreams.values()),
            return_exceptions=True,
        )
        updated: list[ContainerInfo] = []
        for (task_id, upstream), result in zip(upstreams.items(), results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "load poll failed for replica %s: %r; dropping from registry",
                    task_id,
                    result,
                )
                continue
            updated.append(
                ContainerInfo(task_id=task_id, upstream=upstream, load=result)
            )
        return updated

    async def poll_containers(self) -> None:
        await asyncio.sleep(CONTAINER_POLL_INTERVAL_SECONDS)
        while True:
            started_at = time.monotonic()
            try:
                self.containers = await self._poll_once()
            except Exception:
                logger.exception("failed to refresh replica loads")
            elapsed = time.monotonic() - started_at
            await asyncio.sleep(max(0, CONTAINER_POLL_INTERVAL_SECONDS - elapsed))

    def build_app(self) -> Any:
        fastapi_app = FastAPI()
        self.client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(
                retries=3,
                limits=httpx.Limits(
                    max_keepalive_connections=200,
                    max_connections=200,
                    keepalive_expiry=60,
                ),
            ),
            follow_redirects=True,
            timeout=CONTAINER_POLL_TIMEOUT_SECONDS,
        )

        @fastapi_app.on_event("startup")
        async def startup() -> None:
            # No network I/O in startup: bind immediately and let the poll loop fill
            # in (~1s) — a hung startup RPC gets the container restarted by Modal
            # before anything logs (siblings may still be cold-booting anyway).
            self.poller_task = asyncio.create_task(self.poll_containers())
            logger.info("registry startup complete")

        @fastapi_app.on_event("shutdown")
        async def shutdown() -> None:
            self.poller_task.cancel()
            await asyncio.gather(self.poller_task, return_exceptions=True)
            await self.client.aclose()

        @fastapi_app.get("/loads", response_model=list[ContainerInfo])
        async def loads() -> list[ContainerInfo]:
            return self.containers

        return fastapi_app


class _ProxyApp(_UvicornApp):
    """Proxies requests to rollout replicas with session-affinity routing."""

    def __init__(
        self,
        *,
        registry_url: str,
        upstream_url: str,
        session_routes: modal.Dict,
        overload_threshold: int,
    ) -> None:
        self.registry_url = registry_url.rstrip("/")
        self.upstream_url = upstream_url.rstrip("/")
        self.session_routes = session_routes
        self.overload_threshold = overload_threshold
        self.containers: dict[str, ContainerInfo] = {}

    @contextlib.contextmanager
    def count_in_flight(self, container: ContainerInfo) -> Generator[None, None, None]:
        container.load += 1
        try:
            yield
        finally:
            container.load = max(0, container.load - 1)

    async def get_containers(self) -> dict[str, ContainerInfo]:
        response = await self.client.get(f"{self.registry_url}/loads")
        response.raise_for_status()
        containers = ContainerInfoList.validate_json(response.content)
        return {container.task_id: container for container in containers}

    async def poll_containers(self) -> None:
        await asyncio.sleep(CONTAINER_POLL_INTERVAL_SECONDS)
        while True:
            started_at = time.monotonic()
            try:
                self.containers = await self.get_containers()
            except Exception:
                logger.exception("failed to refresh replica loads from registry")
            elapsed = time.monotonic() - started_at
            await asyncio.sleep(max(0, CONTAINER_POLL_INTERVAL_SECONDS - elapsed))

    async def upstream_stream(
        self,
        request: Any,
        path: str,
        headers: dict[str, str],
        body: bytes,
        container: ContainerInfo | None,
        log_prefix: str,
    ) -> AsyncGenerator[Any, None]:
        maybe_count = (
            self.count_in_flight(container) if container else contextlib.nullcontext()
        )
        with maybe_count:
            async with self.client.stream(
                url=f"{self.upstream_url}/{path}",
                method=request.method,
                params=request.query_params,
                headers=headers,
                content=body,
                # Long read timeout: non-streaming completions send nothing until
                # generation finishes. Connect should still fail fast.
                timeout=httpx.Timeout(20 * 60, connect=10),
            ) as response:
                yield response  # first yield: caller inspects status/headers

                start = time.perf_counter()
                chunks = 0
                try:
                    async for chunk in response.aiter_raw():
                        if chunk:
                            yield chunk
                            chunks += 1
                    logger.info("%s completed request", log_prefix)
                except asyncio.CancelledError:
                    logger.info(
                        "%s client disconnected after %d chunks, %.3fs",
                        log_prefix,
                        chunks,
                        time.perf_counter() - start,
                    )
                    raise
                except Exception as exc:
                    logger.error(
                        "%s error streaming response: %r after %d chunks, %.3fs",
                        log_prefix,
                        exc,
                        chunks,
                        time.perf_counter() - start,
                    )
                    raise  # the client must see an error, not a silent stream end

    def build_app(self) -> Any:
        self.client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(
                retries=3,
                limits=httpx.Limits(
                    max_keepalive_connections=200,
                    max_connections=200,
                    keepalive_expiry=60,
                ),
            ),
            follow_redirects=True,
        )

        fastapi_app = FastAPI()
        fastapi_app.add_middleware(_RequestCancelledMiddleware)

        @fastapi_app.on_event("startup")
        async def startup() -> None:
            self.poller_task = asyncio.create_task(self.poll_containers())
            logger.info("router startup complete")

        @fastapi_app.on_event("shutdown")
        async def shutdown() -> None:
            self.poller_task.cancel()
            await asyncio.gather(self.poller_task, return_exceptions=True)
            await self.client.aclose()

        @fastapi_app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        )
        async def forward(request: Request, path: str) -> Any:
            # Metrics should hit the internal server.
            if path.startswith("metrics"):
                return Response(content=b"", status_code=200, media_type="text/plain")

            body = await request.body()
            session_id = request.headers.get(SESSION_AFFINITY_HEADER)
            log_prefix = f"[request {uuid.uuid4()}, session {session_id}]"

            try:
                headers = filter_headers(dict(request.headers))

                for attempt in range(MAX_SESSION_RETRIES + 1):
                    container = None
                    if session_id and self.containers:
                        container = await route_session(
                            self.session_routes,
                            session_id,
                            self.containers,
                            self.overload_threshold,
                        )
                        headers["modal-flash-upstream"] = container.upstream

                    logger.info(
                        "%s proxying to replica %s (attempt=%d)",
                        log_prefix,
                        container.task_id if container else None,
                        attempt,
                    )

                    stream = self.upstream_stream(
                        request, path, headers, body, container, log_prefix
                    )
                    response: httpx.Response = await anext(stream)

                    # Resolve stale upstream state to avoid repeated 503s: a pinned
                    # replica that refuses the request leaves rotation, and the
                    # session re-routes instead of retrying its saturator.
                    if response.status_code == 503 and container is not None:
                        logger.warning(
                            "%s replica %s returned 503 on attempt %d/%d; evicting",
                            log_prefix,
                            container.task_id,
                            attempt + 1,
                            MAX_SESSION_RETRIES + 1,
                        )
                        self.containers.pop(container.task_id, None)
                        if attempt < MAX_SESSION_RETRIES:
                            await stream.aclose()
                            continue

                    return StreamingResponse(
                        stream,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.headers.get("content-type"),
                    )

            except httpx.TimeoutException as exc:
                logger.error("%s upstream timeout: %r", log_prefix, exc)
                return Response(
                    content=b"Upstream Timeout",
                    status_code=504,
                    media_type="text/plain",
                )
            except Exception as exc:
                logger.error("%s error proxying request: %r", log_prefix, exc)
                return Response(
                    content=b"Internal Server Error",
                    status_code=500,
                    media_type="text/plain",
                )

        return fastapi_app
