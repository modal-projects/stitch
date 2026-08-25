"""Session-affinity load-balancing router for the rollout pool.

A run-scoped fork of the flash-smart-router (github.com/modal-labs/flash-smart-router),
folded into the run's own Modal app: each recipe deploys a ``RouterRegistry`` and a
``Router`` Flash server class next to its GPU ``Server`` (thin module-level classes whose
lifecycle delegates here, exactly like ``common.server`` does for sglang). One app, one
deploy, one stop — the router scales and dies with the pool.

Why it exists: with a per-container queue bound (sglang ``--max-queued-requests``), Flash's
own sticky routing turns the first saturated replicas into 503 attractors — sessions stuck
on a full replica keep retrying it while the rest of the pool starves. Here, the registry
polls every replica's ``/server_info`` (authoritative membership: applied_version, leases,
sync_state, target_version) and ``/v1/loads`` (live queue depth only); the router pins each
session to a replica with spare capacity via the ``modal-flash-upstream`` header, and a 503
from a pinned replica evicts it from rotation and retries on a healthier one, so load
spreads instead of sticking. When a newer version starves, the registry marks one
old-version replica ``draining`` at a time; draining replicas are excluded from
every proxy selection path until their applied_version flips.

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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, AsyncGenerator, Generator

import httpx
import modal
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, TypeAdapter

from stitch.pools.modal_flash import list_flash_containers_async

from .constants import (
    MODAL_SESSION_ID_HEADER,
    STITCH_LEASE_HEADER,
    STITCH_SESSION_HEADER,
)

logger = logging.getLogger(__name__)

SESSION_ROUTE_TTL_SECONDS = 4 * 60 * 60
SESSION_ROUTE_MAX_UPSTREAMS = 10

CONTAINER_POLL_INTERVAL_SECONDS = 1.0
CONTAINER_POLL_TIMEOUT_SECONDS = 3.0

MAX_SESSION_RETRIES = 3

# Consolidation: a victim is marked draining only after the drain condition
# holds for this many consecutive registry polls (hysteresis against blips).
DRAIN_HYSTERESIS_POLLS = 5
# ...and a persisted victim is cleared only after this many consecutive polls
# where the keep-condition fails: a single /server_info blip (the victim's
# poll failed and dropped it from one snapshot) must not flap the drain.
DRAIN_CLEAR_HYSTERESIS_POLLS = 3
# Drain only when the old version's leases fit on the remaining k-1 replicas
# with this safety margin below the engine overload threshold.
DRAIN_SAFETY_FACTOR = 0.7
# sync_states that mean "new-version capacity is imminent" — draining then is
# premature. HOLDING is deliberately absent: holders are the starved
# lease-holders the drain exists to free.
_DRAIN_BLOCKING_STATES = frozenset({"FETCHING", "STAGING", "COMMITTING"})

_COOKBOOK_DIR = Path(__file__).resolve().parent.parent


def build_router_image(
    experiment: str,
    run_id: str,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> modal.Image:
    """CPU image for the router classes. Like the serving image, it bakes the run
    coordinates and carries the cookbook + stitch sources, so a container's re-import of
    the recipe app module rebuilds the same app name and class wiring."""
    return (
        modal.Image.debian_slim()
        .pip_install("fastapi", "httpx", "pydantic", "uvicorn")
        .env(
            {
                **(extra_env or {}),
                "EXPERIMENT_CONFIG": experiment,
                "RUN_ID": run_id,
            }
        )
        .add_local_python_source("stitch")
        .add_local_dir(
            str(_COOKBOOK_DIR), remote_path="/root/cookbook", ignore=["**/__pycache__"]
        )
    )


def session_routes_dict(app_name: str) -> modal.Dict:
    """The run's session→replica affinity store, namespaced to the run app."""
    return modal.Dict.from_name(f"{app_name}-session-routes", create_if_missing=True)


def consolidation_dict(app_name: str) -> modal.Dict:
    """The run's consolidation record (single fleet-wide drain victim), namespaced
    to the run app. Persisted outside /loads so a registry restart or a victim's
    poll blip cannot silently clear or duplicate the victim."""
    return modal.Dict.from_name(f"{app_name}-consolidation", create_if_missing=True)


class _DictMethod:
    """Duck-types a Modal SDK synchronicity-wrapped Dict method (callable + .aio)."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def __call__(self, *args: Any) -> Any:
        return self.fn(*args)

    async def aio(self, *args: Any) -> Any:
        return self.fn(*args)


class _LocalDict:
    """In-memory stand-in for the consolidation modal.Dict (tests, single-process)."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self.get = _DictMethod(lambda key, default=None: self._store.get(key, default))
        self.put = _DictMethod(lambda key, value: self._store.__setitem__(key, value))


# ── routing state ────────────────────────────────────────────────────────────


class ContainerInfo(BaseModel):
    task_id: str
    upstream: str  # host:port of the replica
    load: int  # queued + running requests, last successful /v1/loads poll
    load_stale: bool = False  # True when load is last-known, not live
    # False while booting or after a terminal engine error; excluded from every
    # selection path. Older sidecars omit it (treated as ready).
    ready: bool = True
    applied_version: int | None = None
    leases: dict[str, int] = Field(default_factory=dict)  # /server_info; JSON keys are strings
    sync_state: str | None = None
    target_version: int | None = None  # None when idle; older sidecars use metrics.target_version
    draining: bool = False  # excluded from every proxy selection path


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
        MODAL_SESSION_ID_HEADER.lower(),
        STITCH_SESSION_HEADER.lower(),
        STITCH_LEASE_HEADER.lower(),
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-prefix",
        "x-forwarded-proto",
        "x-forwarded-server",
    }
    return {k: v for k, v in headers.items() if k.lower() not in removed}


def session_id_from_headers(headers: Any) -> str | None:
    """Session identity for stickiness: ``Stitch-Session-Id``, else ``Modal-Session-ID``."""
    return headers.get(STITCH_SESSION_HEADER) or headers.get(MODAL_SESSION_ID_HEADER)


def relay_lease_header(headers: dict[str, str], session_id: str | None) -> dict[str, str]:
    """Copy the stickiness session id into ``Stitch-Lease-Key``. Mutates ``headers``."""
    if session_id:
        headers[STITCH_LEASE_HEADER] = session_id
    return headers


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
    pool = candidates or list(containers.values())
    if not pool:
        raise ValueError("select_underloaded_container: no containers to select from")
    return random.choice(pool)


def select_packed_container(
    candidates: list[ContainerInfo], exact_version: int
) -> ContainerInfo:
    """Among replicas applied at ``exact_version``, pick the lowest-load
    lease-holder (all candidates if none hold leases). Packing onto the
    most-leased replica stampedes it past the engine overload threshold."""

    def lease_count(container: ContainerInfo) -> int:
        return container.leases.get(str(exact_version), 0)

    lease_holders = [c for c in candidates if lease_count(c) > 0]
    pool = lease_holders or list(candidates)
    if not pool:
        raise ValueError("select_packed_container: no candidates to select from")
    # A stale load is last-known, not live (a paused replica can sit at a
    # frozen zero): when any candidate reports fresh load, pack only across
    # those.
    fresh = [c for c in pool if not c.load_stale]
    pool = fresh or pool
    min_load = min(c.load for c in pool)
    best = [c for c in pool if c.load == min_load]
    return random.choice(best)


def select_drain_victim(
    containers: list[ContainerInfo],
    rollout_concurrency: int,
    *,
    safety_factor: float = DRAIN_SAFETY_FACTOR,
) -> tuple[int, str] | None:
    """The consolidation candidate: ``(version, task_id)`` to drain, or None.

    Oldest applied version V with k>=2 lease-holders and some replica targeting
    newer than V (a k=1 group is skipped, not fatal). Drain when leases at V
    fit on k-1 survivors under ``rollout_concurrency * DRAIN_SAFETY_FACTOR``
    and every survivor's load is below ``rollout_concurrency``. None while any
    replica is FETCHING/STAGING/COMMITTING. Victim is min(task_id).
    """
    targets = [c.target_version for c in containers if c.target_version is not None]
    if not targets:
        return None
    max_target = max(targets)
    if any(
        (c.sync_state or "").upper() in _DRAIN_BLOCKING_STATES for c in containers
    ):
        return None

    by_version: dict[int, list[ContainerInfo]] = {}
    for container in containers:
        if container.applied_version is None or container.draining:
            continue
        if container.leases.get(str(container.applied_version), 0) > 0:
            by_version.setdefault(container.applied_version, []).append(container)

    for version in sorted(by_version):
        if version >= max_target:
            continue
        group = by_version[version]
        k = len(group)
        if k < 2:
            continue
        victim = min(group, key=lambda c: c.task_id)
        survivors = [c for c in group if c.task_id != victim.task_id]
        total_leases = sum(c.leases.get(str(version), 0) for c in group)
        if total_leases > (k - 1) * rollout_concurrency * safety_factor:
            continue
        if any(c.load_stale for c in survivors):
            # A stale survivor load is unverifiable spare capacity — never
            # approve a drain on a frozen (possibly zero) reading.
            continue
        if any(c.load >= rollout_concurrency for c in survivors):
            continue
        return version, victim.task_id
    return None


async def route_session(
    session_routes: modal.Dict,
    session_id: str,
    containers: dict[str, ContainerInfo],
    overload_threshold: int,
    exact_version: int | None = None,
) -> ContainerInfo | None:
    """Pick the replica for one session: sticky if healthy and (when set)
    applied_version == exact_version, else a packed same-version replica.
    Draining replicas are excluded; sticky routes to them are deleted.
    Returns None — and does not persist a route — when exact_version is set
    but no non-draining replica matches it."""
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

    for entry in list(routes):
        # Re-lookup defensively: a concurrent 503-evict can pop the shared map
        # while this coroutine is suspended in save_routes() — indexing would
        # turn that race into a KeyError 500.
        container = containers.get(entry.task_id)
        if container is None:
            continue
        if container.draining:
            # Delete, don't skip: a kept sticky route would re-admit the victim on a blip.
            logger.info(
                "session %s: sticky route %s is draining; evicting",
                session_id,
                container.task_id,
            )
            routes.remove(entry)
            await save_routes()
            continue
        if not container.ready:
            # Booting (stale boot weights) or dead-engine replica: evict like
            # draining so the session migrates instead of pinning stale weights.
            logger.info(
                "session %s: sticky route %s is not ready; evicting",
                session_id,
                container.task_id,
            )
            routes.remove(entry)
            await save_routes()
            continue
        if container.load < overload_threshold:
            if exact_version is not None and container.applied_version != exact_version:
                logger.info(
                    "session %s: sticky route %s has applied_version %s, "
                    "not matching exact_version %d; evicting",
                    session_id,
                    container.task_id,
                    container.applied_version,
                    exact_version,
                )
                continue
            entry.last_sent = current_time
            await save_routes()
            return container

    candidates = [
        container
        for container in containers.values()
        if container.load < overload_threshold
        and not container.draining
        and container.ready
    ]
    # Unpinned fallback is overloaded-but-not-draining; an all-draining fleet yields None.
    fallback = candidates or [
        container
        for container in containers.values()
        if not container.draining and container.ready
    ]

    if exact_version is not None:
        version_matched = [c for c in candidates if c.applied_version == exact_version]
        if not version_matched:
            # Overloaded-but-matching fallback: the threshold ranks versions,
            # it never denies the only correct version. Saturated at-version
            # replicas still take the pinned request (queue there) — a 409
            # here stops lease renewals, the holds collapse, sessions die.
            version_matched = [
                container
                for container in containers.values()
                if container.applied_version == exact_version
                and not container.draining
                and container.ready
            ]
        if version_matched:
            selected = select_packed_container(version_matched, exact_version)
        else:
            # No healthy upstream at the pinned version: return None rather than
            # landing on a wrong-version or draining replica (which would 409
            # and re-pin the session onto the victim).
            logger.info(
                "session %s: no healthy upstream at exact_version=%d "
                "(%d replicas known); not re-pinning",
                session_id,
                exact_version,
                len(containers),
            )
            return None
    else:
        if not fallback:
            logger.info(
                "session %s: no non-draining replica available; no selection",
                session_id,
            )
            return None
        selected = random.choice(fallback)

    reason = (
        f"previous upstreams [{', '.join(entry.task_id for entry in routes)}] overloaded"
        f" (load >= {overload_threshold})"
        if routes
        else "no previous upstream"
    )
    log_suffix = f" (exact_version={exact_version})" if exact_version is not None else ""
    logger.info(
        "session %s: %s; pinning new upstream %s (load %d, applied_version=%s)%s",
        session_id,
        reason,
        selected.task_id,
        selected.load,
        selected.applied_version,
        log_suffix,
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


def serve_registry(
    replica: Any,
    *,
    app_name: str,
    upstream_cls: str,
    rollout_concurrency: int,
    consolidation: Any = None,
) -> None:
    """Start the load-registry server on a ``RouterRegistry`` container (@modal.enter)."""
    # Router containers default to WARNING; routing decisions ("pinning new
    # upstream", "marking … draining") are INFO and must reach the log store.
    logging.basicConfig(level=logging.INFO)
    registry = _RegistryApp(
        app_name=app_name,
        upstream_cls=upstream_cls,
        rollout_concurrency=rollout_concurrency,
        consolidation=consolidation,
    )
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
    logging.basicConfig(level=logging.INFO)
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


class _ReplicaPoll:
    """One replica's poll result. ``load=None`` means /v1/loads failed — the
    replica stays a member (/server_info is the membership contract) with its
    last-known load."""

    def __init__(
        self,
        *,
        load: int | None,
        applied_version: int | None,
        leases: dict[str, int],
        sync_state: str | None,
        target_version: int | None,
        ready: bool,
    ) -> None:
        self.load = load
        self.applied_version = applied_version
        self.leases = leases
        self.sync_state = sync_state
        self.target_version = target_version
        self.ready = ready


class _RegistryApp(_UvicornApp):
    """Poll /server_info (membership) and /v1/loads (queue depth); serve ``/loads``."""

    def __init__(
        self,
        *,
        app_name: str,
        upstream_cls: str,
        rollout_concurrency: int = 1,
        consolidation: Any = None,
    ) -> None:
        self.app_name = app_name
        self.upstream_cls = upstream_cls
        # Engine overload threshold (the recipe's rollout_target_inputs) — the
        # capacity unit of the drain condition, NOT the router's own Flash
        # target_concurrency.
        self.rollout_concurrency = rollout_concurrency
        self.consolidation = consolidation if consolidation is not None else _LocalDict()
        self.containers: list[ContainerInfo] = []
        self._last_loads: dict[str, int] = {}
        self._drain_pending: tuple[int, str] | None = None
        self._drain_pending_polls = 0
        self._drain_clear_polls = 0

    async def _poll_container(self, upstream: str) -> _ReplicaPoll:
        """/server_info is membership (failure drops the replica). /v1/loads is
        the load number only: its failure never drops the replica — a
        STAGING/COMMITTING sidecar blocks /v1/loads for minutes."""
        server_info_response, loads_response = await asyncio.gather(
            self.client.get(f"https://{upstream}/server_info"),
            self.client.get(f"https://{upstream}/v1/loads", params={"include": "core"}),
            return_exceptions=True,
        )
        if isinstance(server_info_response, BaseException):
            raise server_info_response
        server_info_response.raise_for_status()
        data = server_info_response.json()
        raw_leases = data.get("leases") or {}
        # JSON object keys arrive as strings.
        leases = {str(version): int(count) for version, count in raw_leases.items()}
        sync_state = data.get("sync_state")
        target_version = data.get("target_version")
        if (sync_state or "").upper() == "ERROR":
            # An errored replica has no target; never trust a leftover value.
            target_version = None
        elif target_version is None and (sync_state or "").upper() != "IDLE":
            # Older sidecars only expose it under metrics; tolerate absence.
            # An IDLE replica's metrics value is a stale phantom (the metrics
            # dict outlives the target), so the fallback skips IDLE too.
            target_version = (data.get("metrics") or {}).get("target_version")

        load: int | None = None
        if not isinstance(loads_response, BaseException):
            try:
                loads_response.raise_for_status()
                loads = loads_response.json()["loads"]
                queued = sum(int(load["num_waiting_reqs"]) for load in loads)
                running = sum(int(load["num_running_reqs"]) for load in loads)
                load = queued + running
            except Exception as exc:
                logger.debug("loads poll failed for %s: %r", upstream, exc)

        return _ReplicaPoll(
            load=load,
            applied_version=data.get("applied_version"),
            leases=leases,
            sync_state=sync_state,
            target_version=target_version,
            # Older sidecars don't expose `ready`; treat absence as ready.
            ready=bool(data.get("ready", True)),
        )

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
            *(self._poll_container(upstream) for upstream in upstreams.values()),
            return_exceptions=True,
        )
        updated: list[ContainerInfo] = []
        for (task_id, upstream), result in zip(upstreams.items(), results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "server_info poll failed for replica %s: %r; dropping from registry",
                    task_id,
                    result,
                )
                self._last_loads.pop(task_id, None)
                continue
            if result.load is None:
                load, load_stale = self._last_loads.get(task_id, 0), True
            else:
                load, load_stale = result.load, False
                self._last_loads[task_id] = load
            updated.append(
                ContainerInfo(
                    task_id=task_id,
                    upstream=upstream,
                    load=load,
                    load_stale=load_stale,
                    applied_version=result.applied_version,
                    leases=result.leases,
                    sync_state=result.sync_state,
                    target_version=result.target_version,
                    ready=result.ready,
                )
            )
        await self._update_drain(updated)
        return updated

    async def _update_drain(self, containers: list[ContainerInfo]) -> None:
        """Mark/unmark the single fleet-wide drain victim.

        The record lives in the shared consolidation dict (survives registry
        restart). Unmarked when applied_version changes, no replica targets
        newer, or the victim left — each only after DRAIN_CLEAR_HYSTERESIS_POLLS
        consecutive failing polls. A new victim requires
        DRAIN_HYSTERESIS_POLLS consecutive polls of the same candidate.
        """
        record = await self.consolidation.get.aio("victim", None) or {}
        victim_id, version = record.get("victim_task_id"), record.get("version")
        if victim_id is not None and version is not None:
            victim = next((c for c in containers if c.task_id == victim_id), None)
            newer_target = any(
                c.target_version is not None and c.target_version > version
                for c in containers
            )
            if victim is not None and victim.applied_version == version and newer_target:
                victim.draining = True
                self._drain_clear_polls = 0
                return
            # The keep-condition failed on this poll. A single blip (the
            # victim's /server_info poll failed and dropped it from this
            # snapshot, or a stale read mid-flip) must not flap the drain:
            # clear the record only after DRAIN_CLEAR_HYSTERESIS_POLLS
            # consecutive failing polls.
            self._drain_clear_polls += 1
            if self._drain_clear_polls < DRAIN_CLEAR_HYSTERESIS_POLLS:
                if victim is not None and victim.applied_version == version:
                    victim.draining = True  # status quo across the blip
                return
            await self.consolidation.put.aio(
                "victim", {"victim_task_id": None, "version": None}
            )
            if victim is not None:
                victim.draining = False
            self._drain_pending = None
            self._drain_pending_polls = 0
            self._drain_clear_polls = 0

        candidate = select_drain_victim(containers, self.rollout_concurrency)
        if candidate is None:
            self._drain_pending = None
            self._drain_pending_polls = 0
            return
        if candidate == self._drain_pending:
            self._drain_pending_polls += 1
        else:
            self._drain_pending = candidate
            self._drain_pending_polls = 1
        if self._drain_pending_polls >= DRAIN_HYSTERESIS_POLLS:
            victim_version, victim_id = candidate
            await self.consolidation.put.aio(
                "victim", {"victim_task_id": victim_id, "version": victim_version}
            )
            self._drain_clear_polls = 0
            for container in containers:
                if container.task_id == victim_id:
                    container.draining = True
            logger.info(
                "consolidation: marking replica %s draining at version %d",
                victim_id,
                victim_version,
            )

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
            session_id = session_id_from_headers(request.headers)
            log_prefix = f"[request {uuid.uuid4()}, session {session_id}]"

            exact_version: int | None = None
            exact_version_header = request.headers.get("stitch-exact-version", "").strip()
            if exact_version_header:
                try:
                    exact_version = int(exact_version_header)
                except ValueError:
                    logger.warning(
                        "%s ignoring malformed stitch-exact-version header: %r",
                        log_prefix,
                        exact_version_header,
                    )

            try:
                headers = filter_headers(dict(request.headers))
                relay_lease_header(headers, session_id)

                for attempt in range(MAX_SESSION_RETRIES + 1):
                    container = None
                    if session_id and self.containers:
                        container = await route_session(
                            self.session_routes,
                            session_id,
                            self.containers,
                            self.overload_threshold,
                            exact_version=exact_version,
                        )
                        if container is None:
                            # No healthy upstream at the pin: 409, do not re-pin.
                            logger.info(
                                "%s no healthy upstream (exact_version=%s); "
                                "answering 409",
                                log_prefix,
                                exact_version,
                            )
                            return Response(
                                content=b"No healthy upstream",
                                status_code=409,
                                media_type="text/plain",
                            )
                        headers["modal-flash-upstream"] = container.upstream

                    logger.info(
                        "%s proxying to replica %s (attempt=%d, exact_version=%s)",
                        log_prefix,
                        container.task_id if container else None,
                        attempt,
                        exact_version,
                    )

                    stream = self.upstream_stream(
                        request, path, headers, body, container, log_prefix
                    )
                    response: httpx.Response = await anext(stream)

                    # Only 503 evicts. A 409 is a version mismatch; the client
                    # retries and the next registry poll refreshes applied_version.
                    if response.status_code == 503 and container is not None:
                        logger.warning(
                            "%s replica %s returned %d on attempt %d/%d; evicting",
                            log_prefix,
                            container.task_id,
                            response.status_code,
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
