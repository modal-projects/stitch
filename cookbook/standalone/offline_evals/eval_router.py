"""Version-aware routing for offline-eval rollout pools.

Extends the stock session-affinity router (cookbook/common/router.py) through its
two placement hooks; forwarding, streaming, 503 eviction, and cancellation stay on
the stock code path. The registry polls ``/server_info`` (membership: readiness +
applied version) alongside ``/v1/loads`` (queue depth only). A ``stitch-exact-version``
request header pins placement to replicas applied at that version, with or without a
session id: no in-rotation match (or a cold registry) answers a retryable 409, and a
409 from a replica never evicts it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import modal
from fastapi import Response
from pydantic import BaseModel, TypeAdapter

from cookbook.common.router import (
    SESSION_ROUTE_MAX_UPSTREAMS,
    SESSION_ROUTE_TTL_SECONDS,
    ContainerInfo,
    _container_addr,
    _ProxyApp,
    _RegistryApp,
    select_underloaded_container,
)
from cookbook.common.router import filter_headers as _stock_filter_headers
from cookbook.standalone.offline_evals.rollout_control import (
    DEFAULT_SESSION_TTL_SECONDS,
    RolloutPolicy,
    tombstone_session,
)
from stitch.pools.modal_flash import list_flash_containers_async
from stitch.types import VersionRef

logger = logging.getLogger(__name__)

EXACT_VERSION_HEADER = "stitch-exact-version"


class EvalContainerInfo(ContainerInfo):
    """A registry member with rollout state."""

    load_stale: bool = False  # load is last-known, not live
    ready: bool = True  # False while booting or engine-dead; absent on older sidecars
    # From /server_info's ``applied`` identity; None when nothing is applied yet.
    applied_version: int | None = None
    draining: bool = False  # excluded from every proxy selection path
    # Session records pinned here (registry-computed; drives packing) and
    # in-flight requests (/server_info; gates the reconcile nudge).
    live_sessions: int = 0
    active_requests: int = 0


EvalContainerInfoList = TypeAdapter(list[EvalContainerInfo])


class EvalRouteEntry(BaseModel):
    """One session route entry: the replica's applied version at pin time, the
    last placement touch, and the tombstone. Every stored entry carries the
    full schema."""

    task_id: str
    last_sent: float
    version: int | None
    last_seen: float
    tombstoned: bool = False


EvalRouteEntryList = TypeAdapter(list[EvalRouteEntry])


def filter_headers(headers: dict[str, str]) -> dict[str, str]:
    """Stock hop-by-hop filtering plus the whole ``Modal-*`` namespace: the platform
    hashes Modal-Session-ID for its own session affinity, which takes precedence over
    an explicit ``modal-flash-upstream`` pin mid-path. The pin is re-added after."""
    filtered = _stock_filter_headers(headers)
    return {k: v for k, v in filtered.items() if not k.lower().startswith("modal-")}


def _parse_exact_version(
    headers: Mapping[str, str], session_id: str | None, *, warn: bool = True
) -> int | None:
    """The version pin, or None. Placement is advisory: a malformed pin is ignored."""
    raw = headers.get(EXACT_VERSION_HEADER, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        if warn:
            logger.warning(
                "[session %s] ignoring malformed %s header: %r",
                session_id,
                EXACT_VERSION_HEADER,
                raw,
            )
        return None


def _in_rotation(container: EvalContainerInfo) -> bool:
    return container.ready and not container.draining


def _version_matched(
    containers: dict[str, EvalContainerInfo],
    overload_threshold: int,
    exact_version: int,
) -> list[EvalContainerInfo]:
    """In-rotation replicas applied at ``exact_version``, preferring those under the
    overload threshold: the threshold ranks replicas, it never denies the only correct
    version — a pinned request queues on the lowest-load match rather than failing."""
    in_rotation = [
        c
        for c in containers.values()
        if _in_rotation(c) and c.applied_version == exact_version
    ]
    underloaded = [c for c in in_rotation if c.load < overload_threshold]
    return underloaded or in_rotation


def select_packed_container(candidates: list[EvalContainerInfo]) -> EvalContainerInfo:
    """Pack pinned-version traffic onto replicas already holding live sessions:
    lowest load among the holders, or among all candidates when none hold any.
    Packing concentrates same-version sessions so empty old-version replicas
    drain to zero live sessions and become reconcile candidates."""
    holders = [c for c in candidates if c.live_sessions > 0]
    # A stale load is last-known, not live (a staged replica can sit at a
    # frozen zero): when any candidate reports fresh load, pack across those.
    fresh = [c for c in (holders or candidates) if not c.load_stale]
    pool = fresh or holders or list(candidates)
    return min(pool, key=lambda c: c.load)


async def route_session(
    session_routes: modal.Dict,
    session_id: str,
    containers: dict[str, EvalContainerInfo],
    overload_threshold: int,
    exact_version: int | None = None,
) -> EvalContainerInfo | None:
    """Pick the replica for one session: its freshest in-rotation sticky route with
    headroom (and, when pinned, the matching applied version), else a fresh pick —
    lowest-load version match pinned, random with headroom unpinned. Returns None
    without persisting a route when nothing in rotation matches. Mutates and persists
    the session's routes."""
    current_time = time.time()

    routes: list[EvalRouteEntry] = EvalRouteEntryList.validate_python(
        await session_routes.get.aio(session_id, [])
    )
    # Drop expired routes, tombstoned entries (an ended session re-pins clean),
    # and replicas this pod hasn't discovered (they may just be new; every
    # session lands on some pod's view, so this only trims the dead).
    routes = [
        entry
        for entry in routes
        if current_time - entry.last_sent <= SESSION_ROUTE_TTL_SECONDS
        and entry.task_id in containers
        and not entry.tombstoned
    ]
    routes.sort(key=lambda entry: entry.last_sent, reverse=True)

    async def save_routes() -> None:
        await session_routes.put.aio(
            session_id,
            EvalRouteEntryList.dump_python(
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
        if not _in_rotation(container):
            # Delete, don't skip: a kept sticky route would re-admit the session
            # onto a replica about to swap weights or one serving boot weights.
            logger.info(
                "session %s: evicting sticky route %s (%s)",
                session_id,
                container.task_id,
                "draining" if container.draining else "not ready",
            )
            routes.remove(entry)
            await save_routes()
            continue
        if container.load < overload_threshold:
            if exact_version is not None and container.applied_version != exact_version:
                logger.info(
                    "session %s: sticky route %s applied_version=%s != pin %d; skipping",
                    session_id,
                    container.task_id,
                    container.applied_version,
                    exact_version,
                )
                continue
            entry.last_sent = current_time
            entry.last_seen = current_time
            await save_routes()
            return container

    if exact_version is not None:
        matched = _version_matched(containers, overload_threshold, exact_version)
        if not matched:
            logger.info(
                "session %s: no in-rotation upstream at exact_version=%d (%d known); "
                "not re-pinning",
                session_id,
                exact_version,
                len(containers),
            )
            return None
        selected = select_packed_container(matched)
    else:
        in_rotation = {
            container.task_id: container
            for container in containers.values()
            if _in_rotation(container)
        }
        if not in_rotation:
            logger.info("session %s: no replica in rotation; no selection", session_id)
            return None
        selected = select_underloaded_container(in_rotation, overload_threshold)

    logger.info(
        "session %s: pinning upstream %s (load %d, applied_version=%s, pin=%s, "
        "prior=[%s])",
        session_id,
        selected.task_id,
        selected.load,
        selected.applied_version,
        exact_version,
        ", ".join(entry.task_id for entry in routes),
    )
    routes.insert(
        0,
        EvalRouteEntry(
            task_id=selected.task_id,
            last_sent=current_time,
            version=selected.applied_version,
            last_seen=current_time,
        ),
    )
    await save_routes()
    return selected


def serve_eval_registry(
    replica: Any,
    *,
    app_name: str,
    upstream_cls: str,
    session_routes: Any,
    control: Any,
    store: Any,
    session_ttl: float = DEFAULT_SESSION_TTL_SECONDS,
) -> None:
    """Start the version-aware load registry on a ``RouterRegistry`` container
    (@modal.enter). Polls go through the upstream class's own URL, derived here
    so recipes only name the class. The registry also drives the rollout control
    loop (rollout_control): it reads the store's ``latest`` pointer each poll and
    reconciles stale replicas toward it. ``session_ttl`` bounds how long an idle
    session record counts as live; explicit tombstones are the primary signal."""
    registry = EvalRegistryApp(
        app_name=app_name,
        upstream_cls=upstream_cls,
        upstream_url=modal.Server.from_name(app_name, upstream_cls).get_url(),
        session_routes=session_routes,
        control=control,
        store=store,
        session_ttl=session_ttl,
    )
    registry.start()
    replica._router_server = registry


def serve_eval_router(
    replica: Any,
    *,
    registry_url: str,
    upstream_url: str,
    session_routes: modal.Dict,
    overload_threshold: int,
) -> None:
    """Start the version-aware session-routing proxy on a ``Router`` container
    (@modal.enter)."""
    router = EvalProxyApp(
        registry_url=registry_url,
        upstream_url=upstream_url,
        session_routes=session_routes,
        overload_threshold=overload_threshold,
    )
    router.start()
    replica._router_server = router


@dataclass
class _ReplicaPoll:
    """One replica's poll result. ``load=None`` means /v1/loads failed — the replica
    stays a member (/server_info is the membership contract) on its last-known load."""

    load: int | None
    applied_version: int | None
    ready: bool
    active_requests: int


class EvalRegistryApp(_RegistryApp):
    """Polls replicas' /server_info (membership, applied version) and /v1/loads
    (queue depth) through the pool URL; serves the snapshot at /loads. Between
    polls it drives the rollout control loop: pointer watch, session liveness,
    pending-reconcile marks, reconciliation nudges."""

    def __init__(
        self,
        *,
        app_name: str,
        upstream_cls: str,
        upstream_url: str,
        session_routes: Any,
        control: Any,
        store: Any,
        session_ttl: float = DEFAULT_SESSION_TTL_SECONDS,
    ) -> None:
        super().__init__(
            app_name=app_name, upstream_cls=upstream_cls, upstream_url=upstream_url
        )
        self.containers: list[EvalContainerInfo] = []
        self._last_loads: dict[str, int] = {}
        self.store = store
        self._pointer_version: int | None = None
        self._rollout_policy = RolloutPolicy(
            session_routes=session_routes,
            control=control,
            session_ttl=session_ttl,
            reconcile=self._post_wake,
        )

    def _read_pointer_version(self) -> int | None:
        """The store's ``latest`` pointer, cached — only a change logs (and arms
        the retire pass). Reuses the standard store read path, so any publish —
        library-path included — is observed. A read failure keeps last-known."""
        if self.store is None:
            return None
        try:
            self.store.refresh()
            pointer = self.store.read_pointer()
        except Exception:
            logger.exception("rollout: pointer read failed; keeping last-known")
            return self._pointer_version
        version = pointer.version if pointer is not None else None
        if version != self._pointer_version:
            logger.info(
                "rollout: store pointer %s -> %s", self._pointer_version, version
            )
            self._pointer_version = version
        return self._pointer_version

    async def _post_wake(self, upstream: str) -> None:
        """POST /wake to one replica through the pool URL (the same path
        polls traverse): the stock sidecar wake, which reconciles the replica
        to the store pointer. A failed nudge is logged, never fatal — the
        control loop re-nudges on its bounded schedule."""
        try:
            response = await self.client.post(
                f"{self.upstream_url}/wake",
                headers={"modal-flash-upstream": upstream},
            )
            response.raise_for_status()
        except Exception:
            logger.exception("rollout: wake nudge failed for %s", upstream)

    async def _poll_container(self, upstream: str) -> _ReplicaPoll:
        """/server_info is membership and liveness in one probe: its failure drops
        the replica. /v1/loads is the load number only: its failure never drops the
        replica — a sidecar mid weight-stage can block /v1/loads for minutes while
        still serving."""
        headers = {"modal-flash-upstream": upstream}
        server_info_response, loads_response = await asyncio.gather(
            self.client.get(f"{self.upstream_url}/server_info", headers=headers),
            self.client.get(
                f"{self.upstream_url}/v1/loads",
                params={"include": "core"},
                headers=headers,
            ),
            return_exceptions=True,
        )
        if isinstance(server_info_response, BaseException):
            raise server_info_response
        server_info_response.raise_for_status()
        data = server_info_response.json()

        applied_version: int | None = None
        if applied := data.get("applied"):
            try:
                applied_version = VersionRef.parse(str(applied)).version
            except ValueError:
                logger.warning(
                    "replica %s: unparseable applied identity %r", upstream, applied
                )

        load: int | None = None
        if not isinstance(loads_response, BaseException):
            try:
                loads_response.raise_for_status()
                loads = loads_response.json()["loads"]
                load = sum(
                    int(entry["num_waiting_reqs"]) + int(entry["num_running_reqs"])
                    for entry in loads
                )
            except Exception as exc:
                logger.debug("loads poll failed for %s: %r", upstream, exc)

        # Older sidecars omit `ready`; absence reads as ready.
        return _ReplicaPoll(
            load=load,
            applied_version=applied_version,
            ready=bool(data.get("ready", True)),
            active_requests=int(data.get("active_requests") or 0),
        )

    async def _poll_once(self) -> list[EvalContainerInfo]:
        discovered = await list_flash_containers_async(self.app_name, self.upstream_cls)
        upstreams = {}
        for container in discovered:
            task_id = (
                container.get("task_id")
                if isinstance(container, dict)
                else getattr(container, "task_id", None)
            )
            addr = _container_addr(container)
            if task_id and addr:
                upstreams[task_id] = addr
        results = await asyncio.gather(
            *(self._poll_container(upstream) for upstream in upstreams.values()),
            return_exceptions=True,
        )
        updated: list[EvalContainerInfo] = []
        for (task_id, upstream), result in zip(upstreams.items(), results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "server_info poll failed for %s: %r; dropped", task_id, result
                )
                self._last_loads.pop(task_id, None)
                continue
            if result.load is None:
                load, load_stale = self._last_loads.get(task_id, 0), True
            else:
                load, load_stale = result.load, False
                self._last_loads[task_id] = load
            updated.append(
                EvalContainerInfo(
                    task_id=task_id,
                    upstream=upstream,
                    load=load,
                    load_stale=load_stale,
                    applied_version=result.applied_version,
                    ready=result.ready,
                    active_requests=result.active_requests,
                )
            )
        pointer_version = self._read_pointer_version()
        try:
            await self._rollout_policy.update(updated, pointer_version, time.time())
        except Exception:
            # The placement snapshot must still refresh — a control-loop bug
            # costs reconciliation, never routing data.
            logger.exception("rollout: control update failed")
        return updated


class EvalProxyApp(_ProxyApp):
    """Version-aware placement over the stock proxy via the two routing hooks."""

    def build_app(self) -> Any:
        fastapi_app = super().build_app()

        @fastapi_app.post("/sessions/{session_id}/end")
        async def end_session(session_id: str) -> Response:
            # Session tombstone — the primary end-of-session signal. Idempotent
            # and 200 always: ending an unknown or already-ended session is a
            # no-op.
            await tombstone_session(self.session_routes, session_id)
            return Response(status_code=200)

        # The decorator appends; the tombstone must precede the stock catch-all
        # proxy route or it never matches.
        routes = fastapi_app.router.routes
        catch_all = next(
            i
            for i, route in enumerate(routes)
            if getattr(route, "path", None) == "/{path:path}"
        )
        routes.insert(catch_all, routes.pop())
        return fastapi_app

    async def get_containers(self) -> dict[str, EvalContainerInfo]:
        response = await self.client.get(f"{self.registry_url}/loads")
        response.raise_for_status()
        containers = EvalContainerInfoList.validate_json(response.content)
        return {container.task_id: container for container in containers}

    async def select_container(
        self, session_id: str | None, headers: dict[str, str]
    ) -> EvalContainerInfo | None:
        # Re-filter the forwarding headers in place: the eval policy also strips
        # the Modal-* namespace (see filter_headers).
        filtered = filter_headers(headers)
        headers.clear()
        headers.update(filtered)

        exact_version = _parse_exact_version(headers, session_id)
        if exact_version is not None and not self.containers:
            return None  # cold registry: unrouted_response answers 409
        if exact_version is not None and not session_id:
            # Pinned without a session: version-aware selection, no sticky state.
            matched = _version_matched(
                self.containers, self.overload_threshold, exact_version
            )
            return select_packed_container(matched) if matched else None
        if session_id and self.containers:
            return await route_session(
                self.session_routes,
                session_id,
                self.containers,
                self.overload_threshold,
                exact_version=exact_version,
            )
        return None

    async def unrouted_response(
        self, session_id: str | None, headers: dict[str, str]
    ) -> Response | None:
        # select_container already warned on a malformed pin.
        exact_version = _parse_exact_version(headers, session_id, warn=False)
        if exact_version is not None:
            # Pinned with no in-rotation match or a cold registry: a retryable
            # 409 — pinned traffic never lands on a wrong-version or draining
            # replica and never falls through to the Flash LB.
            logger.info(
                "[session %s] no in-rotation upstream (exact_version=%d); 409",
                session_id,
                exact_version,
            )
            return Response(
                content=b"No healthy upstream",
                status_code=409,
                media_type="text/plain",
            )
        if session_id and self.containers:
            # Zero replicas in rotation (all booting or draining): a retryable 503.
            logger.info("[session %s] no replica in rotation; 503", session_id)
            return Response(
                content=b"No replica in rotation",
                status_code=503,
                headers={"Retry-After": "1"},
                media_type="text/plain",
            )
        return None  # stock: fall through to the Flash LB
