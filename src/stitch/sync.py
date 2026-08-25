"""The per-replica sync brain: a ``Reconciler`` that converges the replica to the
store's pointer, with rollout requests admitted through a composed ``AdmissionGate``.

Like the watchdogs, the gate is a standalone component wired by injection: the
reconciler supplies a ``served_version`` reader and an ``on_reject`` hook, and drives
weight commits through ``gate.commit``. The reader runs under the gate's condition
lock, so a request's constraint is checked and its serving version captured with the
same coherence the committer holds across the weight apply *and* the version flip —
with no inheritance between the two policies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import Any, Literal

from stitch.engines.base import Engine
from stitch.errors import UnrecoverableEngineError, UnrecoverableSidecarError
from stitch.stores.base import Store
from stitch.types import (
    SyncState,
    VersionConstraint,
    VersionKind,
    VersionManifest,
    VersionRef,
)

logger = logging.getLogger(__name__)

CommitMode = Literal["quiesce", "in_place"]


@contextmanager
def _timed(metrics: dict[str, Any], key: str):
    """Record the block's wall-clock into ``metrics[key]`` (seconds, 3 dp)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        metrics[key] = round(time.perf_counter() - start, 3)


class ConstraintUnmet(Exception):
    """A request's version constraint cannot be met by this replica (a retryable 409)."""

    def __init__(self, error: dict[str, Any]) -> None:
        super().__init__(error["message"])
        self.error = error


class AdmissionGate:
    """Request admission + the commit gate.

    ``quiesce`` drains all in-flight requests before applying; ``in_place``
    pauses the engine and lets non-exact requests already in flight continue on the new
    weights (only exact pins are drained). Admission closes during either commit, so a
    newly arriving request is attributed to the version it will actually run on. An
    incompatible transition (a run switch's boot reset) commits with ``drain_all=True``,
    which also drains all in-flight requests. The version flips before admission reopens.

    Version leases (``version_lease_ttl > 0``) hold a replica on its applied
    version while pinned sessions are active. An ``in_place`` commit waits,
    admission still open, until no unexpired lease names a version below the
    target; the reconciler runs that wait *before* staging, because a
    cpu-destination stage pauses generation for minutes.

    The gate does not own the served version: it reads it through the injected
    ``served_version`` reader, always under its condition lock, and reports every
    rejection to the ``on_reject`` hook (so the owner can treat a 409 as a catch-up
    trigger). A gate constructed without a reader has no served version: every
    constrained request is rejected as retryable until a source is attached.
    """

    def __init__(
        self,
        *,
        commit_mode: CommitMode = "in_place",
        served_version: Callable[[], VersionRef | None] | None = None,
        on_reject: Callable[[dict[str, Any]], None] | None = None,
        on_hold_start: Callable[[], None] | None = None,
        on_hold_end: Callable[[], None] | None = None,
        version_lease_ttl: float = 0.0,
    ) -> None:
        self.commit_mode = commit_mode
        self._served_version = served_version or (lambda: None)
        self._on_reject = on_reject or (lambda error: None)
        self._on_hold_start = on_hold_start
        self._on_hold_end = on_hold_end
        self._lease_ttl = version_lease_ttl
        self._cond = asyncio.Condition()
        self._active = 0
        self._committing = False
        self._drain_all = False
        self._exact_inflight: dict[int, int] = defaultdict(int)
        # Set by the reconciler while a generation-pausing (cpu-destination)
        # stage occupies the engine: admit holds new work until the stage
        # clears; nothing queues behind the pause.
        self._staging_pause = False
        # lease_key -> (exact version, monotonic expiry). Pruned lazily.
        self._leases: dict[str, tuple[int, float]] = {}

    @property
    def active_requests(self) -> int:
        return self._active

    @property
    def staging_pause(self) -> bool:
        return self._staging_pause

    @staging_pause.setter
    def staging_pause(self, active: bool) -> None:
        self._staging_pause = active

    async def clear_staging_pause(self) -> None:
        """Clear the staging hold and wake held admissions — under the cond
        lock, matching how commit clears ``_committing``."""
        async with self._cond:
            self._staging_pause = False
            self._cond.notify_all()

    def _rejection(self, c: VersionConstraint) -> dict[str, Any] | None:
        served = self._served_version()
        applied = served.version if served else None
        # A gate without a reader serves nothing versioned, so reject as retryable rather
        # than serve unversioned. A reconciler always attaches one whose version starts at
        # the boot ref, so None here only occurs in standalone gate use.
        if applied is None or not c.satisfied_by(applied):
            target = c.exact_version if c.exact_version is not None else c.min_version
            return {
                "type": "WeightVersionNotReady",
                "target_version": target,
                "applied": applied,
                "message": f"served version {applied} does not satisfy {c}",
            }
        return None

    def _commit_ready(self) -> bool:
        if self.commit_mode == "in_place" and not self._drain_all:
            return not any(self._exact_inflight.values())
        return self._active == 0

    def _prune_leases(self, now: float) -> None:
        expired = [k for k, (_, exp) in self._leases.items() if exp <= now]
        for k in expired:
            del self._leases[k]

    def leases_snapshot(self) -> dict[int, int]:
        """Live (unexpired) lease count per version, for ``/server_info``."""
        self._prune_leases(time.monotonic())
        counts: dict[int, int] = defaultdict(int)
        for version, _ in self._leases.values():
            counts[version] += 1
        return dict(counts)

    def leases_blocking(self, target_version: int) -> bool:
        """True when a live lease pins a version below ``target_version``."""
        now = time.monotonic()
        self._prune_leases(now)
        return any(version < target_version for version, _ in self._leases.values())

    async def hold_for_leases(self, target_version: int) -> None:
        """Wait until no live lease names a version below ``target_version``.

        No-op for ``quiesce`` gates and when leases are disabled (ttl 0).
        Stragglers renewing below-target leases extend the wait; pins at or
        above the target 409 and never block. Each wait wakes at the soonest
        blocker expiry.
        """
        if self.commit_mode != "in_place" or self._lease_ttl <= 0:
            return
        async with self._cond:
            while True:
                now = time.monotonic()
                self._prune_leases(now)
                blockers = [
                    exp
                    for version, exp in self._leases.values()
                    if version < target_version
                ]
                if not blockers:
                    return
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._cond.wait(), timeout=min(blockers) - now
                    )

    @asynccontextmanager
    async def admit(
        self,
        constraint: VersionConstraint | None = None,
        *,
        lease_key: str | None = None,
    ):
        """Admit one request under a single lock acquisition, yielding the version it
        is served on. Raises :class:`ConstraintUnmet` if the constraint can't be met.
        An admitted exact-pin with a ``lease_key`` renews that key's lease so
        think-time still holds the replica on its version."""
        c = constraint or VersionConstraint()
        async with self._cond:
            # Stock-shaped hold: wait out a running commit and a
            # generation-pausing stage alike; no error surfaces.
            await self._cond.wait_for(
                lambda: not self._committing and not self._staging_pause
            )
            error = self._rejection(c)
            if error is not None:
                self._on_reject(error)
                raise ConstraintUnmet(error)
            served = self._served_version()
            self._active += 1
            if c.exact_version is not None:
                self._exact_inflight[c.exact_version] += 1
                if lease_key is not None and self._lease_ttl > 0:
                    self._leases[lease_key] = (
                        c.exact_version,
                        time.monotonic() + self._lease_ttl,
                    )
        try:
            yield served
        finally:
            async with self._cond:
                self._active -= 1
                if c.exact_version is not None:
                    self._exact_inflight[c.exact_version] -= 1
                    if not self._exact_inflight[c.exact_version]:
                        del self._exact_inflight[c.exact_version]
                self._cond.notify_all()

    async def commit(
        self,
        *,
        apply: Callable[[], Awaitable[None]],
        on_applied: Callable[[], None],
        pause: Callable[[], Awaitable[None]] | None = None,
        resume: Callable[[], Awaitable[None]] | None = None,
        drain_all: bool = False,
        target_version: int | None = None,
    ) -> None:
        """Wait for the commit point, close the gate, apply, flip the served version
        (``on_applied``) while the gate is held, then reopen. ``on_applied`` runs only
        after a successful apply; in ``in_place`` the flip happens before ``resume``.
        ``drain_all`` marks an incompatible transition (a boot reset): drain and gate
        every request regardless of mode — rolling requests may cross a compatible
        weight update, never a change of lineage (stitch#32).
        The ``on_hold_start`` / ``on_hold_end`` hooks (given at construction) bracket
        the pre-close lease hold (if any).
        """
        # Fire the hooks only when the hold can actually run (the same guard
        # hold_for_leases applies), so a no-op hold never surfaces as one.
        can_hold = (
            not drain_all
            and target_version is not None
            and self.commit_mode == "in_place"
            and self._lease_ttl > 0
        )
        if can_hold and self._on_hold_start is not None:
            self._on_hold_start()
        try:
            while True:
                if can_hold:
                    await self.hold_for_leases(target_version)
                # Re-check under the lock before closing: a lease admitted
                # between the hold returning and admission closing would
                # otherwise be committed past and its next request 409s. Close
                # admission before draining (stitch#32), else a new in_place
                # request can straddle a boot reset.
                async with self._cond:
                    if can_hold and self.leases_blocking(target_version):
                        continue
                    self._committing = True
                    self._drain_all = drain_all
                    self._cond.notify_all()
                    break
        finally:
            if can_hold and self._on_hold_end is not None:
                self._on_hold_end()
        try:
            async with self._cond:
                await self._cond.wait_for(self._commit_ready)
            if self.commit_mode == "in_place" and pause is not None:
                await pause()
                try:
                    await apply()
                    on_applied()
                finally:
                    if resume is not None:
                        await resume()
            else:
                await apply()
                on_applied()
        finally:
            async with self._cond:
                self._committing = False
                self._drain_all = False
                self._cond.notify_all()


class Reconciler:
    """Converges one replica to the store's ``latest`` pointer: stage the chain,
    commit once, flip the served version. Admission is delegated to the composed
    :class:`AdmissionGate` (``self.gate``), which reads ``self.applied`` under its
    own lock while commits flip that same attribute through ``gate.commit``. A run
    change restores the immutable boot checkpoint, so one run's chain is never
    mistaken for another's."""

    def __init__(
        self,
        *,
        store: Store,
        engine: Engine,
        run_id: str,
        boot_version: int = 0,
        commit_mode: CommitMode = "in_place",
        flush_cache_on_commit: bool = False,
        version_lease_ttl: float = 0.0,
        debug_requests: bool = False,
        reconcile_interval: float = 5.0,
    ) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        if boot_version < 0:
            raise ValueError("boot_version must be non-negative")
        self.store = store
        self.engine = engine
        # Only a cpu-destination stage pauses generation (the engine pauses its
        # scheduler around the stage RPC); disk-destination stages are pure file
        # I/O. An engine that declares no mode keeps stock behavior: no pause.
        self._stage_pauses_generation = (
            getattr(engine, "delta_update_mode", "disk") == "cpu"
        )
        self.flush_cache_on_commit = flush_cache_on_commit
        self.run_id = run_id
        self.boot_version = boot_version
        self.applied = VersionRef(run_id, boot_version)
        # The gate consults the served version only as a reader; a 409 wakes us.
        self.gate = AdmissionGate(
            commit_mode=commit_mode,
            served_version=lambda: self.applied,
            on_reject=self._on_reject,
            on_hold_start=self._on_hold_start,
            on_hold_end=self._on_hold_end,
            version_lease_ttl=version_lease_ttl,
        )
        self.debug_requests = debug_requests
        self.reconcile_interval = reconcile_interval
        self.sync_state = SyncState.IDLE
        self._hold_prior = SyncState.IDLE
        self.last_error: str | None = None
        self._target_version: int | None = None
        # Latches after first catch-up unless the replica becomes terminal.
        self.ready = False
        self.metrics: dict[str, Any] = {}
        self._boot_monotonic = time.monotonic()
        self._catchup_passes = 0
        self._task: asyncio.Task[None] | None = None
        self._wake_pending = False
        self._destination_ready = False
        self._destination_init_error: str | None = None
        self._periodic_task: asyncio.Task[None] | None = None
        self._terminal_error: UnrecoverableSidecarError | None = None
        self._terminal_error_event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        # A replica enters rotation only after it can both serve and follow the
        # version stream. Initialization also releases boot-checkpoint files before
        # reconciliation refreshes a shared backing store.
        await self._initialize_update_destination()
        if self._terminal_error is not None:
            return
        await self.reconcile()
        if self.reconcile_interval > 0 and self._terminal_error is None:
            self._periodic_task = asyncio.create_task(self._periodic_reconcile())

    async def shutdown(self) -> None:
        if self._periodic_task is not None:
            self._periodic_task.cancel()
            with suppress(BaseException):
                await self._periodic_task

    async def _periodic_reconcile(self) -> None:
        # Convergence backstop: re-check the pointer so a replica that missed its wake (raced the
        # publish, or a lost best-effort wake) still catches up before the next publish.
        while True:
            await asyncio.sleep(self.reconcile_interval)
            self.wake()

    async def _initialize_update_destination(self) -> None:
        try:
            await self.engine.initialize_update_destination(self.boot_version)
            self._destination_ready = True
        except UnrecoverableSidecarError as exc:
            self._destination_init_error = str(exc)
            self._record_terminal_error(exc)
            logger.exception(
                "weight update destination initialization failed terminally"
            )
        except Exception as exc:  # noqa: BLE001
            self._destination_init_error = str(exc)
            error = UnrecoverableEngineError(
                "weight update destination initialization failed: "
                f"{type(exc).__name__}: {exc}"
            )
            self._record_terminal_error(error)
            logger.exception(
                "weight update destination initialization failed terminally"
            )

    def server_info(self) -> dict[str, Any]:
        # applied = version on the GPU (the pool reads it to see which version each replica has);
        # ready = has caught up to the live pointer at least once, latched — the routing gate (/health).
        return {
            "ready": self.ready,
            "applied": self.applied.identity if self.applied else None,
            "applied_version": self.applied.version if self.applied else None,
            "leases": self.gate.leases_snapshot(),
            "sync_state": self.sync_state.value,
            "staging_pause": self.gate.staging_pause,
            "target_version": self._target_version,
            "reason": self.last_error,
            "run_id": self.run_id,
            "commit_mode": self.gate.commit_mode,
            "active_requests": self.gate.active_requests,
            "update_destination_ready": self._destination_ready,
            "update_destination_error": self._destination_init_error,
            "terminal_error": str(self._terminal_error)
            if self._terminal_error
            else None,
            "metrics": self.metrics,
        }

    def readiness_reason(self) -> str:
        """Why /health is still 503, so a not-yet-admitted replica reads as 'catching up', not broken."""
        if self._destination_init_error:
            return (
                "weight update destination initialization failed: "
                f"{self._destination_init_error}"
            )
        if not self._destination_ready:
            return "initializing weight update destination"
        if self.last_error:
            return f"sync error: {self.last_error}"
        applied = self.applied.identity if self.applied else "boot"
        return f"catching up to live version (applied={applied}, state={self.sync_state.value})"

    def expects_engine_progress(self) -> bool:
        """Whether this replica should currently make inference progress."""
        # A cpu-destination stage or a commit RPC can occupy the engine's HTTP
        # loop for minutes (a first cpu-mode delta compiles the full weight
        # image), so a health probe timing out then is not an independent
        # engine-health signal. Disk-destination stages are pure file I/O: the
        # engine keeps serving, and the watchdog keeps probing.
        if not self.ready:
            return False
        if self.sync_state is SyncState.COMMITTING:
            return False
        return not (
            self.sync_state is SyncState.STAGING and self._stage_pauses_generation
        )

    def _on_hold_start(self) -> None:
        """Surface a gate-internal lease hold as HOLDING so drain selection
        and the watchdog stay live (admission open, engine serving)."""
        self._hold_prior = self.sync_state
        self.sync_state = SyncState.HOLDING

    def _on_hold_end(self) -> None:
        self.sync_state = self._hold_prior

    def _clear_live_target(self) -> None:
        # No live target: a stale metrics.target_version must not advertise
        # one through /server_info.
        self._target_version = None
        self.metrics.pop("target_version", None)

    async def wait_for_terminal_error(self) -> None:
        """Raise once reconciliation proves this replica must be replaced."""
        await self._terminal_error_event.wait()
        assert self._terminal_error is not None
        raise self._terminal_error

    def _record_terminal_error(self, error: UnrecoverableSidecarError) -> None:
        if self._terminal_error is None:
            self.ready = False
            self._terminal_error = error
            self._terminal_error_event.set()

    def _on_reject(self, error: dict[str, Any]) -> None:
        self.wake()  # a 409 is our cue to catch up

    def wake(self) -> None:
        """Nudge reconciliation now (a publish wake or a 409).

        Multiple wakes coalesce because each pass reads the authoritative pointer. A
        wake during an active pass requests one more pass instead of being dropped.
        """
        if self._terminal_error is not None:
            return
        if self._task is not None and not self._task.done():
            self._wake_pending = True
            return
        self._task = asyncio.get_running_loop().create_task(self.reconcile())

    async def reconcile(self) -> None:
        """Loop until caught up to the store's (run, latest); on error, record it and
        stop — a later wake or poll retries."""
        while True:
            self._wake_pending = False
            try:
                caught_up = await self._reconcile_once()
            except UnrecoverableSidecarError as exc:
                self.last_error = str(exc)
                self.sync_state = SyncState.ERROR
                self._clear_live_target()
                self._record_terminal_error(exc)
                logger.exception("reconcile failed terminally")
                return
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                self.sync_state = SyncState.ERROR
                self._clear_live_target()
                logger.exception("reconcile failed")
                if self._wake_pending:
                    continue
                return
            if caught_up:
                if self._wake_pending:
                    continue
                self.sync_state = SyncState.IDLE
                self._clear_live_target()
                if not self.ready:
                    logger.info(
                        "caught up to v%d in %d pass(es), %.0fs — entering rotation",
                        self.applied.version if self.applied else 0,
                        self._catchup_passes,
                        time.monotonic() - self._boot_monotonic,
                    )
                self.ready = True
                return
            await asyncio.sleep(1.0)

    def _behind(self, pointer: VersionRef | None) -> bool:
        if pointer is None:
            return False
        if self.applied is None or pointer.run_id != self.applied.run_id:
            return True
        return pointer.version > self.applied.version

    def _range_has_weight_changes(self, target: VersionManifest) -> bool:
        """Return whether the catch-up range can change any checkpoint bytes.

        An empty delta has no payload files. Skipping the GPU update is safe only
        when every version since the applied checkpoint is an empty delta.
        """
        applied = self.applied
        ref = target.ref
        if (
            applied is None
            or applied.run_id != ref.run_id
            or ref.version <= applied.version
        ):
            return True
        for v in range(ref.version, applied.version, -1):
            m = (
                target
                if v == ref.version
                else self.store.read_manifest(VersionRef(ref.run_id, v))
            )
            if m.kind is not VersionKind.DELTA or m.files:
                return True
        return False

    async def _advance_to_newer_head(
        self,
        pointer: VersionRef,
        target: VersionManifest,
        source_dir: str,
        m: dict[str, Any],
        *,
        error_key: str,
        warning: str,
        info: str,
    ) -> tuple[VersionRef, VersionManifest, str]:
        """Re-read the head after a long hold/stage; if it advanced within the
        same run (never across — a run change is a reset, not a coalesce),
        adopt the newer target so the wait never commits an obsolete version."""
        try:
            await asyncio.to_thread(self.store.refresh)
            latest = await asyncio.to_thread(self.store.read_pointer)
            if (
                latest is None
                or latest.run_id != pointer.run_id
                or latest.version <= pointer.version
            ):
                return pointer, target, source_dir
            latest_target = await asyncio.to_thread(self.store.read_manifest, latest)
            latest_source_dir = await asyncio.to_thread(self.store.materialize, latest)
        except Exception as exc:  # noqa: BLE001
            m[error_key] = str(exc)
            logger.warning(warning, pointer.version, exc_info=True)
            return pointer, target, source_dir
        logger.info(info, pointer.version, latest.version)
        return latest, latest_target, latest_source_dir

    async def _reconcile_once(self) -> bool:
        async with self._lock:
            m: dict[str, Any] = {}
            try:
                return await self._reconcile_once_measured(m)
            except Exception as exc:
                m["error"] = str(exc)
                raise
            finally:
                if (
                    len(m) > 1 or "error" in m
                ):  # a no-work pass leaves the last breakdown alone
                    m["at"] = time.time()
                    self.metrics = m
                    timings = {k: v for k, v in m.items() if k.endswith("_s")}
                    if timings:
                        if not self.ready:
                            self._catchup_passes += 1
                        logger.info(
                            "catch-up pass v%s->v%s timing(s): %s",
                            m.get("applied_version"),
                            m.get("target_version"),
                            timings,
                        )

    async def _reconcile_once_measured(self, m: dict[str, Any]) -> bool:
        await asyncio.to_thread(self.store.refresh)
        pointer = await asyncio.to_thread(self.store.read_pointer)
        if pointer is None:
            return True
        if self.applied is None or pointer.run_id != self.applied.run_id:
            await self._switch_run(pointer.run_id)
        if not self._behind(pointer):
            return True

        self.sync_state = SyncState.FETCHING
        self.last_error = None
        m["target_version"] = pointer.version
        m["applied_version"] = self.applied.version if self.applied else -1
        target = await asyncio.to_thread(self.store.read_manifest, pointer)
        source_dir = await asyncio.to_thread(self.store.materialize, pointer)
        logger.info(
            "catch-up: %s -> v%d, preparing weight update",
            "boot" if self.applied is None else f"v{self.applied.version}",
            pointer.version,
        )

        has_weight_changes = await asyncio.to_thread(
            self._range_has_weight_changes, target
        )

        def on_applied() -> None:
            self.applied = pointer

        if not has_weight_changes:
            # Nothing changed across the applied→target range: advance the
            # version without preparing or loading byte-identical weights.
            await self.gate.commit(
                apply=self._commit_noop,
                on_applied=on_applied,
                target_version=pointer.version,
            )
            m["skipped_weight_update"] = True
        else:
            initial_pointer = pointer
            self._target_version = pointer.version
            if (
                self.gate.commit_mode == "in_place"
                and self.gate.leases_blocking(pointer.version)
            ):
                # A cpu-destination stage pauses generation for minutes, so wait
                # out live leases first — admission open, engine serving.
                # HOLDING is not a stage/commit: the watchdog stays fully live.
                self.sync_state = SyncState.HOLDING
                with _timed(m, "hold_s"):
                    await self.gate.hold_for_leases(pointer.version)
                # The hold can last minutes; re-read the head so coalescing
                # stages the newest target, never a stale one.
                pointer, target, source_dir = await self._advance_to_newer_head(
                    pointer,
                    target,
                    source_dir,
                    m,
                    error_key="hold_coalesce_error",
                    warning="could not inspect a newer target after the lease hold; "
                    "staging v%d",
                    info="catch-up: lease hold advanced head v%d -> v%d before staging",
                )
                self._target_version = pointer.version
            # Preparation runs while serving. Re-read the head once before the
            # commit so a slow stage does not force an avoidable GPU update to an
            # already-obsolete version. One re-check bounds the work when the
            # trainer publishes continuously.
            self.sync_state = SyncState.STAGING
            # A cpu-destination stage pauses generation, possibly for minutes:
            # hold new admissions for the pause rather than enqueue abandoned
            # requests behind the paused engine. Disk-destination stages serve
            # throughout.
            self.gate.staging_pause = self._stage_pauses_generation
            try:
                with _timed(m, "stage_s"):
                    await self.engine.stage(target, source_dir)
                    staged_pointer = pointer
                    pointer, target, source_dir = await self._advance_to_newer_head(
                        pointer,
                        target,
                        source_dir,
                        m,
                        error_key="coalesce_error",
                        warning="could not inspect a newer target; committing staged v%d",
                        info="catch-up: staging advanced head v%d -> v%d before commit",
                    )
                    if pointer is not staged_pointer:
                        await self.engine.stage(target, source_dir)
            finally:
                await self.gate.clear_staging_pause()

            if pointer != initial_pointer:
                m["initial_target_version"] = initial_pointer.version
                m["target_version"] = pointer.version
                m["coalesced_versions"] = pointer.version - initial_pointer.version
            logger.info("catch-up: loading v%d into the engine", pointer.version)

            async def apply() -> None:
                self.sync_state = SyncState.COMMITTING
                with _timed(m, "commit_s"):
                    await self.engine.commit(
                        target,
                        flush_cache=self.flush_cache_on_commit,
                    )

            try:
                await self.gate.commit(
                    apply=apply,
                    on_applied=on_applied,
                    pause=self.engine.pause,
                    resume=self.engine.resume,
                    target_version=pointer.version,
                )
            except UnrecoverableSidecarError:
                raise
            except Exception as exc:
                raise UnrecoverableEngineError(
                    "weight commit failed; live engine state is uncertain"
                ) from exc
        self.sync_state = SyncState.FETCHING
        pointer = await asyncio.to_thread(self.store.read_pointer)
        return not self._behind(pointer)  # a mid-pass publish is next tick's work

    async def _commit_noop(self) -> None:
        self.sync_state = SyncState.COMMITTING

    async def _switch_run(self, new_run: str | None) -> None:
        """Select a new run at its boot version, resetting patched weights to the
        boot checkpoint.

        A reset commits with ``drain_all`` because it is incompatible with
        in-flight requests. Selecting the boot run on an unpatched engine only
        updates attribution and does not pause generation.
        """
        old_run = self.applied.run_id if self.applied else None
        logger.info("run change %r -> %r: restoring boot checkpoint", old_run, new_run)
        # Reset if weights differ from the boot checkpoint, or a prior error may
        # have left them dirty (stitch#32).
        was_patched = self.applied is not None and (
            self.applied.version != self.boot_version or self.last_error is not None
        )

        async def apply() -> None:
            if was_patched:
                self.sync_state = SyncState.COMMITTING
                await self.engine.reset()

        def on_applied() -> None:
            self.applied = VersionRef(new_run, self.boot_version)
            self.last_error = None

        try:
            await self.gate.commit(
                apply=apply,
                on_applied=on_applied,
                pause=self.engine.pause if was_patched else None,
                resume=self.engine.resume if was_patched else None,
                drain_all=True,
            )
        except UnrecoverableSidecarError:
            raise
        except Exception as exc:
            raise UnrecoverableEngineError(
                "boot checkpoint restore failed; live engine state is uncertain"
            ) from exc
