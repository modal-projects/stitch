"""The per-replica sync brain: an ``AdmissionGate`` (gate rollout requests on the
served version) and a ``Reconciler`` (converge the replica to the store's pointer).

They share one lock and one ``applied`` version so that reporting stays correct:
a request's constraint is checked and its serving version captured under the same
lock the committer holds across the weight apply *and* the version flip.
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
from stitch.stores.base import Store
from stitch.types import SyncState, VersionConstraint, VersionKind, VersionManifest, VersionRef

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

    ``quiesce`` drains all in-flight requests and flushes before applying; ``in_place``
    pauses the engine and lets non-exact requests keep decoding on stale KV (only exact
    pins are drained) — but only across *compatible* commits: an incompatible transition
    (a run switch's base reset) commits with ``drain_all=True``, which drains and gates
    everything regardless of mode. Either way the committer holds the lock across the
    apply and the version flip, so no request is ever admitted seeing the stale version
    on new weights.
    """

    def __init__(self, *, commit_mode: CommitMode = "in_place") -> None:
        self.commit_mode = commit_mode
        self.applied: VersionRef | None = None
        self._cond = asyncio.Condition()
        self._active = 0
        self._committing = False
        self._drain_all = False
        self._exact_inflight: dict[int, int] = defaultdict(int)

    @property
    def active_requests(self) -> int:
        return self._active

    def _gated(self, c: VersionConstraint) -> bool:
        if not self._committing:
            return False
        # in_place gates only exact pins across a compatible commit; drain_all / quiesce gate everything.
        if self._drain_all or self.commit_mode != "in_place":
            return True
        return c.exact_version is not None

    def _rejection(self, c: VersionConstraint) -> dict[str, Any] | None:
        applied = self.applied.version if self.applied else None
        # Until the first sync lands (applied is None) there is no served version to stamp, so
        # reject as retryable rather than serve unversioned; the background reconcile sets it shortly.
        if applied is None or not c.satisfied_by(applied):
            target = c.exact_version if c.exact_version is not None else c.min_version
            return {
                "type": "WeightVersionNotReady",
                "target_version": target,
                "applied": applied,
                "message": f"served version {applied} does not satisfy {c}",
            }
        return None

    def _on_reject(self, error: dict[str, Any]) -> None:
        """Hook, run under the lock, when a request is rejected."""

    def _commit_ready(self) -> bool:
        if self.commit_mode == "in_place" and not self._drain_all:
            return not any(self._exact_inflight.values())
        return self._active == 0

    @asynccontextmanager
    async def admit(self, constraint: VersionConstraint | None = None):
        """Admit one request under a single lock acquisition, yielding the version it
        is served on. Raises :class:`ConstraintUnmet` if the constraint can't be met."""
        c = constraint or VersionConstraint()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._gated(c))
            error = self._rejection(c)
            if error is not None:
                self._on_reject(error)
                raise ConstraintUnmet(error)
            served = self.applied
            self._active += 1
            if c.exact_version is not None:
                self._exact_inflight[c.exact_version] += 1
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
    ) -> None:
        """Wait for the commit point, close the gate, apply, flip the served version
        (``on_applied``) while the gate is held, then reopen. ``on_applied`` runs only
        after a successful apply; in ``in_place`` the flip happens before ``resume``.
        ``drain_all`` marks an incompatible transition (a base reset): drain and gate
        every request regardless of mode — rolling requests may cross a compatible
        weight update, never a change of lineage (stitch#32)."""
        # Close admission before draining (stitch#32), else a new in_place request can straddle a base reset.
        async with self._cond:
            self._committing = True
            self._drain_all = drain_all
            self._cond.notify_all()
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


class Reconciler(AdmissionGate):
    """Converges one replica to the store's ``latest`` pointer: stage the chain,
    reload once, flip the served version. A run change resets to base (the engine
    reseeds), so a run's chain is never mistaken for another's."""

    def __init__(
        self,
        *,
        store: Store,
        engine: Engine,
        run_id: str | None = None,
        commit_mode: CommitMode = "in_place",
        flush_cache_on_commit: bool = False,
        debug_requests: bool = False,
        reconcile_interval: float = 5.0,
    ) -> None:
        super().__init__(commit_mode=commit_mode)
        self.store = store
        self.engine = engine
        self.flush_cache_on_commit = flush_cache_on_commit
        self.run_id = run_id  # static label for server_info, not the active chain's run
        self.debug_requests = debug_requests
        self.reconcile_interval = reconcile_interval
        self.sync_state = SyncState.IDLE
        self.last_error: str | None = None
        self.ready = False  # latches on first catch-up, stays set even when later stale; the /health routing gate
        self.metrics: dict[str, Any] = {}
        self._boot_monotonic = time.monotonic()  # for the "caught up in Ns" join summary
        self._catchup_passes = 0  # reload passes a fresh joiner pays before first ready (it chases the live version)
        self._task: asyncio.Task[None] | None = None
        self._prefetch_task: asyncio.Task[None] | None = None
        self._prefetch_done = False
        self._prefetch_error: str | None = None
        self._periodic_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def startup(self) -> None:
        # Background base seed so the first real stage() is delta-only; a later stage waits on it (see
        # _reconcile_once_measured), so the two writes to the checkpoint never race.
        self._prefetch_task = asyncio.create_task(self._prefetch_base())
        await self.reconcile()
        if self.reconcile_interval > 0:
            self._periodic_task = asyncio.create_task(self._periodic_reconcile())

    async def shutdown(self) -> None:
        for task in (self._periodic_task, self._prefetch_task):
            if task is not None:
                task.cancel()
                with suppress(BaseException):
                    await task

    async def _periodic_reconcile(self) -> None:
        # Convergence backstop: re-check the pointer so a replica that missed its wake (raced the
        # publish, or a lost best-effort wake) still catches up before the next publish.
        while True:
            await asyncio.sleep(self.reconcile_interval)
            self.wake()

    async def _prefetch_base(self) -> None:
        try:
            await self.engine.prefetch()
            self._prefetch_done = True
        except Exception as exc:  # noqa: BLE001 — best-effort; first pull falls back to a full copy
            self._prefetch_error = str(exc)
            logger.exception("base prefetch failed; first sync will pay the full base copy")

    def server_info(self) -> dict[str, Any]:
        # applied = version on the GPU (the pool reads it to see which version each replica has);
        # ready = has caught up to the live pointer at least once, latched — the routing gate (/health).
        return {
            "ready": self.ready,
            "applied": self.applied.identity if self.applied else None,
            "sync_state": self.sync_state.value,
            "reason": self.last_error,
            "run_id": self.run_id,
            "commit_mode": self.commit_mode,
            "active_requests": self._active,
            "prefetch_done": self._prefetch_done,
            "prefetch_error": self._prefetch_error,
            "metrics": self.metrics,
        }

    def readiness_reason(self) -> str:
        """Why /health is still 503, so a not-yet-admitted replica reads as 'catching up', not broken."""
        if self._prefetch_error:
            return f"base seed failed: {self._prefetch_error}"
        if not self._prefetch_done:
            return "seeding base checkpoint"
        if self.last_error:
            return f"sync error: {self.last_error}"
        applied = self.applied.identity if self.applied else "base"
        return f"catching up to live version (applied={applied}, state={self.sync_state.value})"

    def _on_reject(self, error: dict[str, Any]) -> None:
        self.wake()  # a 409 is our cue to catch up

    def wake(self) -> None:
        """Nudge a reconcile now (a publish wake or a 409). Non-cancelling: starts a
        task only if none is running; the running loop re-reads the authoritative pointer."""
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self.reconcile())

    async def reconcile(self) -> None:
        """Loop until caught up to the store's (run, latest); on error, record it and
        stop — a later wake or poll retries."""
        while True:
            try:
                caught_up = await self._reconcile_once()
                if caught_up:
                    # A publish can land mid-commit; re-check before idling.
                    await asyncio.to_thread(self.store.refresh)
                    if self._behind(self.store.read_pointer()):
                        caught_up = False
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                self.sync_state = SyncState.ERROR
                logger.exception("reconcile failed")
                return
            if caught_up:
                self.sync_state = SyncState.IDLE
                if not self.ready:
                    logger.info(
                        "caught up to v%d after %d catch-up pass(es) in %.0fs — entering rotation",
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

    def _touched_names(self, target: VersionManifest) -> list[str] | None:
        """Union of the tensor names touched across the versions this pass applies
        (``applied``+1 .. ``target``), for an engine that can reload only those (O(delta)).
        Returns None — reload everything — when the applied→target range reseeds from a FULL
        anchor or the baseline is unknown: a full reload is always correct, a partial one only
        when the served weights already hold every tensor the deltas didn't touch."""
        applied = self.applied
        ref = target.ref
        if applied is None or applied.run_id != ref.run_id or ref.version <= applied.version:
            return None
        names: set[str] = set()
        for v in range(ref.version, applied.version, -1):
            m = target if v == ref.version else self.store.read_manifest(VersionRef(ref.run_id, v))
            # full reload (None) for either: a FULL anchor in range (the stage reseeds from it), or a
            # non-empty delta with no recorded touched names (an empty union would skip it → stale)
            if m.kind is not VersionKind.DELTA or (m.files and not m.tensor_names):
                return None
            names.update(m.tensor_names)
        return sorted(names)

    async def _reconcile_once(self) -> bool:
        async with self._lock:
            m: dict[str, Any] = {}
            try:
                return await self._reconcile_once_measured(m)
            except Exception as exc:
                m["error"] = str(exc)
                raise
            finally:
                if len(m) > 1 or "error" in m:  # a no-work pass leaves the last breakdown alone
                    m["at"] = time.time()
                    self.metrics = m
                    timings = {k: v for k, v in m.items() if k.endswith("_s")}
                    if timings:
                        if not self.ready:
                            self._catchup_passes += 1
                        logger.info("catch-up pass v%s->v%s timing(s): %s",
                                    m.get("applied_version"), m.get("target_version"), timings)

    async def _reconcile_once_measured(self, m: dict[str, Any]) -> bool:
        # offload: refresh() may block on I/O.
        with _timed(m, "refresh_s"):
            await asyncio.to_thread(self.store.refresh)
        pointer = self.store.read_pointer()
        if pointer is None:
            return True
        if self.applied is None or pointer.run_id != self.applied.run_id:
            await self._switch_run(pointer.run_id)
        if not self._behind(pointer):
            return True

        self.sync_state = SyncState.PREFETCHING
        self.last_error = None
        m["target_version"] = pointer.version
        m["applied_version"] = self.applied.version if self.applied else -1
        with _timed(m, "read_manifest_s"):
            target = self.store.read_manifest(pointer)
        with _timed(m, "materialize_s"):
            source_dir = await asyncio.to_thread(self.store.materialize, pointer)
        logger.info(
            "catch-up: %s -> v%d, staging deltas",
            "base" if self.applied is None else f"v{self.applied.version}",
            pointer.version,
        )

        # Touched tensor names for an O(delta) reload: the union across the versions this pass
        # applies (applied+1 .. target). None => the range reseeds from a FULL anchor, so the
        # engine must reload everything. Computed under the pre-flip `applied` (on_applied moves it).
        with _timed(m, "touched_names_s"):
            weight_names = await asyncio.to_thread(self._touched_names, target)

        # Both prefetch and stage write the host checkpoint; wait for the prefetch so they're ordered.
        # If it failed, this stage seeds the full base itself.
        if self._prefetch_task is not None and not self._prefetch_task.done():
            with _timed(m, "prefetch_wait_s"):
                await self._prefetch_task
        if self._prefetch_error is not None:
            m["paid_base_copy"] = True

        # Stage (host-side apply) runs while serving; the gate covers only the reload.
        self.sync_state = SyncState.PREPARING
        with _timed(m, "stage_s"):
            await self.engine.stage(target, source_dir)

        def on_applied() -> None:
            self.applied = pointer

        if weight_names is not None and not weight_names:
            # Nothing changed across the applied→target range: advance the version, no reload —
            # the weights are byte-identical, so the KV cache stays valid.
            await self.commit(apply=self._commit_noop, on_applied=on_applied)
            m["skipped_reload"] = True
        else:
            m["reload_names"] = "full" if weight_names is None else len(weight_names)
            logger.info(
                "catch-up: reloading v%d (%s)",
                pointer.version,
                "all tensors" if weight_names is None else f"{len(weight_names)} tensors",
            )

            async def apply() -> None:
                self.sync_state = SyncState.COMMITTING
                with _timed(m, "commit_s"):
                    await self.engine.commit(
                        pointer, flush_cache=self.flush_cache_on_commit, weight_names=weight_names
                    )

            await self.commit(
                apply=apply,
                on_applied=on_applied,
                pause=self.engine.pause,
                resume=self.engine.resume,
            )
        self.sync_state = SyncState.PREFETCHING
        return not self._behind(self.store.read_pointer())  # a mid-pass publish is next tick's work

    async def _commit_noop(self) -> None:
        self.sync_state = SyncState.COMMITTING

    async def _switch_run(self, new_run: str | None) -> None:
        """Rebase onto a new run: reset to base (the engine reseeds on the next stage).
        Commits with ``drain_all`` — a base reset is an incompatible transition, so even
        in ``in_place`` mode no rolling request is admitted during, or decodes across,
        the wipe."""
        old_run = self.applied.run_id if self.applied else None
        logger.info("run change %r -> %r: resetting to base", old_run, new_run)
        # Reset if weights are patched (v>0), or a prior error may have left the checkpoint dirty (stitch#32).
        was_patched = self.applied is not None and (
            self.applied.version > 0 or self.last_error is not None
        )

        async def apply() -> None:
            if was_patched:
                await self.engine.reset()

        def on_applied() -> None:
            self.applied = VersionRef(new_run, 0)
            self.last_error = None

        await self.commit(
            apply=apply,
            on_applied=on_applied,
            pause=self.engine.pause,
            resume=self.engine.resume,
            drain_all=True,
        )
