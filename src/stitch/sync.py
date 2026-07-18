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
from stitch.types import SyncState, VersionConstraint, VersionKind, VersionRef

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
        if c.satisfied_by(applied):
            return None
        target = c.exact_version if c.exact_version is not None else c.min_version
        return {
            "type": "WeightVersionNotReady",
            "target_version": target,
            "applied": applied,
            "message": f"served version {applied} does not satisfy {c}",
        }

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
        self.metrics: dict[str, Any] = {}
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
        # ready = serveable (version applied on the GPU); does NOT wait on the base prefetch.
        # prefetch_* expose whether the O(delta) fast path is primed.
        return {
            "ready": self.applied is not None and self.sync_state is not SyncState.ERROR,
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
                return
            await asyncio.sleep(1.0)

    def _behind(self, pointer: VersionRef | None) -> bool:
        if pointer is None:
            return False
        if self.applied is None or pointer.run_id != self.applied.run_id:
            return True
        return pointer.version > self.applied.version

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

    async def _reconcile_once_measured(self, m: dict[str, Any]) -> bool:
        # offload: refresh() may block on I/O.
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
        target = self.store.read_manifest(pointer)
        source_dir = await asyncio.to_thread(self.store.materialize, pointer)
        m["target_version"] = pointer.version

        # Both prefetch and stage write the host checkpoint; wait for the prefetch so they're ordered.
        # If it failed, this stage seeds the full base itself.
        if self._prefetch_task is not None and not self._prefetch_task.done():
            await self._prefetch_task
        if self._prefetch_error is not None:
            m["paid_base_copy"] = True

        # Stage (host-side apply) runs while serving; the gate covers only the reload.
        self.sync_state = SyncState.PREPARING
        with _timed(m, "stage_s"):
            await self.engine.stage(target, source_dir)

        def on_applied() -> None:
            self.applied = pointer

        if target.kind is VersionKind.DELTA and not target.files:
            # Empty delta: advance the version, no reload — weights are byte-identical, KV stays valid.
            await self.commit(apply=self._commit_noop, on_applied=on_applied)
            m["skipped_reload"] = True
        else:
            async def apply() -> None:
                self.sync_state = SyncState.COMMITTING
                with _timed(m, "commit_s"):
                    await self.engine.commit(pointer, flush_cache=self.flush_cache_on_commit)

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
