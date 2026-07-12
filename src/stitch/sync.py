"""Generic rollout server sync manager."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol

from stitch.bulletin import BulletinBoard
from stitch.protocol import (
    EngineAdapter,
    SyncState,
    VersionManifest,
    WeightVersionPolicy,
    evaluate_version_policy,
)


logger = logging.getLogger(__name__)
CommitMode = Literal["quiesce", "in_place"]


class PolicyViolation(Exception):
    """A request's weight-version policy cannot be satisfied by this server."""

    def __init__(self, error: Mapping[str, Any]) -> None:
        super().__init__(error["error"]["message"])
        self.error = error


class RolloutSyncManager(Protocol):
    """The surface stitch.servers.sglang.create_app drives. WeightSyncManager
    implements it, and both the bulletin-board sidecar and the hot-load provider
    sidecar build one — they differ only in board backend and front-door surface.

    create_app also probes optional members defensively (shutdown_sync, and the
    sync-route trio queue_sync/queued_target_version/sync_state); those are not
    part of this required contract.
    """

    debug_requests: bool
    current_version: int
    active_requests: int

    async def startup_sync(self) -> None: ...

    async def server_info(self) -> dict[str, Any]: ...

    def request_context(self, policy: WeightVersionPolicy | None = None) -> Any: ...


class RolloutAdmissionGate:
    """Shared request-admission + commit gate for rollout sync managers.

    Owns the active-request accounting and the ``_committing`` gate, and
    centralizes the lock discipline that keeps version reporting correct: a
    request's policy is checked and its serving version captured under the same
    ``_active_cond`` acquisition the committer uses, and commits hold the gate
    across the engine apply *and* the version advance (cleared only after).
    Subclasses provide ``current_version`` and override the hooks for their
    policy / admission / exact-pin specifics.
    """

    def __init__(self, *, commit_mode: CommitMode = "quiesce") -> None:
        self.commit_mode = commit_mode
        self._active_cond = asyncio.Condition()
        self._active_requests = 0
        self._committing = False
        # Exact-version pins in flight, summed across versions, so a commit can
        # wait for strict traffic to drain. Tracked on the gate (not per
        # subclass) so the bulletin-board manager and the hot-load shim share
        # the in_place commit semantics.
        self._exact_inflight: dict[int, int] = defaultdict(int)

    @property
    def active_requests(self) -> int:
        return self._active_requests

    @property
    def inflight_exact_versions(self) -> dict[str, int]:
        return {str(version): count for version, count in sorted(self._exact_inflight.items()) if count}

    def _admission_gated(self, policy: WeightVersionPolicy) -> bool:
        if not self._committing:
            return False
        if self.commit_mode == "in_place":
            # Non-strict requests cross commits freely: stamped with the version
            # current at admission, a mislabel around the commit is old-era
            # impurity only. Exact pins must not cross.
            return policy.exact_version is not None
        return True

    def _policy_error(self, policy: WeightVersionPolicy) -> dict[str, Any] | None:
        return evaluate_version_policy(self.current_version, policy)

    def _on_admit(self, policy: WeightVersionPolicy) -> None:
        if policy.exact_version is not None:
            self._exact_inflight[int(policy.exact_version)] += 1

    def _on_release(self, policy: WeightVersionPolicy) -> None:
        if policy.exact_version is not None:
            key = int(policy.exact_version)
            self._exact_inflight[key] -= 1
            if not self._exact_inflight[key]:
                del self._exact_inflight[key]

    def _on_policy_violation(self, error: dict[str, Any]) -> None:
        """Hook run under the lock when admission is rejected."""

    def _commit_ready(self) -> bool:
        """The predicate a commit waits on before closing the admission gate:
        in_place drains only exact pins (so at most one exact version is ever
        live); quiesce drains all in-flight proxied requests."""
        if self.commit_mode == "in_place":
            return not any(self._exact_inflight.values())
        return self._active_requests == 0

    @asynccontextmanager
    async def request_context(self, policy: WeightVersionPolicy | None = None):
        """Admit one request: gate on in-progress commits, enforce the policy,
        and capture the serving version — all under one ``_active_cond``
        acquisition, so the yielded version is exactly what the engine serves
        the request on. Raises :class:`PolicyViolation` when the policy fails.
        """
        policy = policy or WeightVersionPolicy()
        async with self._active_cond:
            await self._active_cond.wait_for(lambda: not self._admission_gated(policy))
            error = self._policy_error(policy)
            if error is not None:
                self._on_policy_violation(error)
                raise PolicyViolation(error)
            start_version = self.current_version
            self._active_requests += 1
            self._on_admit(policy)
        try:
            yield start_version
        finally:
            async with self._active_cond:
                self._active_requests -= 1
                self._on_release(policy)
                self._active_cond.notify_all()

    async def _begin_commit(self, ready: Callable[[], bool]) -> None:
        """Wait for the quiesce predicate, then close the admission gate under
        the same lock acquisition (so no request slips in between)."""
        async with self._active_cond:
            await self._active_cond.wait_for(ready)
            self._committing = True

    async def _end_commit(self) -> None:
        async with self._active_cond:
            self._committing = False
            self._active_cond.notify_all()

    async def commit_version(
        self,
        *,
        apply: Callable[[], Awaitable[None]],
        on_applied: Callable[[], None],
        pause: Callable[[], Awaitable[None]] | None = None,
        resume: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Drive one commit through the gate: wait for the commit point and
        close the admission gate, apply the new weights, advance the served
        version (``on_applied``) while the gate is still held, then reopen.

        In ``in_place`` mode the engine is paused around the apply and resumed in
        a finally, with the version advanced before resume so new admissions see
        the new namespace. On failure the gate (and pause) are unwound and the
        served version is left unchanged — ``on_applied`` runs only after a
        successful apply.
        """
        await self._begin_commit(self._commit_ready)
        try:
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
            await self._end_commit()


class WeightSyncManager(RolloutAdmissionGate):
    """Local rollout server sync manager.

    Commit modes:

    - ``quiesce`` (default): wait for active proxied requests per
      ``commit_wait_policy``, flush the engine cache, then apply. Safe on any
      engine build.
    - ``in_place``: pause the engine in place, apply without flushing, and
      continue — in-flight requests resume decoding on their existing KV.
      Cross-version KV isolation comes from the sidecar stamping a composed,
      version-namespaced ``extra_key`` onto every proxied request. Only
      exact-version requests are quiesced/gated.
    """

    def __init__(
        self,
        *,
        board: BulletinBoard,
        engine: EngineAdapter,
        run_id: str | None = None,
        commit_mode: CommitMode = "quiesce",
        debug_requests: bool = False,
    ) -> None:
        super().__init__(commit_mode=commit_mode)
        self.board = board
        self.engine = engine
        self.run_id = run_id
        self.debug_requests = debug_requests
        # The active *chain* identity, parsed from the pointer (`<run_id>/weight_vN`).
        # Distinct from `self.run_id` (the static replica label). None for the
        # run-less stitch layout, where the manager never switches.
        self.current_run_id: str | None = None
        self.current_version = 0
        self.latest_seen_version = 0
        self.queued_target_version: int | None = None
        self.sync_state = SyncState.IDLE
        self.last_sync_error: str | None = None
        # Per-stage wall-clock breakdown of the most recent sync pass (seconds),
        # exposed via server_info()["metrics"] so a scraper can attribute sync
        # latency to board refresh / host apply / drain / pause / reload without
        # log access. Written whole (never mutated in place) at the end of each
        # pass, including failed ones (with an "error" key).
        self.last_sync_metrics: dict[str, Any] = {}
        self._sync_task: asyncio.Task[None] | None = None
        self._sync_lock = asyncio.Lock()

    async def startup_sync(self) -> None:
        prepare = getattr(self.engine, "prepare", None)
        if prepare is not None:
            await prepare()
        # Converge to the board's current (run_id, latest): _sync_once follows a
        # run switch (re-materialize + reset to v0) and replays the chain.
        await self.sync_to()

    async def server_info(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "backend": self.engine.backend,
            "commit_mode": self.commit_mode,
            "current_run_id": self.current_run_id,
            "current_version": self.current_version,
            "latest_seen_version": self.latest_seen_version,
            "queued_target_version": self.queued_target_version,
            "sync_state": self.sync_state.value,
            "last_sync_error": self.last_sync_error,
            "sync_task_active": self._sync_task is not None and not self._sync_task.done(),
            "active_requests": self._active_requests,
            "inflight_exact_versions": self.inflight_exact_versions,
            "metrics": self.last_sync_metrics,
        }

    def _on_policy_violation(self, error: dict[str, Any]) -> None:
        if error["error"]["type"] == "WeightVersionNotReady":
            self.queue_sync(error["error"]["target_version"])

    def queue_sync(self, target_version: int | None = None) -> None:
        run_id, latest = self.board.read_latest()
        self.latest_seen_version = max(self.latest_seen_version, latest)
        hint = int(target_version) if target_version is not None else 0
        # A run change is always work (the pointer moved to a fresh chain, even if
        # its version number is lower than what we serve); within a run it's work
        # only when the target exceeds what we've applied.
        needs_switch = run_id != self.current_run_id
        if not needs_switch and max(latest, hint) <= self.current_version:
            return
        self.queued_target_version = max(latest, hint, self.queued_target_version or 0)
        if self.sync_state is SyncState.IDLE:
            self.sync_state = SyncState.QUEUED
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.get_running_loop().create_task(self.sync_to())

    async def sync_to(self, target_version: int | None = None) -> None:
        # target_version is an optional lower-bound hint from a waker; the
        # authoritative target is always the board's current (run_id, latest), so
        # we converge to it — and follow a run switch — regardless of the hint.
        if target_version is not None:
            self.queued_target_version = max(int(target_version), self.queued_target_version or 0)

        while True:
            try:
                reached = await self._sync_once()
                if reached:
                    # A wake/publish can land while _sync_once is already
                    # committing an older board snapshot. Re-check before going
                    # idle so the active sync task does not drop that work.
                    await self.board.refresh()
                    run_id, latest = self.board.read_latest()
                    self.latest_seen_version = max(self.latest_seen_version, latest)
                    queued = self.queued_target_version or 0
                    if queued > latest:
                        # A wake hint is only a prompt to look at the board; the
                        # durable head is the authority. A hint the head never
                        # reaches (stale, or from a previous incarnation) must
                        # not pin the manager in QUEUED forever — anything
                        # published after this pass converges via the periodic
                        # reconcile.
                        self.queued_target_version = queued = latest
                    if run_id != self.current_run_id or max(latest, queued) > self.current_version:
                        reached = False
            except Exception as exc:  # noqa: BLE001
                self.last_sync_error = str(exc)
                self.sync_state = SyncState.ERROR
                logger.exception("Weight sync failed")
                return

            if reached:
                self.queued_target_version = None
                self.sync_state = SyncState.IDLE
                return
            await asyncio.sleep(1.0)

    async def _switch_run(self, new_run_id: str | None) -> None:
        """Rebase onto a new run's chain: re-materialize base and reset to v0.

        A new run forks at base (slime sets ``base_version=000000`` for v1 of every
        run), so switching chains means discarding the current weights and replaying
        the new chain. The re-materialize runs through the commit gate (pause →
        reset → resume) exactly like a version commit, so no in-flight request
        decodes across the weight wipe.
        """
        logger.info(
            "run change %r -> %r: re-materializing base, resetting to v0",
            self.current_run_id,
            new_run_id,
        )
        reset = getattr(self.engine, "reset", None)
        was_patched = self.current_version > 0

        async def _apply() -> None:
            if reset is not None and was_patched:
                await reset()

        def _applied() -> None:
            self.current_run_id = new_run_id
            self.current_version = 0
            self.latest_seen_version = 0
            self.last_sync_error = None

        await self.commit_version(
            apply=_apply,
            on_applied=_applied,
            pause=self.engine.pause_generation,
            resume=self.engine.continue_generation,
        )

    async def _sync_once(self) -> bool:
        async with self._sync_lock:
            metrics: dict[str, Any] = {}
            try:
                return await self._sync_once_measured(metrics)
            except Exception as exc:
                metrics["error"] = str(exc)
                raise
            finally:
                # A pass that found no work leaves the previous breakdown alone.
                if len(metrics) > 1 or "error" in metrics:
                    metrics["recorded_at"] = time.time()
                    self.last_sync_metrics = metrics

    async def _sync_once_measured(self, metrics: dict[str, Any]) -> bool:
        t0 = time.perf_counter()
        await self.board.refresh()
        metrics["board_refresh_s"] = round(time.perf_counter() - t0, 3)
        run_id, latest = self.board.read_latest()
        if run_id != self.current_run_id:
            await self._switch_run(run_id)
        self.latest_seen_version = max(self.latest_seen_version, latest)
        if latest <= self.current_version:
            return True

        self.sync_state = SyncState.PREFETCHING
        self.last_sync_error = None

        # Verify the whole tail is a contiguous chain before touching the
        # engine, so a broken chain fails before any pause/apply. apply_deltas
        # re-verifies each step's base_version internally as it replays.
        t0 = time.perf_counter()
        expected_base = self.current_version
        target_manifest: VersionManifest | None = None
        tail_transition_files = 0
        # Union of the tail's touched tensor names, for engines that support
        # partial reloads. Any manifest without names makes the tail unknown.
        tail_weights: set[str] | None = set()
        for version in range(self.current_version + 1, latest + 1):
            manifest = self.board.read_manifest(self.current_run_id, version)
            if manifest.base_version != expected_base:
                raise RuntimeError(
                    f"cannot apply version {version}: manifest base "
                    f"{manifest.base_version} != expected {expected_base}"
                )
            expected_base = version
            tail_transition_files += len(manifest.transition_files)
            if tail_weights is not None and manifest.transition_weights:
                tail_weights.update(manifest.transition_weights)
            elif manifest.transition_files:
                tail_weights = None  # a version with files but unknown names
            target_manifest = manifest
        assert target_manifest is not None  # latest > current_version, so the loop ran
        target_path = str(self.board.version_dir(self.current_run_id, latest))
        metrics["manifest_verify_s"] = round(time.perf_counter() - t0, 3)
        metrics["target_version"] = latest
        metrics["tail_versions"] = latest - self.current_version
        metrics["tail_transition_files"] = tail_transition_files
        if tail_weights is not None:
            metrics["tail_weights"] = len(tail_weights)

        # Compose the tail and reload once: the adapter replays every delta from
        # the applied version up to `latest` host-side, then does a single engine
        # reload — not one reload per intermediate version.
        self.sync_state = SyncState.PREPARING

        # An adapter that offers the staged split gets the host-side patch done
        # BEFORE the commit gate: between reloads nothing reads the local
        # checkpoint (weights live on the GPU), so staging can run while the
        # engine still serves and the pause covers only the engine reload.
        stage = getattr(self.engine, "stage_manifest", None)
        commit = getattr(self.engine, "commit_manifest", None)
        staged = stage is not None and commit is not None
        if staged:
            t0 = time.perf_counter()
            stage_detail = await stage(target_manifest, target_path)
            metrics["stage_s"] = round(time.perf_counter() - t0, 3)
            if stage_detail:
                metrics["stage_detail"] = stage_detail

        def on_applied() -> None:
            self.current_version = latest

        # A tail that changes zero bytes (an all-empty disk-delta publish, the
        # common case for low-churn QAT) still must advance the served version,
        # but reloading identical weights would pause the engine for nothing —
        # commit the version bump without pause, flush, or reload. Stale KV
        # stays numerically valid because the weights are byte-identical.
        if target_manifest.backend == "disk_delta" and tail_transition_files == 0:

            async def apply_nothing() -> None:
                self.sync_state = SyncState.COMMITTING

            t0 = time.perf_counter()
            await self.commit_version(apply=apply_nothing, on_applied=on_applied)
            metrics["gate_s"] = round(time.perf_counter() - t0, 3)
            metrics["skipped_noop_reload"] = True
            logger.info(
                "v=%s: empty delta tail — advanced version without engine reload", latest
            )
        else:

            async def apply(manifest: VersionManifest = target_manifest, version_path: str = target_path) -> None:
                # quiesce flushes before applying; in_place skips the flush
                # (the gate paused the engine and stale KV resumes as-is).
                if self.commit_mode != "in_place":
                    t = time.perf_counter()
                    await self.engine.flush_cache()
                    metrics["flush_s"] = round(time.perf_counter() - t, 3)
                self.sync_state = SyncState.COMMITTING
                t = time.perf_counter()
                if staged:
                    commit_detail = await commit(
                        manifest,
                        version_path,
                        weight_names=sorted(tail_weights) if tail_weights else None,
                    )
                    if commit_detail:
                        metrics["commit_detail"] = commit_detail
                else:
                    await self.engine.apply_manifest(manifest, version_path)
                metrics["commit_s"] = round(time.perf_counter() - t, 3)

            async def pause() -> None:
                t = time.perf_counter()
                await self.engine.pause_generation()
                metrics["pause_s"] = round(time.perf_counter() - t, 3)

            async def resume() -> None:
                t = time.perf_counter()
                await self.engine.continue_generation()
                metrics["resume_s"] = round(time.perf_counter() - t, 3)

            # on_applied bumps current_version under the gate (before
            # continue_generation in in_place mode), so a request can never be
            # admitted observing the stale version on mutated weights.
            t0 = time.perf_counter()
            await self.commit_version(
                apply=apply,
                on_applied=on_applied,
                pause=pause,
                resume=resume,
            )
            metrics["gate_s"] = round(time.perf_counter() - t0, 3)
            # The gate time not spent in a measured sub-stage is the wait for
            # in-flight requests to drain (exact pins in in_place mode; all
            # proxied requests in quiesce mode).
            metrics["drain_wait_s"] = round(
                metrics["gate_s"]
                - sum(metrics.get(k, 0.0) for k in ("flush_s", "commit_s", "pause_s", "resume_s")),
                3,
            )
        self.sync_state = SyncState.PREFETCHING

        # Reached the board's latest for this run as captured at pass start; a
        # version published mid-pass is picked up by the next reconcile tick.
        return self.current_version >= latest
