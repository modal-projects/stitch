"""Generic rollout server sync manager."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol

from stitch.bulletin import BulletinBoard
from stitch.protocol import (
    EngineAdapter,
    SyncState,
    VersionManifest,
    WeightVersionPolicy,
    evaluate_version_policy,
)


logger = logging.getLogger(__name__)


class PolicyViolation(Exception):
    """A request's weight-version policy cannot be satisfied by this server."""

    def __init__(self, error: Mapping[str, Any]) -> None:
        super().__init__(error["error"]["message"])
        self.error = error


class RolloutSyncManager(Protocol):
    """The surface stitch.servers.sglang.create_app drives."""

    debug_requests: bool
    current_version: int

    async def startup_sync(self) -> None: ...

    async def server_info(self) -> dict[str, Any]: ...

    def request_context(self, policy: WeightVersionPolicy | None = None) -> Any:
        ...


class WeightSyncManager:
    """Local rollout server sync manager."""

    def __init__(
        self,
        *,
        board: BulletinBoard,
        engine: EngineAdapter,
        run_id: str | None = None,
        debug_requests: bool = False,
    ) -> None:
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
            "current_run_id": self.current_run_id,
            "current_version": self.current_version,
            "latest_seen_version": self.latest_seen_version,
            "queued_target_version": self.queued_target_version,
            "sync_state": self.sync_state.value,
            "last_sync_error": self.last_sync_error,
            "sync_task_active": self._sync_task is not None and not self._sync_task.done(),
        }

    def _policy_error(self, policy: WeightVersionPolicy) -> dict[str, Any] | None:
        return evaluate_version_policy(self.current_version, policy)

    def _on_policy_violation(self, error: dict[str, Any]) -> None:
        if error["error"]["type"] == "WeightVersionNotReady":
            self.queue_sync(error["error"]["target_version"])

    @asynccontextmanager
    async def request_context(self, policy: WeightVersionPolicy | None = None):
        policy = policy or WeightVersionPolicy()
        error = self._policy_error(policy)
        if error is not None:
            self._on_policy_violation(error)
            raise PolicyViolation(error)
        yield self.current_version

    async def commit_version(
        self,
        *,
        apply: Callable[[], Awaitable[None]],
        on_applied: Callable[[], None],
        pause: Callable[[], Awaitable[None]] | None = None,
        resume: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        paused = False
        try:
            if pause is not None:
                await pause()
                paused = True
            await apply()
            on_applied()
        finally:
            if paused and resume is not None:
                await resume()

    async def validate_policy(self, policy: WeightVersionPolicy) -> tuple[bool, int, Mapping[str, Any] | None]:
        """Advisory pre-check. The authoritative check is in request_context."""
        error = self._policy_error(policy)
        if error is not None and error["error"]["type"] == "WeightVersionNotReady":
            self.queue_sync(error["error"]["target_version"])
        return error is None, self.current_version, error

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
        the new chain. The re-materialize runs under the engine pause/reset/resume
        sequence just like a version commit, so no in-flight request decodes across
        the weight wipe.
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
            await self.board.refresh()
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
            expected_base = self.current_version
            target_manifest: VersionManifest | None = None
            for version in range(self.current_version + 1, latest + 1):
                manifest = self.board.read_manifest(self.current_run_id, version)
                if manifest.base_version != expected_base:
                    raise RuntimeError(
                        f"cannot apply version {version}: manifest base "
                        f"{manifest.base_version} != expected {expected_base}"
                    )
                expected_base = version
                target_manifest = manifest
            assert target_manifest is not None  # latest > current_version, so the loop ran
            target_path = str(self.board.version_dir(self.current_run_id, latest))

            # Compose the tail and reload once: apply_manifest(target) replays
            # every delta from the applied version up to `latest` host-side, then
            # does a single engine reload — not one reload per intermediate version.
            self.sync_state = SyncState.PREPARING

            async def apply(manifest: VersionManifest = target_manifest, version_path: str = target_path) -> None:
                self.sync_state = SyncState.COMMITTING
                await self.engine.apply_manifest(manifest, version_path)

            await self.commit_version(
                apply=apply,
                on_applied=lambda: setattr(self, "current_version", latest),
                pause=self.engine.pause_generation,
                resume=self.engine.continue_generation,
            )
            self.sync_state = SyncState.PREFETCHING

            # Reached the board's latest for this run as captured at pass start; a
            # version published mid-pass is picked up by the next reconcile tick.
            return self.current_version >= latest
