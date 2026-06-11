"""Generic rollout server sync manager."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol

from stitch.bulletin import BulletinBoard
from stitch.protocol import (
    SyncState,
    VersionManifest,
    WeightVersionPolicy,
    version_not_ready_error,
    version_too_old_error,
)


logger = logging.getLogger(__name__)
CommitWaitPolicy = Literal["quiesce_all", "exact_only"]
CommitMode = Literal["quiesce", "in_place"]


class PolicyViolation(Exception):
    """A request's weight-version policy cannot be satisfied by this server."""

    def __init__(self, error: Mapping[str, Any]) -> None:
        super().__init__(error["error"]["message"])
        self.error = error


class EngineAdapter(Protocol):
    backend: str

    async def flush_cache(self) -> None: ...

    async def apply_manifest(self, manifest: VersionManifest, version_path: str) -> None: ...

    # Required only for commit_mode="in_place".
    async def pause_generation(self) -> None: ...

    async def continue_generation(self) -> None: ...


class WeightSyncManager:
    """Local rollout server sync manager.

    Commit modes:

    - ``quiesce`` (default): wait for active proxied requests per
      ``commit_wait_policy``, flush the engine cache, then apply. Safe on any
      engine build.
    - ``in_place``: pause the engine in place, apply without flushing, and
      continue — in-flight requests resume decoding on their existing KV.
      Cross-version KV isolation comes from the sidecar stamping a composed,
      version-namespaced ``extra_key`` onto every proxied request. Requires
      an engine build with the overlap-drain fix (see
      docs/kv-version-namespace-design.md, "Mandatory engine fixes" #1);
      without it a forward in flight at pause time can race the weight
      mutation. Only exact-version requests are quiesced/gated.
    """

    def __init__(
        self,
        *,
        board: BulletinBoard,
        engine: EngineAdapter,
        run_id: str | None = None,
        commit_wait_policy: CommitWaitPolicy = "quiesce_all",
        commit_mode: CommitMode = "quiesce",
        debug_requests: bool = False,
    ) -> None:
        self.board = board
        self.engine = engine
        self.run_id = run_id
        self.commit_wait_policy = commit_wait_policy
        self.commit_mode = commit_mode
        self.debug_requests = debug_requests
        self.current_version = 0
        self.latest_seen_version = 0
        self.queued_target_version: int | None = None
        self.sync_state = SyncState.IDLE
        self.last_sync_error: str | None = None
        self._sync_task: asyncio.Task[None] | None = None
        self._sync_lock = asyncio.Lock()
        self._active_cond = asyncio.Condition()
        self._active_requests = 0
        self._exact_inflight: dict[int, int] = defaultdict(int)
        # True from the moment the quiesce predicate passes until the engine
        # apply finishes (or fails). While set, request admission is gated:
        # without this, requests arriving while the sync task awaits
        # flush/apply network calls would validate against the stale
        # current_version and could be served on the new weights.
        self._committing = False

    @property
    def active_requests(self) -> int:
        return self._active_requests

    async def startup_sync(self) -> None:
        while True:
            await self.board.refresh()
            latest = self.board.read_latest()
            self.latest_seen_version = max(self.latest_seen_version, latest)
            if latest <= self.current_version:
                return
            await self.sync_to(latest)

    async def server_info(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "backend": self.engine.backend,
            "commit_mode": self.commit_mode,
            "current_version": self.current_version,
            "latest_seen_version": self.latest_seen_version,
            "queued_target_version": self.queued_target_version,
            "sync_state": self.sync_state.value,
            "last_sync_error": self.last_sync_error,
            "sync_task_active": self._sync_task is not None and not self._sync_task.done(),
            "active_requests": self._active_requests,
            "inflight_exact_versions": {
                str(version): count for version, count in sorted(self._exact_inflight.items()) if count
            },
        }

    @asynccontextmanager
    async def request_context(self, policy: WeightVersionPolicy | None = None):
        """Admit one request: gate on in-progress commits, enforce the policy,
        and pin exact versions. Yields the weight version the request is served
        on. Raises :class:`PolicyViolation` when the policy cannot be satisfied.

        The policy check happens after the commit gate, under the same lock the
        committer uses, so the version it sees is the version the engine serves
        the request on.
        """
        policy = policy or WeightVersionPolicy()
        async with self._active_cond:
            await self._active_cond.wait_for(lambda: not self._admission_gated(policy))
            error = self._policy_error(policy)
            if error is not None:
                if error["error"]["type"] == "WeightVersionNotReady":
                    self.queue_sync(error["error"]["target_version"])
                raise PolicyViolation(error)
            start_version = self.current_version
            self._active_requests += 1
            if policy.exact_version is not None:
                self._exact_inflight[int(policy.exact_version)] += 1
        try:
            yield start_version
        finally:
            async with self._active_cond:
                self._active_requests -= 1
                if policy.exact_version is not None:
                    key = int(policy.exact_version)
                    self._exact_inflight[key] -= 1
                    if not self._exact_inflight[key]:
                        del self._exact_inflight[key]
                self._active_cond.notify_all()

    def _admission_gated(self, policy: WeightVersionPolicy) -> bool:
        if not self._committing:
            return False
        if self.commit_mode == "in_place":
            # Non-strict requests cross commits freely: they are stamped with
            # the version current at admission, and a mislabel around the
            # commit is old-era impurity only. Exact pins must not cross.
            return policy.exact_version is not None
        return True

    def _policy_error(self, policy: WeightVersionPolicy) -> dict[str, Any] | None:
        current = self.current_version
        if policy.exact_version is not None:
            target = int(policy.exact_version)
            if current < target:
                return version_not_ready_error(current, target)
            if current > target:
                return version_too_old_error(current, target)
            return None
        if policy.min_required_version is not None and current < int(policy.min_required_version):
            return version_not_ready_error(current, int(policy.min_required_version))
        return None

    async def validate_policy(self, policy: WeightVersionPolicy) -> tuple[bool, int, Mapping[str, Any] | None]:
        """Advisory pre-check. The authoritative check is in request_context."""
        error = self._policy_error(policy)
        if error is not None and error["error"]["type"] == "WeightVersionNotReady":
            self.queue_sync(error["error"]["target_version"])
        return error is None, self.current_version, error

    def queue_sync(self, target_version: int | None = None) -> None:
        target = (
            max(self.board.read_latest(), self.latest_seen_version)
            if target_version is None
            else int(target_version)
        )
        if target <= self.current_version:
            return
        self.queued_target_version = max(target, self.queued_target_version or 0)
        if self.sync_state is SyncState.IDLE:
            self.sync_state = SyncState.QUEUED
        if self._sync_task is None or self._sync_task.done():
            self._sync_task = asyncio.get_running_loop().create_task(self.sync_to())

    async def sync_to(self, target_version: int | None = None) -> None:
        if target_version is not None and int(target_version) > self.current_version:
            self.queued_target_version = max(int(target_version), self.queued_target_version or 0)

        while True:
            target = self.queued_target_version
            if target is None or target <= self.current_version:
                self.queued_target_version = None
                self.sync_state = SyncState.IDLE
                return

            try:
                reached_target = await self._sync_once(target)
            except Exception as exc:  # noqa: BLE001
                self.last_sync_error = str(exc)
                self.sync_state = SyncState.ERROR
                logger.exception("Weight sync failed")
                return

            if reached_target:
                continue
            await asyncio.sleep(1.0)

    async def _sync_once(self, target_version: int) -> bool:
        async with self._sync_lock:
            await self.board.refresh()
            latest = self.board.read_latest()
            self.latest_seen_version = max(self.latest_seen_version, latest)
            target = min(int(target_version), latest)
            if target <= self.current_version:
                return self.current_version >= int(target_version)

            self.sync_state = SyncState.PREFETCHING
            self.last_sync_error = None
            for version in range(self.current_version + 1, target + 1):
                manifest = self.board.read_manifest(version)
                if manifest.base_version != self.current_version:
                    raise RuntimeError(
                        f"cannot apply version {version}: manifest base "
                        f"{manifest.base_version} != current {self.current_version}"
                    )
                self.sync_state = SyncState.PREPARING
                await self._wait_for_commit_point()
                self.sync_state = SyncState.COMMITTING
                try:
                    version_path = str(self.board.version_dir(version))
                    if self.commit_mode == "in_place":
                        await self.engine.pause_generation()
                        try:
                            await self.engine.apply_manifest(manifest, version_path)
                            # Bump before continue: requests admitted from here
                            # on are stamped with the new namespace, while
                            # already-admitted ones keep their old stamp (the
                            # accepted old-era mislabel window).
                            self.current_version = version
                        finally:
                            await self.engine.continue_generation()
                    else:
                        await self.engine.flush_cache()
                        await self.engine.apply_manifest(manifest, version_path)
                        self.current_version = version
                finally:
                    async with self._active_cond:
                        self._committing = False
                        self._active_cond.notify_all()
                self.sync_state = SyncState.PREFETCHING

            if self.queued_target_version is not None and self.queued_target_version <= self.current_version:
                self.queued_target_version = None
                self.sync_state = SyncState.IDLE
            return self.current_version >= int(target_version)

    async def _wait_for_commit_point(self) -> None:
        async with self._active_cond:
            if self.commit_mode == "in_place":
                # Exact pins must not cross a commit; summed over all versions
                # so at most one exact version is ever live.
                await self._active_cond.wait_for(lambda: not any(self._exact_inflight.values()))
            elif self.commit_wait_policy == "quiesce_all":
                await self._active_cond.wait_for(lambda: self._active_requests == 0)
            else:
                await self._active_cond.wait_for(lambda: not self._exact_inflight.get(self.current_version))
            # Set under the same lock acquisition that observed the quiesce
            # predicate, so no request can be admitted between the two.
            self._committing = True
