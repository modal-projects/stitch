"""Active consolidation: the router's drain-victim policy.

Victim selection/eligibility, the drain condition math, hysteresis counters,
and the shared-Dict victim record. The registry (``cookbook.common.router``)
keeps polling and drives the policy through ``DrainPolicy.update``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from cookbook.common.router import ContainerInfo

logger = logging.getLogger(__name__)

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


def consolidation_dict(app_name: str) -> modal.Dict:
    """The run's consolidation record (single fleet-wide drain victim), namespaced
    to the run app. Persisted outside /loads so a registry restart or a victim's
    poll blip cannot silently clear or duplicate the victim."""
    return modal.Dict.from_name(f"{app_name}-consolidation", create_if_missing=True)


class _LocalDict:
    """In-memory stand-in for the consolidation modal.Dict (tests, single-process)."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self.get = SimpleNamespace(aio=self._get)
        self.put = SimpleNamespace(aio=self._put)

    async def _get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    async def _put(self, key: str, value: Any) -> None:
        self._store[key] = value


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


class DrainPolicy:
    """The drain policy's mutable state: hysteresis counters + the shared record."""

    def __init__(self, rollout_concurrency: int | None, consolidation: Any) -> None:
        self.rollout_concurrency = rollout_concurrency
        self.consolidation = consolidation
        self._drain_pending: tuple[int, str] | None = None
        self._drain_pending_polls = 0
        self._drain_clear_polls = 0

    async def update(self, containers: list[ContainerInfo]) -> None:
        """Mark/unmark the single fleet-wide drain victim.

        The record lives in the shared consolidation dict (survives registry
        restart). Unmarked when applied_version changes, no replica targets
        newer, or the victim left — each only after DRAIN_CLEAR_HYSTERESIS_POLLS
        consecutive failing polls. A new victim requires
        DRAIN_HYSTERESIS_POLLS consecutive polls of the same candidate.
        """
        if self.rollout_concurrency is None:
            # Drain policy disabled: no victim selection, no record mutation.
            return
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
