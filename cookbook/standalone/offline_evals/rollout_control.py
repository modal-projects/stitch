"""Router-owned session lifecycle and reconcile control — the rollout policy loop.

Session records carry a pinned version, a last-seen timestamp, and a tombstone.
Once per registry poll, ``RolloutPolicy.update`` counts live sessions per
replica, marks stale zero-live replicas pending-reconcile in the shared
rollout-control Dict, and nudges them toward the ``latest`` pointer with the
stock sidecar ``POST /wake``. Marks surface in the ``/loads`` snapshot as
``draining`` — the one exclusion concept proxies honor, so a mark costs no new
per-request RPC. The loop is single-writer: exactly one registry container
drives it. The sidecar's body-pin 409 is the correctness backstop — a bug here
costs availability, never wrong weights. The policy records which replicas it
commanded; an applied version that advances while unmarked is an uncommanded
flip (e.g. a mismatched pin reaching a held replica via the stock 409 wake)
and is logged at error level.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from cookbook.standalone.offline_evals.eval_router import EvalContainerInfo

logger = logging.getLogger(__name__)

# A session record with no placement within session_ttl is treated as
# tombstoned. Last resort only — explicit tombstones (POST /sessions/{id}/end)
# are the primary end-of-session signal.
DEFAULT_SESSION_TTL_SECONDS = 1800.0

# Consolidation: a victim is marked only after the drain condition holds for
# this many consecutive registry polls (hysteresis against blips).
CONSOLIDATION_HYSTERESIS_POLLS = 5
# Consolidate only when the old version's live sessions fit on the k-1
# survivors with this safety margin below rollout_concurrency.
CONSOLIDATION_SAFETY_FACTOR = 0.7
# A mark is cleared only after this many consecutive polls where the replica
# has reached the pointer (or left): a single /server_info blip must not
# readmit a mid-reconcile replica.
MARK_CLEAR_HYSTERESIS_POLLS = 3
# A marked replica that hasn't flipped is re-nudged after this many polls...
RECONCILE_RENUDGE_POLLS = 10
# ...at most this many times per pointer (bounded retries + log, never wedge).
MAX_RECONCILE_NUDGES = 3

KIND_RETIRE = "retire"  # zero-live replica catching up to the pointer
KIND_CONSOLIDATE = "consolidate"  # drained old-version victim (one fleet-wide)

_MARKS_KEY = "marks"


def rollout_control_dict(app_name: str) -> modal.Dict:
    """The run's pending-reconcile marks, namespaced to the run app. Persisted
    outside /loads so a registry restart cannot silently clear or duplicate a
    mark mid-reconcile."""
    return modal.Dict.from_name(f"{app_name}-rollout-control", create_if_missing=True)


# ── session records ──────────────────────────────────────────────────────────
# A session record is the route-entry list the placement path persists per
# session id (MRU-first): plain JSON dicts carrying
# ``version``/``last_seen``/``tombstoned`` on every entry.


def session_is_live(record: list[dict], now: float, session_ttl: float) -> bool:
    """A session lives at its freshest route entry while that entry is neither
    tombstoned nor TTL-expired."""
    if not record:
        return False
    freshest = record[0]
    if freshest["tombstoned"]:
        return False
    return now - freshest["last_seen"] <= session_ttl


def live_session_counts(
    sessions: dict[str, list[dict]], now: float, session_ttl: float
) -> dict[str, int]:
    """live[replica] = count of session records at that replica: not tombstoned,
    not expired."""
    counts: dict[str, int] = {}
    for record in sessions.values():
        if session_is_live(record, now, session_ttl):
            task_id = record[0]["task_id"]
            counts[task_id] = counts.get(task_id, 0) + 1
    return counts


async def tombstone_session(session_routes: Any, session_id: str) -> None:
    """Mark a session's record tombstoned. Idempotent: an unknown or already
    tombstoned session is a no-op."""
    record = await session_routes.get.aio(session_id, [])
    if not record:
        return
    for entry in record:
        entry["tombstoned"] = True
    await session_routes.put.aio(session_id, record)


# ── consolidation ────────────────────────────────────────────────────────────


def select_consolidation_victim(
    containers: list[EvalContainerInfo],
    live_counts: dict[str, int],
    pointer_version: int,
    rollout_concurrency: int,
) -> tuple[int, str] | None:
    """The consolidation candidate: ``(version, task_id)`` to drain, or None.

    Oldest applied version V behind the pointer with k>=2 ready, unmarked
    replicas holding live sessions. Drain when the live sessions at V fit on
    the k-1 survivors under ``rollout_concurrency * CONSOLIDATION_SAFETY_FACTOR``
    and every survivor's load is below ``rollout_concurrency``. Victim is
    min(task_id).
    """
    by_version: dict[int, list[EvalContainerInfo]] = {}
    for container in containers:
        if (
            container.applied_version is None
            or container.draining
            or not container.ready
        ):
            continue
        if live_counts.get(container.task_id, 0) <= 0:
            continue
        by_version.setdefault(container.applied_version, []).append(container)

    for version in sorted(by_version):
        if version >= pointer_version:
            continue  # at the pointer: consolidation destination, never a victim
        group = by_version[version]
        k = len(group)
        if k < 2:
            continue
        victim = min(group, key=lambda c: c.task_id)
        survivors = [c for c in group if c.task_id != victim.task_id]
        total_live = sum(live_counts.get(c.task_id, 0) for c in group)
        if total_live > (k - 1) * rollout_concurrency * CONSOLIDATION_SAFETY_FACTOR:
            continue
        if any(c.load_stale for c in survivors):
            # A stale survivor load is unverifiable spare capacity — never
            # approve a drain on a frozen (possibly zero) reading.
            continue
        if any(c.load >= rollout_concurrency for c in survivors):
            continue
        return version, victim.task_id
    return None


# ── the policy loop ──────────────────────────────────────────────────────────


class RolloutPolicy:
    """The control loop's mutable state: hysteresis counters, nudge budgets,
    and the shared pending-reconcile marks."""

    def __init__(
        self,
        *,
        session_routes: Any,
        control: Any,
        session_ttl: float,
        rollout_concurrency: int | None,
        reconcile: Callable[[str], Awaitable[None]],
    ) -> None:
        self.session_routes = session_routes
        self.control = control
        self.session_ttl = session_ttl
        # Engine overload threshold — the capacity unit of the consolidation
        # condition. None disables consolidation (no victim is ever selected);
        # the zero-live retire loop is always on.
        self.rollout_concurrency = rollout_concurrency
        self._reconcile = reconcile
        self._consolidation_pending: tuple[int, str] | None = None
        self._consolidation_pending_polls = 0
        self._clear_polls: dict[str, int] = {}
        self._nudges: dict[str, dict[str, int]] = {}
        self._poll_index = 0
        self._expiry_logged: set[str] = set()
        # Last applied version seen per replica; the commanded set is the
        # marks, so an advance while unmarked is an uncommanded flip.
        self._applied_seen: dict[str, int] = {}

    def _alarm_uncommanded_flips(
        self, containers: list[EvalContainerInfo], marks: dict[str, dict]
    ) -> None:
        """Error-log any replica whose applied version advanced since the last
        poll without a pending-reconcile mark — only marked replicas are woken
        by this loop, so an unmarked advance came from outside it."""
        for container in containers:
            applied = container.applied_version
            if applied is None:
                continue
            previous = self._applied_seen.get(container.task_id)
            if (
                previous is not None
                and applied > previous
                and container.task_id not in marks
            ):
                logger.error(
                    "rollout: replica %s flipped v%d -> v%d without a "
                    "registry-commanded wake",
                    container.task_id,
                    previous,
                    applied,
                )
            self._applied_seen[container.task_id] = applied
        present = {container.task_id for container in containers}
        for task_id in list(self._applied_seen):
            if task_id not in present:
                del self._applied_seen[task_id]

    def _log_expired(self, sessions: dict[str, list[dict]], now: float) -> None:
        """One log line per record the TTL tombstones (last resort)."""
        expired = set()
        for session_id, record in sessions.items():
            if not record or record[0]["tombstoned"]:
                continue
            if now - record[0]["last_seen"] > self.session_ttl:
                expired.add(session_id)
                if session_id not in self._expiry_logged:
                    logger.info(
                        "rollout: session %s idle beyond TTL (%.0fs); "
                        "treated as tombstoned",
                        session_id,
                        self.session_ttl,
                    )
        self._expiry_logged = expired

    async def _repoint_sessions(
        self,
        sessions: dict[str, list[dict]],
        victim_id: str,
        survivor: EvalContainerInfo,
        now: float,
    ) -> None:
        """Migrate the victim's live sessions onto the survivor: rewrite their
        route entries so the next request lands there. Eager by design —
        migrated stragglers re-prefill on their survivor."""
        for session_id, record in sessions.items():
            if not session_is_live(record, now, self.session_ttl):
                continue
            if record[0]["task_id"] != victim_id:
                continue
            for entry in record:
                if entry["task_id"] == victim_id:
                    entry["task_id"] = survivor.task_id
                    entry["version"] = survivor.applied_version
                    entry["last_seen"] = now
            await self.session_routes.put.aio(session_id, record)
            logger.info(
                "rollout: re-pointed session %s from victim %s to survivor %s",
                session_id,
                victim_id,
                survivor.task_id,
            )

    async def update(
        self,
        containers: list[EvalContainerInfo],
        pointer_version: int | None,
        now: float,
    ) -> None:
        """One control-loop pass over this poll's container snapshot.

        Annotates the snapshot (``live_sessions`` from the session records,
        ``draining`` for marked replicas), marks stale zero-live replicas and
        at most one consolidation victim pending-reconcile in the shared Dict,
        nudges marked replicas whose active_requests drained to zero, and
        clears marks once applied_version reaches the pointer
        (MARK_CLEAR_HYSTERESIS_POLLS consecutive polls)."""
        self._poll_index += 1
        if pointer_version is None:
            return  # no pointer read yet: nothing to reconcile toward

        sessions = {
            session_id: record
            async for session_id, record in self.session_routes.items.aio()
        }
        marks: dict[str, dict] = dict(await self.control.get.aio(_MARKS_KEY, {}) or {})
        live = live_session_counts(sessions, now, self.session_ttl)
        self._log_expired(sessions, now)
        self._alarm_uncommanded_flips(containers, marks)
        for container in containers:
            container.live_sessions = live.get(container.task_id, 0)
        by_id = {container.task_id: container for container in containers}
        changed = False

        # Consolidation (opt-in): one fleet-wide victim at a time.
        if self.rollout_concurrency is not None and not any(
            mark.get("kind") == KIND_CONSOLIDATE for mark in marks.values()
        ):
            candidate = select_consolidation_victim(
                containers, live, pointer_version, self.rollout_concurrency
            )
            if candidate is None:
                self._consolidation_pending = None
                self._consolidation_pending_polls = 0
            else:
                if candidate == self._consolidation_pending:
                    self._consolidation_pending_polls += 1
                else:
                    self._consolidation_pending = candidate
                    self._consolidation_pending_polls = 1
                if self._consolidation_pending_polls >= CONSOLIDATION_HYSTERESIS_POLLS:
                    version, victim_id = candidate
                    marks[victim_id] = {
                        "version": version,
                        "kind": KIND_CONSOLIDATE,
                        "marked_at": now,
                    }
                    changed = True
                    self._consolidation_pending = None
                    self._consolidation_pending_polls = 0
                    survivor = min(
                        (
                            c
                            for c in by_id.values()
                            if c.applied_version == version and c.task_id != victim_id
                        ),
                        key=lambda c: c.load,
                    )
                    logger.info(
                        "rollout: consolidating v%d: victim %s drains to %s",
                        version,
                        victim_id,
                        survivor.task_id,
                    )
                    await self._repoint_sessions(sessions, victim_id, survivor, now)
        else:
            self._consolidation_pending = None
            self._consolidation_pending_polls = 0

        # Retire: every stale replica with zero live sessions can catch up.
        for container in containers:
            if (
                container.task_id not in marks
                and container.ready
                and not container.draining
                and container.applied_version is not None
                and container.applied_version < pointer_version
                and live.get(container.task_id, 0) == 0
            ):
                marks[container.task_id] = {
                    "version": container.applied_version,
                    "kind": KIND_RETIRE,
                    "marked_at": now,
                }
                changed = True
                logger.info(
                    "rollout: marking %s pending-reconcile "
                    "(applied v%d < pointer v%d, no live sessions)",
                    container.task_id,
                    container.applied_version,
                    pointer_version,
                )

        # Drive marked replicas: exclude from placement, nudge at zero active
        # requests, clear once the flip (or departure) holds for the hysteresis.
        for task_id in list(marks):
            container = by_id.get(task_id)
            keep = (
                container is not None
                and container.applied_version is not None
                and container.applied_version < pointer_version
            )
            if not keep:
                if container is not None:
                    container.draining = True  # status quo across the clear window
                clear_polls = self._clear_polls.get(task_id, 0) + 1
                self._clear_polls[task_id] = clear_polls
                if clear_polls >= MARK_CLEAR_HYSTERESIS_POLLS:
                    del marks[task_id]
                    changed = True
                    self._clear_polls.pop(task_id, None)
                    self._nudges.pop(task_id, None)
                    logger.info(
                        "rollout: replica %s unmarked (reached pointer v%d or left)",
                        task_id,
                        pointer_version,
                    )
                continue
            self._clear_polls.pop(task_id, None)
            container.draining = True  # the exclusion proxies already honor
            nudge = self._nudges.setdefault(
                task_id,
                {"count": 0, "last_poll": -RECONCILE_RENUDGE_POLLS, "pointer": -1},
            )
            if nudge["pointer"] != pointer_version:
                # A new pointer is a fresh target: reset the nudge budget.
                nudge.update(
                    count=0, last_poll=-RECONCILE_RENUDGE_POLLS, pointer=pointer_version
                )
            if container.active_requests > 0:
                continue  # wait for in-flight requests to drain
            if nudge["count"] >= MAX_RECONCILE_NUDGES:
                if not nudge.get("exhausted"):
                    logger.warning(
                        "rollout: replica %s unflipped after %d nudges; "
                        "staying marked, no more nudges",
                        task_id,
                        MAX_RECONCILE_NUDGES,
                    )
                    nudge["exhausted"] = 1
                continue
            if self._poll_index - nudge["last_poll"] >= RECONCILE_RENUDGE_POLLS:
                logger.info(
                    "rollout: POST /wake to replica %s (nudge %d/%d)",
                    task_id,
                    nudge["count"] + 1,
                    MAX_RECONCILE_NUDGES,
                )
                await self._reconcile(container.upstream)
                nudge["count"] += 1
                nudge["last_poll"] = self._poll_index

        if changed:
            await self.control.put.aio(_MARKS_KEY, marks)
