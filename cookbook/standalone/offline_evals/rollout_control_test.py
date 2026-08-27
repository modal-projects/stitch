"""Rollout control loop: session liveness, tombstones/TTL, and the retire flip
walk (mark, exclusion, nudge, clear) — all against in-memory fakes (no Modal,
no cloud)."""

from __future__ import annotations

import asyncio
import logging

from cookbook.standalone.offline_evals.eval_router import (
    EvalContainerInfo,
    route_session,
)
from cookbook.standalone.offline_evals.rollout_control import (
    KIND_RETIRE,
    MARK_CLEAR_HYSTERESIS_POLLS,
    MAX_RECONCILE_NUDGES,
    RolloutPolicy,
    live_session_counts,
    tombstone_session,
)
from cookbook.standalone.offline_evals.testing import _LocalDict

NOW = 1_000_000.0
TTL = 1800.0


def _member(
    task_id: str,
    *,
    load: int = 0,
    applied: int | None = 1,
    ready: bool = True,
    active: int = 0,
    load_stale: bool = False,
) -> EvalContainerInfo:
    return EvalContainerInfo(
        task_id=task_id,
        upstream=f"{task_id}:8000",
        load=load,
        applied_version=applied,
        ready=ready,
        active_requests=active,
        load_stale=load_stale,
    )


def _record(
    task_id: str, *, version: int | None = 1, last_seen: float = NOW
) -> list[dict]:
    return [
        {
            "task_id": task_id,
            "version": version,
            "last_seen": last_seen,
            "tombstoned": False,
        }
    ]


def _policy(
    *,
    session_routes: _LocalDict | None = None,
    control: _LocalDict | None = None,
    ttl: float = TTL,
) -> tuple[RolloutPolicy, _LocalDict, _LocalDict, list[str]]:
    session_routes = session_routes if session_routes is not None else _LocalDict()
    control = control if control is not None else _LocalDict()
    calls: list[str] = []

    async def reconcile(upstream: str) -> None:
        calls.append(upstream)

    policy = RolloutPolicy(
        session_routes=session_routes,
        control=control,
        session_ttl=ttl,
        reconcile=reconcile,
    )
    return policy, session_routes, control, calls


def _fleet(*members: EvalContainerInfo) -> list[EvalContainerInfo]:
    return list(members)


def test_session_liveness_ttl_and_tombstone(caplog) -> None:
    sessions = {
        "fresh": _record("ta-0"),
        "expired": _record("ta-1", last_seen=NOW - TTL - 1),
        "tombstoned": [{**_record("ta-2")[0], "tombstoned": True}],
        "empty": [],
        # The freshest entry decides: an ended session is dead everywhere.
        "ended": [
            {**_record("ta-5")[0], "tombstoned": True},
            _record("ta-6")[0],
        ],
    }
    counts = live_session_counts(sessions, NOW, TTL)
    assert counts == {"ta-0": 1}

    policy, _, _, _ = _policy()
    with caplog.at_level(
        logging.INFO, logger="cookbook.standalone.offline_evals.rollout_control"
    ):
        policy._log_expired(sessions, NOW)
        policy._log_expired(sessions, NOW)
    expired_logs = [r for r in caplog.records if "treated as tombstoned" in r.message]
    assert len(expired_logs) == 1, "one log line per TTL-expired record, once"
    assert {r.args[0] for r in expired_logs} == {"expired"}

    async def run() -> None:
        session_routes = _LocalDict()
        await session_routes.put.aio(
            "s1", _record("ta-0") + _record("ta-1", last_seen=NOW - 10)
        )
        await tombstone_session(session_routes, "s1")
        record = await session_routes.get.aio("s1")
        assert all(entry["tombstoned"] for entry in record)
        assert live_session_counts({"s1": record}, NOW, TTL) == {}
        # Idempotent: re-tombstone and unknown sessions are no-ops.
        await tombstone_session(session_routes, "s1")
        await tombstone_session(session_routes, "never-seen")
        assert await session_routes.get.aio("never-seen", None) is None

    asyncio.run(run())


def test_retire_flip_walk(caplog) -> None:
    """Pointer moves -> zero-live stale replica is marked -> exclusion visible to
    placement (the mark beats the pin) -> wake nudge -> mark clears on the flip
    (3-poll hysteresis). Nudges wait for active_requests == 0, re-nudge on a
    bounded schedule, and a new pointer resets the budget. A commanded flip is
    silent; an unmarked replica advancing is error-logged as uncommanded."""

    async def run() -> None:
        policy, _, control, calls = _policy()
        fleet = _fleet(_member("ta-0", applied=1, active=2), _member("ta-1", applied=2))
        await policy.update(fleet, 2, NOW)
        assert calls == [], "in-flight requests gate the nudge"
        fleet = _fleet(_member("ta-0", applied=1), _member("ta-1", applied=2))
        await policy.update(fleet, 2, NOW)
        marks = await control.get.aio("marks")
        assert list(marks) == ["ta-0"]
        assert marks["ta-0"]["kind"] == KIND_RETIRE
        assert fleet[0].draining and not fleet[1].draining
        assert calls == ["ta-0:8000"], "first nudge fires at active_requests == 0"

        # The mark is the exclusion proxies honor: ta-0 is never placed, even
        # though unpinned ta-1 and pinned-to-v1 both would otherwise consider it.
        containers = {c.task_id: c for c in fleet}
        picked = await route_session(_LocalDict(), "s", containers, 4)
        assert picked.task_id == "ta-1"
        picked = await route_session(_LocalDict(), "s", containers, 4, exact_version=1)
        assert picked is None, "marked replica never selected, even for a pinned match"
        containers["ta-9"] = _member("ta-9", applied=1)
        picked = await route_session(_LocalDict(), "s", containers, 4, exact_version=1)
        assert picked.task_id == "ta-9", "only the marked v1 holder is excluded"

        # No re-nudge before the re-nudge interval; re-nudges are bounded, so a
        # replica that never flips goes quiet instead of wedging the loop.
        fleet = _fleet(_member("ta-0", applied=1), _member("ta-1", applied=2))
        for _ in range(5):
            await policy.update(fleet, 2, NOW)
        assert calls == ["ta-0:8000"], "no re-nudge inside the interval"
        for _ in range(35):
            await policy.update(fleet, 2, NOW)
        assert calls == ["ta-0:8000"] * MAX_RECONCILE_NUDGES, "bounded re-nudges"

        # A new pointer is a fresh target: the nudge budget resets (ta-1 is now
        # also stale and zero-live, so it is retire-marked and nudged as well).
        await policy.update(fleet, 3, NOW)
        assert calls.count("ta-0:8000") == MAX_RECONCILE_NUDGES + 1
        assert "ta-1:8000" in calls

        # The flip needs 3 polls at the pointer to clear the mark.
        for _ in range(MARK_CLEAR_HYSTERESIS_POLLS - 1):
            fleet = _fleet(_member("ta-0", applied=3), _member("ta-1", applied=3))
            await policy.update(fleet, 3, NOW)
            assert fleet[0].draining, "status quo across the clear window"
            assert (await control.get.aio("marks")) != {}
        fleet = _fleet(_member("ta-0", applied=3), _member("ta-1", applied=3))
        await policy.update(fleet, 3, NOW)
        assert (await control.get.aio("marks")) == {}
        fleet = _fleet(_member("ta-0", applied=3), _member("ta-1", applied=3))
        await policy.update(fleet, 3, NOW)
        assert not fleet[0].draining and not fleet[1].draining
        assert calls.count("ta-0:8000") == MAX_RECONCILE_NUDGES + 1
        assert calls.count("ta-1:8000") == 1, "no nudges after the flip"

        # Uncommanded-flip alarm: ta-1 reaches a new pointer with no mark
        # (flipped outside the loop) -> one error log; ta-0's flip under its
        # mark is commanded -> silent.
        policy, _, control, _ = _policy()
        fleet = _fleet(_member("ta-0", applied=1), _member("ta-1", applied=2))
        await policy.update(fleet, 2, NOW)
        assert list(await control.get.aio("marks")) == ["ta-0"]
        with caplog.at_level(
            logging.ERROR,
            logger="cookbook.standalone.offline_evals.rollout_control",
        ):
            fleet = _fleet(_member("ta-0", applied=1), _member("ta-1", applied=3))
            await policy.update(fleet, 3, NOW)
            fleet = _fleet(_member("ta-0", applied=3), _member("ta-1", applied=3))
            await policy.update(fleet, 3, NOW)
        flips = [r for r in caplog.records if "without a" in r.message]
        assert len(flips) == 1 and flips[0].levelname == "ERROR"
        assert flips[0].args == ("ta-1", 2, 3)

    asyncio.run(run())
