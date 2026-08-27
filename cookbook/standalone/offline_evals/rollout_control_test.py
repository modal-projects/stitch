"""Rollout control loop: session liveness, tombstones/TTL, the retire flip
walk (mark, exclusion, nudge, clear), and consolidation arithmetic + migration
— all against in-memory fakes (no Modal, no cloud)."""

from __future__ import annotations

import asyncio
import logging

from cookbook.standalone.offline_evals.eval_router import (
    EvalContainerInfo,
    route_session,
)
from cookbook.standalone.offline_evals.rollout_control import (
    CONSOLIDATION_HYSTERESIS_POLLS,
    KIND_CONSOLIDATE,
    KIND_RETIRE,
    MARK_CLEAR_HYSTERESIS_POLLS,
    MAX_RECONCILE_NUDGES,
    RolloutPolicy,
    live_session_counts,
    select_consolidation_victim,
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
    concurrency: int | None = None,
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
        rollout_concurrency=concurrency,
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


def test_consolidation_walk() -> None:
    """Trigger arithmetic (drain bar, min-task_id victim, blockers, oldest
    group) then the walk: disabled by default, hysteresis, persisted mark,
    re-point migration, one-victim invariant, reconcile after drain, clear on
    flip."""

    def holders() -> list[EvalContainerInfo]:
        return [_member("ta-0", applied=1), _member("ta-1", applied=1)]

    # k=2, concurrency=10: the drain bar is (k-1) * 10 * 0.7 = 7 live sessions.
    assert select_consolidation_victim(holders(), {"ta-0": 3, "ta-1": 4}, 2, 10) == (
        1,
        "ta-0",
    )
    assert select_consolidation_victim(holders(), {"ta-0": 3, "ta-1": 5}, 2, 10) is None
    # Victim is min(task_id).
    fleet = [_member("ta-b", applied=1), _member("ta-a", applied=1)]
    assert select_consolidation_victim(fleet, {"ta-a": 1, "ta-b": 1}, 2, 10) == (
        1,
        "ta-a",
    )
    # A survivor at the overload bar, or with a stale load reading, blocks.
    fleet = [_member("ta-0", applied=1), _member("ta-1", load=10, applied=1)]
    assert select_consolidation_victim(fleet, {"ta-0": 1, "ta-1": 1}, 2, 10) is None
    fleet = [_member("ta-0", applied=1), _member("ta-1", applied=1, load_stale=True)]
    assert select_consolidation_victim(fleet, {"ta-0": 1, "ta-1": 1}, 2, 10) is None
    # k=1 groups are skipped; a version at the pointer is never a victim;
    # replicas with zero live sessions are retired, not consolidated.
    assert select_consolidation_victim([_member("ta-0")], {"ta-0": 1}, 2, 10) is None
    fleet = [_member("ta-0", applied=2), _member("ta-1", applied=2)]
    assert select_consolidation_victim(fleet, {"ta-0": 1, "ta-1": 1}, 2, 10) is None
    assert select_consolidation_victim(holders(), {"ta-0": 0, "ta-1": 3}, 2, 10) is None
    # Oldest eligible version wins: v1 over-full, v2 (behind pointer 3) drains.
    fleet = [
        _member("ta-0", applied=1),
        _member("ta-1", applied=1),
        _member("ta-2", applied=2),
        _member("ta-3", applied=2),
    ]
    live = {"ta-0": 5, "ta-1": 5, "ta-2": 1, "ta-3": 1}
    assert select_consolidation_victim(fleet, live, 3, 10) == (2, "ta-2")

    async def run() -> None:
        sessions, control = _LocalDict(), _LocalDict()
        # 6 live sessions at v1 (3+3), bar for k=2/concurrency=10 is 7.
        for i, task_id in enumerate(["ta-0"] * 3 + ["ta-1"] * 3):
            await sessions.put.aio(f"s{i}", _record(task_id))

        def fleet(active: int = 3) -> list[EvalContainerInfo]:
            # The victim carries in-flight requests until the drain phase, so
            # the nudge provably waits for active_requests == 0.
            return _fleet(
                _member("ta-0", load=0, applied=1, active=active),
                _member("ta-1", load=1, applied=1),
                _member("ta-2", applied=2),
            )

        # Disabled by default: without rollout_concurrency the same over-full
        # group is never marked (the zero-live retire loop is always on, but
        # every member here holds live sessions).
        disabled, _, _, disabled_calls = _policy(
            session_routes=sessions, control=control
        )
        for _ in range(CONSOLIDATION_HYSTERESIS_POLLS + 1):
            await disabled.update(fleet(), 2, NOW)
        assert (await control.get.aio("marks", {})) == {}, (
            "None disables consolidation; live-holding replicas are never marked"
        )
        assert disabled_calls == []

        policy, _, _, calls = _policy(
            session_routes=sessions, control=control, concurrency=10
        )

        for _ in range(CONSOLIDATION_HYSTERESIS_POLLS - 1):
            current = fleet()
            await policy.update(current, 2, NOW)
            assert not any(c.draining for c in current), "marked before hysteresis"
            assert (await control.get.aio("marks", {})) == {}
        current = fleet()
        await policy.update(current, 2, NOW)
        marks = await control.get.aio("marks")
        assert marks["ta-0"]["kind"] == KIND_CONSOLIDATE
        assert current[0].draining
        # Re-point: the victim's live sessions moved to the lowest-load survivor.
        for i in range(3):
            record = await sessions.get.aio(f"s{i}")
            assert record[0]["task_id"] == "ta-1"
            assert record[0]["version"] == 1
        for i in range(3, 6):
            assert (await sessions.get.aio(f"s{i}"))[0]["task_id"] == "ta-1"

        # One-victim invariant: a second eligible group (ta-1+ta-3 at v1) is not
        # touched while ta-0's consolidation mark stands.
        await sessions.put.aio("s6", _record("ta-3"))
        for _ in range(CONSOLIDATION_HYSTERESIS_POLLS + 1):
            current = fleet() + [_member("ta-3", applied=1)]
            await policy.update(current, 2, NOW)
        assert list(await control.get.aio("marks")) == ["ta-0"]
        assert calls == [], "in-flight requests gate the victim's nudge"

        # Drained: the next poll nudges the victim.
        current = fleet(active=0)
        await policy.update(current, 2, NOW)
        assert calls == ["ta-0:8000"]

        # A restarted policy (fresh state, same Dicts) keeps driving the mark.
        restarted, _, _, _ = _policy(
            session_routes=sessions, control=control, concurrency=10
        )
        current = fleet()
        await restarted.update(current, 2, NOW)
        assert current[0].draining, "the persisted mark survives a registry restart"

        # Flip clears the mark after the hysteresis.
        for _ in range(MARK_CLEAR_HYSTERESIS_POLLS):
            current = fleet()
            current[0].applied_version = 2
            await restarted.update(current, 2, NOW)
        assert (await control.get.aio("marks")) == {}

    asyncio.run(run())
