"""Drain-victim policy: eligibility/condition math, hysteresis, shared-Dict record."""

from __future__ import annotations

import asyncio
from typing import Any

from cookbook.common.consolidation import (
    DRAIN_CLEAR_HYSTERESIS_POLLS,
    DRAIN_HYSTERESIS_POLLS,
    _LocalDict,
    select_drain_victim,
)
from cookbook.common.router import ContainerInfo, _RegistryApp


def _member(
    task_id: str,
    *,
    load: int = 0,
    applied: int | None = 1,
    leases: dict[str, int] | None = None,
    sync_state: str | None = "IDLE",
    target: int | None = 2,
    draining: bool = False,
) -> ContainerInfo:
    return ContainerInfo(
        task_id=task_id,
        upstream=f"{task_id}:8000",
        load=load,
        applied_version=applied,
        leases=leases or {},
        sync_state=sync_state,
        target_version=target,
        draining=draining,
    )


def _starving_fleet() -> list[ContainerInfo]:
    return [
        _member("ta-0", leases={"1": 2}),
        _member("ta-1", leases={"1": 2}),
        _member("ta-2", applied=1, leases={}, sync_state="HOLDING"),
    ]


def _drain_registry(
    consolidation: Any | None = None, victim: dict | None = None
) -> tuple[_RegistryApp, Any]:
    consolidation = consolidation if consolidation is not None else _LocalDict()
    if victim is not None:
        asyncio.run(consolidation.put.aio("victim", victim))
    registry = _RegistryApp(
        app_name="app",
        upstream_cls="Server",
        rollout_concurrency=10,
        consolidation=consolidation,
    )
    return registry, consolidation


def test_drain_disabled_without_rollout_concurrency() -> None:
    # Default opt-out: no rollout_concurrency, no victim selection — even with
    # a fleet that would otherwise drain, and no consolidation record written.
    registry = _RegistryApp(app_name="app", upstream_cls="Server")
    fleet = _starving_fleet()
    for _ in range(DRAIN_HYSTERESIS_POLLS + 1):
        asyncio.run(registry._update_drain(fleet))
    assert not any(c.draining for c in fleet)
    assert asyncio.run(registry.consolidation.get.aio("victim")) is None


def test_drain_lifecycle() -> None:
    # drain-victim policy gates: drain bar, transitioning blocks, survivor load.
    holders = lambda a, b: [  # noqa: E731
        _member("ta-0", leases={"1": a}),
        _member("ta-1", leases={"1": b}),
    ]
    assert select_drain_victim(holders(3, 4), 10) == (1, "ta-0"), (
        "k=2 concurrency=10 drain bar is 7 leases"
    )
    assert select_drain_victim(holders(3, 5), 10) is None
    for state in ("STAGING", "COMMITTING", "FETCHING"):
        blocked = holders(1, 1) + [_member("ta-2", applied=1, sync_state=state)]
        assert select_drain_victim(blocked, 10) is None, state
    allowed = holders(1, 1) + [_member("ta-2", applied=1, sync_state="HOLDING")]
    assert select_drain_victim(allowed, 10) == (1, "ta-0"), (
        "HOLDING must not block draining"
    )
    overloaded = [_member("ta-0", load=0, leases={"1": 1}), _member("ta-1", load=10, leases={"1": 1})]
    assert select_drain_victim(overloaded, 10) is None, "survivor at overload bar"
    stale = [_member("ta-0", load=0, leases={"1": 1}), _member("ta-1", load=0, leases={"1": 1})]
    stale[1].load_stale = True
    assert select_drain_victim(stale, 10) is None, "stale survivor load blocks drain"

    registry, consolidation = _drain_registry()
    fleet = _starving_fleet()
    for i in range(DRAIN_HYSTERESIS_POLLS - 1):
        asyncio.run(registry._update_drain(fleet))
        assert not any(c.draining for c in fleet), f"marked early at poll {i + 1}"
    asyncio.run(registry._update_drain(fleet))
    assert [c.task_id for c in fleet if c.draining] == ["ta-0"]
    assert asyncio.run(consolidation.get.aio("victim")) == {
        "victim_task_id": "ta-0",
        "version": 1,
    }

    first, consolidation = _drain_registry()
    fleet = _starving_fleet()
    for _ in range(DRAIN_HYSTERESIS_POLLS):
        asyncio.run(first._update_drain(fleet))
    restarted, _ = _drain_registry(consolidation)
    fleet = _starving_fleet()
    asyncio.run(restarted._update_drain(fleet))
    assert [c.task_id for c in fleet if c.draining] == ["ta-0"]

    registry, _ = _drain_registry(victim={"victim_task_id": "ta-c", "version": 2})
    fleet = [
        _member("ta-a", applied=1, leases={"1": 1}, target=3),
        _member("ta-b", applied=1, leases={"1": 1}, target=3),
        _member("ta-c", applied=2, leases={"2": 1}, target=3),
        _member("ta-d", applied=2, leases={"2": 1}, target=3),
    ]
    for _ in range(DRAIN_HYSTERESIS_POLLS + 1):
        asyncio.run(registry._update_drain(fleet))
    assert [c.task_id for c in fleet if c.draining] == ["ta-c"]

    def _advanced(fleet: list[ContainerInfo]) -> list[ContainerInfo]:
        fleet[0].applied_version = 2
        return fleet

    def _no_target(fleet: list[ContainerInfo]) -> list[ContainerInfo]:
        for c in fleet:
            c.target_version = None
        return fleet

    cleared = {"victim_task_id": None, "version": None}
    for record, mutate in [
        ({"victim_task_id": "ta-0", "version": 1}, _advanced),
        ({"victim_task_id": "ta-0", "version": 1}, _no_target),
        ({"victim_task_id": "ta-dead", "version": 1}, lambda fleet: fleet),
    ]:
        registry, consolidation = _drain_registry(victim=record)
        fleet = mutate(_starving_fleet())
        for _ in range(DRAIN_CLEAR_HYSTERESIS_POLLS):
            asyncio.run(registry._update_drain(fleet))
        assert not any(c.draining for c in fleet)
        assert asyncio.run(consolidation.get.aio("victim")) == cleared

    registry, consolidation = _drain_registry(
        victim={"victim_task_id": "ta-0", "version": 1}
    )
    fleet = _starving_fleet()
    for _ in range(DRAIN_CLEAR_HYSTERESIS_POLLS - 1):
        asyncio.run(registry._update_drain(fleet[1:]))
        got = asyncio.run(consolidation.get.aio("victim"))
        assert got["victim_task_id"] == "ta-0"
    asyncio.run(registry._update_drain(fleet))
    assert [c.task_id for c in fleet if c.draining] == ["ta-0"]
    for _ in range(DRAIN_CLEAR_HYSTERESIS_POLLS - 1):
        asyncio.run(registry._update_drain(fleet[1:]))
    assert (asyncio.run(consolidation.get.aio("victim")))["victim_task_id"] == "ta-0"
    asyncio.run(registry._update_drain(fleet[1:]))
    assert (asyncio.run(consolidation.get.aio("victim")))["victim_task_id"] is None
