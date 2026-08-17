from types import SimpleNamespace

import pytest

from cookbook.common import launch
from stitch import service


def _run(events: list) -> SimpleNamespace:
    return SimpleNamespace(
        __name__="cookbook.miles_disagg.app",
        APP_NAME="app-run",
        modal_cfg=SimpleNamespace(rollout_min_containers=32),
        spawn_train=lambda: events.append("spawn") or "call",
    )


def test_spawn_waits_for_the_deployed_pool_floor(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(launch, "_pool_reachable", lambda _pool: True)
    monkeypatch.setattr(
        service,
        "await_pool_ready",
        lambda pool, **kwargs: events.append(("ready", pool.app_name, kwargs)),
    )

    assert launch.spawn_on_pool(_run(events)) == "call"
    assert events == [
        ("ready", "app-run", {"replica_floor": 32}),
        "spawn",
    ]


def test_spawn_fails_fast_with_the_deploy_instruction(monkeypatch) -> None:
    events = []
    monkeypatch.setenv("RUN_ID", "myrun01")
    monkeypatch.setattr(launch, "_pool_reachable", lambda _pool: False)

    with pytest.raises(SystemExit, match="modal deploy -m cookbook.miles_disagg.app"):
        launch.spawn_on_pool(_run(events))
    assert events == []  # never deploys, never spawns
