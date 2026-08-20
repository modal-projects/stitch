from types import SimpleNamespace

from cookbook.common import launch
from stitch import service


def test_deploy_pool_waits_for_readiness(monkeypatch) -> None:
    events = []
    run = SimpleNamespace(
        APP_NAME="app-run",
        app=SimpleNamespace(deploy=lambda: events.append("deploy")),
        modal_cfg=SimpleNamespace(rollout_min_containers=32),
    )

    def wait(pool, **kwargs):
        events.append(("ready", pool.app_name, kwargs))

    monkeypatch.setattr(service, "await_pool_ready", wait)

    launch.deploy_pool(run)
    assert events == [
        "deploy",
        ("ready", "app-run", {"replica_floor": 32}),
    ]


def test_deploy_pool_waits_for_readiness_before_spawning(monkeypatch) -> None:
    events = []
    run = SimpleNamespace(
        APP_NAME="app-run",
        app=SimpleNamespace(deploy=lambda: events.append("deploy")),
        modal_cfg=SimpleNamespace(rollout_min_containers=32),
        spawn_train=lambda: events.append("spawn") or "call",
    )

    def wait(pool, **kwargs):
        events.append(("ready", pool.app_name, kwargs))

    monkeypatch.setattr(service, "await_pool_ready", wait)

    assert launch.deploy_pool_and_spawn(run) == "call"
    assert events == [
        "deploy",
        ("ready", "app-run", {"replica_floor": 32}),
        "spawn",
    ]
