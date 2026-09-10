from types import SimpleNamespace

from cookbook.common import launch
from stitch import service


def test_pool_reachable_with_only_native_server(monkeypatch) -> None:
    import modal
    from modal.exception import NotFoundError

    def from_name(app_name, class_name):
        assert app_name == "app-run"
        if class_name != "Server":
            raise NotFoundError(class_name)
        return SimpleNamespace(get_url=lambda: "https://pool.modal.direct")

    monkeypatch.setattr(modal.Server, "from_name", from_name)

    assert launch.pool_reachable(SimpleNamespace(APP_NAME="app-run"))


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
