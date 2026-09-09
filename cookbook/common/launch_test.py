from types import SimpleNamespace

import pytest

from cookbook.common import launch
from cookbook.common.config import ModalConfig
from stitch import service
from stitch.types import VersionRef


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
        modal_cfg=ModalConfig(rollout_min_containers=32),
        spawn_train=lambda: events.append("spawn") or "call",
    )

    def wait(pool, **kwargs):
        events.append(("ready", pool.app_name, kwargs))

    monkeypatch.setattr(service, "await_pool_ready", wait)

    assert launch.deploy_pool_and_spawn(run) == "call"
    assert events == [
        "deploy",
        (
            "ready",
            "app-run",
            {"replica_floor": 32, "min_ready": None, "latest": None},
        ),
        "spawn",
    ]


def test_rollout_readiness_uses_config_threshold_and_claimed_version(monkeypatch):
    observed = {}
    latest = VersionRef("run", 3)
    config = ModalConfig(rollout_min_containers=32, rollout_min_ready=16)

    def wait(pool, **kwargs):
        observed.update(app_name=pool.app_name, **kwargs)

    monkeypatch.setattr(service, "await_pool_ready", wait)

    launch.await_rollout_ready("app-run", config, latest=latest)

    assert observed == {
        "app_name": "app-run",
        "replica_floor": 32,
        "min_ready": 16,
        "latest": latest,
    }
    assert config.rollout_min_containers == 32


def test_bypass_skips_wait_and_is_forwarded_to_trainer(monkeypatch):
    events = []
    run = SimpleNamespace(
        APP_NAME="app-run",
        app=SimpleNamespace(deploy=lambda: events.append("deploy")),
        modal_cfg=ModalConfig(rollout_min_containers=32, rollout_min_ready=16),
        spawn_train=lambda **kwargs: events.append(("spawn", kwargs)) or "call",
    )

    def wait(*args, **kwargs):
        pytest.fail("bypassed readiness must not probe the rollout pool")

    monkeypatch.setattr(service, "await_pool_ready", wait)

    assert launch.deploy_pool_and_spawn(run, skip_rollout_ready_check=True) == "call"
    assert events == ["deploy", ("spawn", {"skip_rollout_ready_check": True})]


def test_bypass_does_not_allow_resume_on_a_missing_pool(monkeypatch):
    monkeypatch.setattr(launch, "pool_reachable", lambda run: False)
    monkeypatch.setenv("RUN_ID", "run")
    run = SimpleNamespace(APP_NAME="app-run", __name__="cookbook.miles_disagg.app")

    with pytest.raises(SystemExit, match="No deployed pool"):
        launch.spawn_on_pool(run, skip_rollout_ready_check=True)
