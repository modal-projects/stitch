import os
from types import SimpleNamespace

import modal
import pytest

import cookbook.common.launch as common_launch
from cookbook.inference_only import launch


def _mock_remote_claim(monkeypatch):
    """Mock the Modal function-call boundary; return the recorded invocations."""
    calls = []
    claim_fn = SimpleNamespace(remote=lambda: calls.append("remote"))
    monkeypatch.setattr(
        modal.Function,
        "from_name",
        lambda app_name, name: calls.append((app_name, name)) or claim_fn,
    )
    return calls


def _mock_deploy_pool(monkeypatch):
    """Capture deploy_pool calls, invoking after_deploy like the real one does."""
    deployed = []

    def deploy_pool(run, after_deploy=None):
        deployed.append(run)
        if after_deploy is not None:
            after_deploy()

    monkeypatch.setattr(common_launch, "deploy_pool", deploy_pool)
    return deployed


def test_main_mints_run_id_and_deploys_pool(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENT_CONFIG", "glm5_2_fp8")
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setattr(
        launch.uuid, "uuid4", lambda: SimpleNamespace(hex="abcd1234rest")
    )
    run = SimpleNamespace(APP_NAME="stitch-glm5-2-fp8-abcd1234")
    monkeypatch.setattr(launch, "_load_run", lambda: run)
    deployed = _mock_deploy_pool(monkeypatch)
    _mock_remote_claim(monkeypatch)

    launch.main()

    assert os.environ["RUN_ID"] == "abcd1234"
    assert deployed == [run]


def test_main_honors_explicit_run_id(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENT_CONFIG", "glm5_2_fp8")
    monkeypatch.setenv("RUN_ID", "hero-run")
    run = SimpleNamespace(APP_NAME="stitch-glm5-2-fp8-hero-run")
    monkeypatch.setattr(launch, "_load_run", lambda: run)
    deployed = _mock_deploy_pool(monkeypatch)
    _mock_remote_claim(monkeypatch)

    launch.main()

    assert os.environ["RUN_ID"] == "hero-run"
    assert deployed == [run]


def test_main_requires_experiment_config(monkeypatch) -> None:
    monkeypatch.delenv("EXPERIMENT_CONFIG", raising=False)
    with pytest.raises(SystemExit, match="EXPERIMENT_CONFIG"):
        launch.main()


def test_main_invokes_remote_claim_function(monkeypatch) -> None:
    """Launch claims the boot pointer by invoking the deployed app's one-shot
    function remotely — never by constructing a store in the local process."""
    monkeypatch.setenv("EXPERIMENT_CONFIG", "glm5_2_fp8")
    monkeypatch.setenv("RUN_ID", "hero-run")
    run = SimpleNamespace(APP_NAME="stitch-glm5-2-fp8-hero-run")
    monkeypatch.setattr(launch, "_load_run", lambda: run)
    _mock_deploy_pool(monkeypatch)
    calls = _mock_remote_claim(monkeypatch)

    launch.main()

    assert calls == [(run.APP_NAME, launch.CLAIM_FUNCTION_NAME), "remote"]


def test_deploy_pool_claims_after_deploy_before_await(monkeypatch) -> None:
    """The remote claim lands between app deploy and the readiness wait, inside
    the replicas' pointer-wait window."""
    import stitch.service as service

    events = []
    run = SimpleNamespace(
        APP_NAME="stitch-glm5-2-fp8-hero-run",
        app=SimpleNamespace(deploy=lambda: events.append("deploy")),
        modal_cfg=SimpleNamespace(rollout_min_containers=1),
    )
    monkeypatch.setattr(
        service, "await_pool_ready", lambda *a, **k: events.append("await")
    )

    common_launch.deploy_pool(run, after_deploy=lambda: events.append("claim"))

    assert events == ["deploy", "claim", "await"]
