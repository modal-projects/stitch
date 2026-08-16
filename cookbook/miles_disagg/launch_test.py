from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from cookbook.common import launch as common_launch
from cookbook.miles_disagg import launch
from cookbook.miles_disagg.resume import (
    RESUME_POINT_ENV,
    ResumePoint,
    ResumePointNotFound,
)


def test_supervised_attempt_preserves_run_id(monkeypatch) -> None:
    captured = {}

    def run(command, *, env, check):
        captured.update(command=command, env=env, check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", run)
    point = ResumePoint(9, "old", "/trainer", "/rollout")

    launch._run_supervised_attempt(run_id="old", resume_point=point)

    assert captured["env"]["RUN_ID"] == "old"
    assert ResumePoint.from_json(captured["env"][RESUME_POINT_ENV]) == point
    assert captured["command"][-1] == "--run-attempt"
    assert captured["check"] is False


def test_auto_resume_uses_checkpoint_from_failed_attempt(monkeypatch) -> None:
    launches = []
    point = ResumePoint(19, "first-re", "/trainer", "/rollout")
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setattr(launch.uuid, "uuid4", lambda: SimpleNamespace(hex="first-rest"))
    monkeypatch.setattr(launch, "validate_auto_resume_config", lambda _cfg: None)
    monkeypatch.setattr(launch, "_resume_point_for_run", lambda _exp, run_id: point)

    def launch_attempt(**kwargs):
        launches.append(kwargs)
        return subprocess.CompletedProcess([], 1 if len(launches) == 1 else 0)

    monkeypatch.setattr(launch, "_run_supervised_attempt", launch_attempt)

    launch._run_auto_resume(SimpleNamespace(miles=object()), None)

    assert launches == [
        {"run_id": "first-re", "resume_point": None},
        {"run_id": "first-re", "resume_point": point},
    ]


def test_auto_resume_starts_from_requested_checkpoint(monkeypatch) -> None:
    point = ResumePoint(9, "old", "/trainer", "/rollout")
    launches = []
    monkeypatch.setattr(launch.uuid, "uuid4", lambda: SimpleNamespace(hex="new"))
    monkeypatch.setattr(launch, "validate_auto_resume_config", lambda _cfg: None)
    monkeypatch.setattr(launch, "_resume_point_for_run", lambda _exp, _run_id: point)

    def launch_attempt(**kwargs):
        launches.append(kwargs)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(launch, "_run_supervised_attempt", launch_attempt)

    launch._run_auto_resume(SimpleNamespace(miles=object()), "old")

    assert launches == [{"run_id": "old", "resume_point": point}]


def test_auto_resume_honors_explicit_run_id(monkeypatch) -> None:
    launches = []
    monkeypatch.setenv("RUN_ID", "hero-run")
    monkeypatch.setattr(launch, "validate_auto_resume_config", lambda _cfg: None)
    monkeypatch.setattr(
        launch,
        "_run_supervised_attempt",
        lambda **kwargs: launches.append(kwargs) or subprocess.CompletedProcess([], 0),
    )

    launch._run_auto_resume(SimpleNamespace(miles=object()), None)

    assert launches == [{"run_id": "hero-run", "resume_point": None}]


def test_set_attempt_env_uses_one_resume_payload() -> None:
    env = {"STITCH_UNUSED": "value"}
    point = ResumePoint(9, "old", "/trainer", "/rollout")

    launch._set_attempt_env(env, run_id="old", resume_point=point)

    assert env == {
        "STITCH_UNUSED": "value",
        "RUN_ID": "old",
        RESUME_POINT_ENV: point.to_json(),
    }


def test_set_attempt_env_clears_a_stale_resume_point() -> None:
    env = {RESUME_POINT_ENV: "stale"}

    launch._set_attempt_env(env, run_id="fresh", resume_point=None)

    assert env == {"RUN_ID": "fresh"}


def test_set_attempt_env_rejects_cross_run_resume() -> None:
    point = ResumePoint(9, "old", "/trainer", "/rollout")

    with pytest.raises(ValueError, match="belongs to run"):
        launch._set_attempt_env({}, run_id="new", resume_point=point)


def test_manual_launch_runs_directly_without_attempt_subprocess(monkeypatch) -> None:
    exp = SimpleNamespace(miles=object())
    point = ResumePoint(9, "old", "/trainer", "/rollout")
    configured = {}
    ran = []
    monkeypatch.setattr(
        launch,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                run_attempt=False,
                auto_resume=False,
                resume_from="old",
            )
        ),
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", "test")
    monkeypatch.setattr(launch.importlib, "import_module", lambda _name: exp)
    monkeypatch.setattr(launch, "validate_resume_config", lambda _cfg: None)
    monkeypatch.setattr(launch, "_resume_point_for_run", lambda _exp, _run_id: point)
    monkeypatch.setattr(
        launch,
        "_set_attempt_env",
        lambda _env, **kwargs: configured.update(kwargs),
    )
    monkeypatch.setattr(
        launch, "_run_attempt", lambda *, supervise: ran.append(supervise)
    )
    monkeypatch.setattr(
        launch.uuid, "uuid4", lambda: SimpleNamespace(hex="newrunid-rest")
    )

    launch.main()

    assert configured == {"run_id": "old", "resume_point": point}
    assert ran == [False]


def test_fresh_manual_launch_honors_explicit_run_id(monkeypatch) -> None:
    configured = {}
    monkeypatch.setattr(
        launch,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                run_attempt=False,
                auto_resume=False,
                resume_from=None,
            )
        ),
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", "test")
    monkeypatch.setenv("RUN_ID", "hero-run")
    monkeypatch.setattr(
        launch.importlib,
        "import_module",
        lambda _name: SimpleNamespace(miles=object()),
    )
    monkeypatch.setattr(
        launch,
        "_set_attempt_env",
        lambda _env, **kwargs: configured.update(kwargs),
    )
    monkeypatch.setattr(launch, "_run_attempt", lambda *, supervise: None)

    launch.main()

    assert configured == {"run_id": "hero-run", "resume_point": None}


def test_supervised_attempt_leaves_pool_deployed_when_trainer_fails(
    monkeypatch,
) -> None:
    run = SimpleNamespace(APP_NAME="app")
    monkeypatch.setattr(launch.importlib, "import_module", lambda _name: run)
    monkeypatch.delenv(RESUME_POINT_ENV, raising=False)

    def fail(_run):
        raise RuntimeError("not ready")

    monkeypatch.setattr(common_launch, "deploy_pool_and_spawn", fail)

    with pytest.raises(RuntimeError, match="not ready"):
        launch._run_attempt(supervise=True)


def test_resume_restores_pointer_before_pool_readiness(monkeypatch) -> None:
    events = []
    point = ResumePoint(9, "run", "/trainer", "/rollout")
    volume = object()
    run = SimpleNamespace(
        APP_NAME="app-run",
        modal_cfg=SimpleNamespace(rollout_min_containers=2),
        exp=SimpleNamespace(EXPERIMENT_VOLUME_NAME="runs"),
        app=SimpleNamespace(
            deploy=lambda *, strategy: events.append(("deploy", strategy))
        ),
        spawn_train=lambda: events.append(("spawn",)) or object(),
    )
    monkeypatch.setenv("RUN_ID", "run")
    monkeypatch.setenv(RESUME_POINT_ENV, point.to_json())
    monkeypatch.setattr(launch.importlib, "import_module", lambda _name: run)
    monkeypatch.setattr(
        launch.modal.Volume, "from_name", lambda *_args, **_kwargs: volume
    )
    monkeypatch.setattr(
        launch,
        "restore_resume_point",
        lambda actual_volume, actual_point: (
            events.append(("restore", actual_volume, actual_point))
            or SimpleNamespace(identity="run/weight_v000009")
        ),
    )

    import stitch.service

    monkeypatch.setattr(
        stitch.service,
        "await_pool_ready",
        lambda _pool, **_kwargs: events.append(("ready",)),
    )

    launch._run_attempt(supervise=False)

    assert events == [
        ("deploy", "recreate"),
        ("restore", volume, point),
        ("ready",),
        ("spawn",),
    ]


def test_auto_resume_reuses_previous_checkpoint_when_attempt_has_no_new_checkpoint(
    monkeypatch,
) -> None:
    previous = ResumePoint(9, "old", "/trainer", "/rollout")

    def missing(_exp, _run_id):
        raise ResumePointNotFound("no checkpoint")

    monkeypatch.setattr(launch, "_resume_point_for_run", missing)

    assert launch._resume_point_after_failure(object(), "failed", previous) == previous


def test_auto_resume_requires_a_checkpoint_before_first_retry(monkeypatch) -> None:
    def missing(_exp, _run_id):
        raise ResumePointNotFound("no checkpoint")

    monkeypatch.setattr(launch, "_resume_point_for_run", missing)

    with pytest.raises(ValueError, match="no checkpoint"):
        launch._resume_point_after_failure(object(), "failed", None)
