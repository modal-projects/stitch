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


def test_supervised_attempt_passes_resume_point_to_fresh_process(monkeypatch) -> None:
    captured = {}

    def run(command, *, env, check):
        captured.update(command=command, env=env, check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", run)
    point = ResumePoint(9, "old", "/trainer", "/rollout")

    launch._run_supervised_attempt(run_id="new", resume_point=point)

    assert captured["env"]["RUN_ID"] == "new"
    assert ResumePoint.from_json(captured["env"][RESUME_POINT_ENV]) == point
    assert captured["command"][-1] == "--run-attempt"
    assert captured["check"] is False


def test_auto_resume_uses_checkpoint_from_failed_attempt(monkeypatch) -> None:
    run_ids = iter((SimpleNamespace(hex="first"), SimpleNamespace(hex="second")))
    launches = []
    point = ResumePoint(19, "first", "/trainer", "/rollout")
    monkeypatch.setattr(launch.uuid, "uuid4", lambda: next(run_ids))
    monkeypatch.setattr(launch, "validate_auto_resume_config", lambda _cfg: None)
    monkeypatch.setattr(launch, "_resume_point_for_run", lambda _exp, run_id: point)

    def launch_attempt(**kwargs):
        launches.append(kwargs)
        return subprocess.CompletedProcess([], 1 if len(launches) == 1 else 0)

    monkeypatch.setattr(launch, "_run_supervised_attempt", launch_attempt)

    launch._run_auto_resume(SimpleNamespace(miles=object()), None)

    assert launches == [
        {"run_id": "first", "resume_point": None},
        {"run_id": "second", "resume_point": point},
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

    assert launches == [{"run_id": "new", "resume_point": point}]


def test_set_attempt_env_uses_one_resume_payload() -> None:
    env = {"STITCH_UNUSED": "value"}
    point = ResumePoint(9, "old", "/trainer", "/rollout")

    launch._set_attempt_env(env, run_id="new", resume_point=point)

    assert env == {
        "STITCH_UNUSED": "value",
        "RUN_ID": "new",
        RESUME_POINT_ENV: point.to_json(),
    }


def test_set_attempt_env_clears_a_stale_resume_point() -> None:
    env = {RESUME_POINT_ENV: "stale"}

    launch._set_attempt_env(env, run_id="fresh", resume_point=None)

    assert env == {"RUN_ID": "fresh"}


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

    assert configured == {"run_id": "newrunid", "resume_point": point}
    assert ran == [False]


def test_supervised_attempt_stops_pool_when_launch_fails(monkeypatch) -> None:
    run = SimpleNamespace(APP_NAME="app")
    stopped = {}
    monkeypatch.setattr(launch.importlib, "import_module", lambda _name: run)

    def fail(_run):
        raise RuntimeError("not ready")

    monkeypatch.setattr(common_launch, "deploy_pool_and_spawn", fail)

    def stop(command, *, check):
        stopped.update(command=command, check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", stop)

    with pytest.raises(RuntimeError, match="not ready"):
        launch._run_attempt(supervise=True)

    assert stopped["command"][-4:] == ["app", "stop", "--yes", "app"]
    assert stopped["check"] is False


def test_auto_resume_reuses_source_when_attempt_has_no_new_checkpoint(
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
