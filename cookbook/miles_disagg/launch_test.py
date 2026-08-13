from __future__ import annotations

from types import SimpleNamespace

import pytest

from cookbook.miles_disagg import launch
from cookbook.miles_disagg.resume import (
    RESUME_POINT_ENV,
    ResumePoint,
    ResumePointNotFound,
)


def _point(rollout_id: int = 9, run_id: str = "run") -> ResumePoint:
    return ResumePoint(
        rollout_id,
        rollout_id + 1,
        run_id,
        "/trainer",
        "/rollout",
    )


def test_auto_resume_reuses_one_run_and_deployment(monkeypatch) -> None:
    attempts = []
    point = _point(19)
    monkeypatch.setattr(launch, "validate_auto_resume_config", lambda _cfg: None)
    monkeypatch.setattr(launch.uuid, "uuid4", lambda: SimpleNamespace(hex="run-rest"))
    monkeypatch.setattr(launch, "_resume_point_for_run", lambda _exp, _run: point)

    def run_attempt(**kwargs):
        attempts.append((kwargs, launch.os.environ.get(RESUME_POINT_ENV)))
        if len(attempts) == 1:
            raise RuntimeError("trainer stopped")

    monkeypatch.setattr(launch, "_run_attempt", run_attempt)

    launch._run_auto_resume(SimpleNamespace(miles=object()), None)

    assert [attempt[0] for attempt in attempts] == [
        {"deploy": True, "supervise": True},
        {"deploy": False, "supervise": True},
    ]
    assert attempts[0][1] is None
    assert ResumePoint.from_json(attempts[1][1]) == point
    assert launch.os.environ["RUN_ID"] == "run-rest"


def test_auto_resume_existing_run_never_deploys(monkeypatch) -> None:
    point = _point()
    attempts = []
    monkeypatch.setattr(launch, "validate_auto_resume_config", lambda _cfg: None)
    monkeypatch.setattr(launch, "_resume_point_for_run", lambda _exp, _run: point)
    monkeypatch.setattr(
        launch,
        "_run_attempt",
        lambda **kwargs: attempts.append(kwargs),
    )

    launch._run_auto_resume(SimpleNamespace(miles=object()), "run")

    assert attempts == [{"deploy": False, "supervise": True}]
    assert launch.os.environ["RUN_ID"] == "run"


def test_set_attempt_env_uses_one_resume_payload() -> None:
    env = {"STITCH_UNUSED": "value"}
    point = _point()

    launch._set_attempt_env(env, run_id="run", resume_point=point)

    assert env == {
        "STITCH_UNUSED": "value",
        "RUN_ID": "run",
        RESUME_POINT_ENV: point.to_json(),
    }


def test_set_attempt_env_clears_a_stale_resume_point() -> None:
    env = {RESUME_POINT_ENV: "stale"}

    launch._set_attempt_env(env, run_id="fresh", resume_point=None)

    assert env == {"RUN_ID": "fresh"}


def test_manual_resume_uses_existing_run_and_pool(monkeypatch) -> None:
    exp = SimpleNamespace(miles=object())
    point = _point()
    configured = {}
    ran = []
    monkeypatch.setattr(
        launch,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                auto_resume=False,
                resume_from="run",
            )
        ),
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", "test")
    monkeypatch.setattr(launch.importlib, "import_module", lambda _name: exp)
    monkeypatch.setattr(launch, "validate_resume_config", lambda _cfg: None)
    monkeypatch.setattr(launch, "_resume_point_for_run", lambda _exp, _run: point)
    monkeypatch.setattr(
        launch,
        "_set_attempt_env",
        lambda _env, **kwargs: configured.update(kwargs),
    )
    monkeypatch.setattr(launch, "_run_attempt", lambda **kwargs: ran.append(kwargs))

    launch.main()

    assert configured == {"run_id": "run", "resume_point": point}
    assert ran == [{"deploy": False, "supervise": False}]


def test_auto_resume_reuses_checkpoint_when_attempt_has_no_new_save(
    monkeypatch,
) -> None:
    previous = _point(9, "run")

    def missing(_exp, _run_id):
        raise ResumePointNotFound("no checkpoint")

    monkeypatch.setattr(launch, "_resume_point_for_run", missing)

    assert launch._resume_point_after_failure(object(), "run", previous) == previous


def test_auto_resume_requires_a_checkpoint_before_first_retry(monkeypatch) -> None:
    def missing(_exp, _run_id):
        raise ResumePointNotFound("no checkpoint")

    monkeypatch.setattr(launch, "_resume_point_for_run", missing)

    with pytest.raises(ValueError, match="no checkpoint"):
        launch._resume_point_after_failure(object(), "run", None)


def test_resume_fails_before_loading_a_volume_with_s3(monkeypatch) -> None:
    monkeypatch.setenv("STITCH_STORE_BACKEND", "s3")
    monkeypatch.setenv("STITCH_S3_SECRET_NAME", "test-secret")

    with pytest.raises(ValueError, match="requires the Modal Volume store"):
        launch._resume_point_for_run(
            SimpleNamespace(EXPERIMENT_VOLUME_NAME="test-volume"),
            "run",
        )
