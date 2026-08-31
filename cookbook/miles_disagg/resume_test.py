from __future__ import annotations

from types import SimpleNamespace

import pytest

from cookbook.miles_disagg.resume import (
    ResumePoint,
    export_version,
    prepare_attempt,
    read_trainer_call,
    record_trainer_call,
    resolve_resume_point,
    restore_boot_pointer,
    restore_resume_point,
    validate_auto_resume_config,
    validate_resume_config,
)

_INDEX = "model.safetensors.index.json"


def _published(version: int) -> dict[str, bytes]:
    name = f"old/updates/weight_v{version:06d}/{_INDEX}"
    return {name: b'{"metadata": {"version": "%06d"}}' % version}


class _Volume:
    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def read_file(self, path: str):
        try:
            yield self.files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc

    def reload(self) -> None:
        pass

    def iterdir(self, path: str, *, recursive: bool):
        assert recursive is False
        prefix = path.rstrip("/") + "/"
        entries = {
            name.split("/", 1)[0]
            for key in self.files
            if key.startswith(prefix)
            for name in [key.removeprefix(prefix)]
        }
        return [SimpleNamespace(path=f"{path}/{name}") for name in sorted(entries)]

    def batch_upload(self, *, force: bool):
        volume = self

        class _Upload:
            def __enter__(self):
                assert force is True
                return self

            def put_file(self, source, path: str) -> None:
                volume.files[path] = source.read()

            def __exit__(self, *_args) -> None:
                return None

        return _Upload()


class _Config:
    save_interval = 20
    save_hf = "hf_checkpoints/weight_v{rollout_id:06d}"
    no_save_optim = False


@pytest.mark.parametrize(("iteration", "version"), [(0, 1), (19, 20), (119, 120)])
def test_export_version(iteration: int, version: int) -> None:
    assert export_version(iteration) == version


def test_resolve_resume_point_pairs_megatron_and_hf_checkpoints() -> None:
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"119\n",
            "old/checkpoints/iter_0000119/state": b"checkpoint",
            "old/hf_checkpoints/weight_v000119/.complete": b"",
            **_published(120),
        }
    )

    assert resolve_resume_point(
        volume, source_run_id="old", save_hf=_Config.save_hf
    ) == ResumePoint(
        version=120,
        iteration=119,
        source_run_id="old",
        trainer_checkpoint="/stitch/old/checkpoints",
        rollout_checkpoint="/stitch/old/hf_checkpoints/weight_v000119",
    )


def test_resolve_resume_point_falls_back_to_previous_complete_pair() -> None:
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"119\n",
            "old/checkpoints/iter_0000099/state": b"checkpoint",
            "old/checkpoints/iter_0000119/state": b"checkpoint",
            "old/hf_checkpoints/weight_v000099/.complete": b"",
            **_published(100),
        }
    )

    assert resolve_resume_point(
        volume, source_run_id="old", save_hf=_Config.save_hf
    ) == ResumePoint(
        version=100,
        iteration=99,
        source_run_id="old",
        trainer_checkpoint="/stitch/old/checkpoints",
        rollout_checkpoint="/stitch/old/hf_checkpoints/weight_v000099",
    )


def test_resolve_resume_point_requires_the_exports_publication() -> None:
    # No publication for iteration 119: the resume point falls back to 99.
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"119\n",
            "old/checkpoints/iter_0000099/state": b"checkpoint",
            "old/checkpoints/iter_0000119/state": b"checkpoint",
            "old/hf_checkpoints/weight_v000099/.complete": b"",
            "old/hf_checkpoints/weight_v000119/.complete": b"",
            **_published(100),
        }
    )

    resolved = resolve_resume_point(
        volume, source_run_id="old", save_hf=_Config.save_hf
    )

    assert (resolved.version, resolved.iteration) == (100, 99)


def test_resolve_resume_point_rejects_a_mislabeled_publication() -> None:
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"119\n",
            "old/checkpoints/iter_0000119/state": b"checkpoint",
            "old/hf_checkpoints/weight_v000119/.complete": b"",
            f"old/updates/weight_v000120/{_INDEX}": b'{"metadata": {"version": "0007"}}',
        }
    )

    with pytest.raises(ValueError, match="identifies v7, not v120"):
        resolve_resume_point(volume, source_run_id="old", save_hf=_Config.save_hf)


def test_resolve_resume_point_skips_iteration_zero() -> None:
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"0\n",
            "old/checkpoints/iter_0000000/state": b"checkpoint",
            "old/hf_checkpoints/weight_v000000/.complete": b"",
            **_published(1),
        }
    )

    with pytest.raises(ValueError, match="no complete Megatron/HF checkpoint pair"):
        resolve_resume_point(volume, source_run_id="old", save_hf=_Config.save_hf)


def test_resolve_resume_point_requires_a_complete_checkpoint_pair() -> None:
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"119\n",
            "old/checkpoints/iter_0000119/state": b"checkpoint",
        }
    )

    with pytest.raises(ValueError, match="no complete Megatron/HF checkpoint pair"):
        resolve_resume_point(volume, source_run_id="old", save_hf=_Config.save_hf)


def test_resolve_resume_point_rejects_path_like_run_id() -> None:
    with pytest.raises(ValueError, match="invalid resume run id"):
        resolve_resume_point(
            _Volume({}), source_run_id="../old", save_hf=_Config.save_hf
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("save_interval", None, "positive save_interval"),
        ("save_hf", None, "requires save_hf"),
        ("save_hf", "hf_checkpoints/latest", "rollout_id"),
        ("no_save_optim", True, "optimizer checkpointing"),
        ("no_save_rng", True, "RNG checkpointing"),
    ],
)
def test_validate_auto_resume_config(field: str, value: object, message: str) -> None:
    cfg = _Config()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=message):
        validate_auto_resume_config(cfg)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("no_load_optim", "loading optimizer state"),
        ("no_load_rng", "loading RNG state"),
    ],
)
def test_validate_resume_config(field: str, message: str) -> None:
    cfg = _Config()
    setattr(cfg, field, True)

    with pytest.raises(ValueError, match=message):
        validate_resume_config(cfg)


def test_validate_resume_requires_per_step_weight_updates() -> None:
    cfg = _Config()
    cfg.update_weights_interval = 2

    with pytest.raises(ValueError, match="update_weights_interval == 1"):
        validate_resume_config(cfg)


def test_resume_point_json_round_trip() -> None:
    point = ResumePoint(8, 7, "run", "/trainer", "/rollout")

    assert ResumePoint.from_json(point.to_json()) == point


def test_restore_resume_point_overwrites_trackers_but_preserves_updates() -> None:
    volume = _Volume(
        {
            "run/latest": b"run/weight_v000012",
            "run/checkpoints/latest_checkpointed_iteration.txt": b"12",
            "run/updates/weight_v000009/old": b"preserved",
        }
    )
    point = ResumePoint(8, 7, "run", "/trainer", "/rollout")

    restored = restore_resume_point(volume, point)

    assert restored.identity == "run/weight_v000008"
    assert volume.files["run/latest"] == b"run/weight_v000008"
    assert volume.files["run/checkpoints/latest_checkpointed_iteration.txt"] == b"7"
    assert volume.files["run/updates/weight_v000009/old"] == b"preserved"


def test_restore_resume_point_completes_an_interrupted_publish() -> None:
    # latest one behind the resume point: an interrupted publish, completed here.
    volume = _Volume({"run/latest": b"run/weight_v000007"})
    point = ResumePoint(8, 7, "run", "/trainer", "/rollout")

    assert restore_resume_point(volume, point).identity == "run/weight_v000008"
    assert volume.files["run/latest"] == b"run/weight_v000008"


def test_restore_resume_point_rejects_a_checkpoint_ahead_of_latest() -> None:
    volume = _Volume({"run/latest": b"run/weight_v000006"})
    point = ResumePoint(8, 7, "run", "/trainer", "/rollout")

    with pytest.raises(ValueError, match="newer than latest"):
        restore_resume_point(volume, point)


def test_prepare_attempt_restores_the_newest_pair() -> None:
    volume = _Volume(
        {
            "old/latest": b"old/weight_v000012",
            "old/checkpoints/latest_checkpointed_iteration.txt": b"7",
            "old/checkpoints/iter_0000007/state": b"checkpoint",
            "old/hf_checkpoints/weight_v000007/.complete": b"",
            **_published(8),
        }
    )

    point = prepare_attempt(volume, run_id="old", save_hf=_Config.save_hf)

    assert point is not None and (point.version, point.iteration) == (8, 7)
    assert volume.files["old/latest"] == b"old/weight_v000008"
    assert volume.files["old/checkpoints/latest_checkpointed_iteration.txt"] == b"7"


def test_prepare_attempt_restarts_from_scratch_before_the_first_pair() -> None:
    volume = _Volume({"old/latest": b"old/weight_v000003"})

    assert prepare_attempt(volume, run_id="old", save_hf=_Config.save_hf) is None
    assert volume.files["old/latest"] == b"old/weight_v000000"


def test_prepare_attempt_is_a_noop_on_a_fresh_run() -> None:
    volume = _Volume({})

    assert prepare_attempt(volume, run_id="old", save_hf=_Config.save_hf) is None
    assert volume.files == {}


def test_restore_boot_pointer_rejects_a_foreign_run() -> None:
    volume = _Volume({"old/latest": b"other/weight_v000003"})

    with pytest.raises(ValueError, match="belongs to run"):
        restore_boot_pointer(volume, "old")


def test_trainer_call_record_round_trip() -> None:
    volume = _Volume({})

    assert read_trainer_call(volume, "old") is None
    record_trainer_call(volume, "old", "fc-123")
    assert read_trainer_call(volume, "old") == "fc-123"
    record_trainer_call(volume, "old", "fc-456")  # a newer spawn supersedes
    assert read_trainer_call(volume, "old") == "fc-456"
