from __future__ import annotations

import pytest

from cookbook.miles_disagg.resume import (
    ResumePoint,
    resolve_resume_point,
    validate_auto_resume_config,
    validate_resume_config,
)


class _Volume:
    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def read_file(self, path: str):
        try:
            yield self.files[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc


class _Config:
    save_interval = 20
    save_hf = "hf_checkpoints/weight_v{rollout_id:06d}"
    no_save_optim = False


def test_resolve_resume_point_pairs_megatron_and_hf_checkpoints() -> None:
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"119\n",
            "old/hf_checkpoints/weight_v000119/.complete": b"",
        }
    )

    assert resolve_resume_point(
        volume, source_run_id="old", save_hf=_Config.save_hf
    ) == ResumePoint(
        version=119,
        source_run_id="old",
        trainer_checkpoint="/stitch/old/checkpoints",
        rollout_checkpoint="/stitch/old/hf_checkpoints/weight_v000119",
    )


def test_resolve_resume_point_requires_matching_complete_hf_export() -> None:
    volume = _Volume({"old/checkpoints/latest_checkpointed_iteration.txt": b"119\n"})

    with pytest.raises(ValueError, match="no complete HF export"):
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


def test_resume_point_json_round_trip() -> None:
    point = ResumePoint(7, "run", "/trainer", "/rollout")

    assert ResumePoint.from_json(point.to_json()) == point
