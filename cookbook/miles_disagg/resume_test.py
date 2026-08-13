from __future__ import annotations

from types import SimpleNamespace

import pytest

from cookbook.miles_disagg.resume import (
    ResumePoint,
    resolve_resume_point,
    restore_resume_point,
    saved_checkpoint_version,
    validate_auto_resume_config,
    validate_resume_config,
    wait_for_restored_pointer,
)


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


@pytest.mark.parametrize(
    ("rollout_id", "resumed", "version"),
    [
        (19, False, 20),
        (120, True, 120),
    ],
)
def test_saved_checkpoint_version(rollout_id: int, resumed: bool, version: int) -> None:
    assert saved_checkpoint_version(rollout_id, resumed=resumed) == version


def test_resolve_resume_point_pairs_megatron_and_hf_checkpoints() -> None:
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"119\n",
            "old/checkpoints/iter_0000119/state": b"checkpoint",
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


def test_resolve_resume_point_falls_back_to_previous_complete_pair() -> None:
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"119\n",
            "old/checkpoints/iter_0000099/state": b"checkpoint",
            "old/checkpoints/iter_0000119/state": b"checkpoint",
            "old/hf_checkpoints/weight_v000099/.complete": b"",
        }
    )

    assert resolve_resume_point(
        volume, source_run_id="old", save_hf=_Config.save_hf
    ) == ResumePoint(
        version=99,
        source_run_id="old",
        trainer_checkpoint="/stitch/old/checkpoints",
        rollout_checkpoint="/stitch/old/hf_checkpoints/weight_v000099",
    )


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
    point = ResumePoint(7, "run", "/trainer", "/rollout")

    assert ResumePoint.from_json(point.to_json()) == point


def test_restore_resume_point_overwrites_trackers_but_preserves_updates() -> None:
    volume = _Volume(
        {
            "run/latest": b"run/weight_v000012",
            "run/checkpoints/latest_checkpointed_iteration.txt": b"12",
            "run/updates/weight_v000009/old": b"preserved",
        }
    )
    point = ResumePoint(8, "run", "/trainer", "/rollout")

    restored = restore_resume_point(volume, point)

    assert restored.identity == "run/weight_v000008"
    assert volume.files["run/latest"] == b"run/weight_v000008"
    assert volume.files["run/checkpoints/latest_checkpointed_iteration.txt"] == b"8"
    assert volume.files["run/updates/weight_v000009/old"] == b"preserved"


def test_restore_resume_point_rejects_a_checkpoint_ahead_of_latest() -> None:
    volume = _Volume({"run/latest": b"run/weight_v000007"})
    point = ResumePoint(8, "run", "/trainer", "/rollout")

    with pytest.raises(ValueError, match="newer than latest"):
        restore_resume_point(volume, point)


def test_engine_startup_waits_for_exact_restored_pointer() -> None:
    volume = _Volume({"run/latest": b"run/weight_v000008"})
    point = ResumePoint(8, "run", "/trainer", "/rollout")

    restored = wait_for_restored_pointer(volume, point, timeout=1)

    assert restored.identity == "run/weight_v000008"


def test_engine_startup_does_not_accept_abandoned_suffix(monkeypatch) -> None:
    volume = _Volume({"run/latest": b"run/weight_v000012"})
    point = ResumePoint(8, "run", "/trainer", "/rollout")
    reloads = 0

    def reload() -> None:
        nonlocal reloads
        reloads += 1
        if reloads == 2:
            volume.files["run/latest"] = b"run/weight_v000008"

    volume.reload = reload
    monkeypatch.setattr("cookbook.miles_disagg.resume.time.sleep", lambda _delay: None)

    restored = wait_for_restored_pointer(volume, point, timeout=1)

    assert restored.identity == "run/weight_v000008"
    assert reloads == 2
