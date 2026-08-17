from __future__ import annotations

from types import SimpleNamespace

import pytest

from cookbook.miles_disagg.resume import (
    ResumePoint,
    export_version,
    resolve_resume_point,
    restore_resume_point,
    validate_auto_resume_config,
    validate_resume_config,
    wait_for_restored_pointer,
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
    # A crash between save and publish leaves the export without its published
    # version; the resume point falls back one save interval.
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
    # A fresh actor also reports iteration 0, so iteration 0 never resumes.
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
    point = ResumePoint(7, 6, "run", "/trainer", "/rollout")

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


def test_restore_resume_point_rejects_a_checkpoint_ahead_of_latest() -> None:
    volume = _Volume({"run/latest": b"run/weight_v000007"})
    point = ResumePoint(8, 7, "run", "/trainer", "/rollout")

    with pytest.raises(ValueError, match="newer than latest"):
        restore_resume_point(volume, point)


def test_engine_startup_waits_for_exact_restored_pointer() -> None:
    volume = _Volume({"run/latest": b"run/weight_v000008"})
    point = ResumePoint(8, 7, "run", "/trainer", "/rollout")

    restored = wait_for_restored_pointer(volume, point, timeout=1)

    assert restored.identity == "run/weight_v000008"


def test_engine_startup_does_not_accept_abandoned_suffix(monkeypatch) -> None:
    volume = _Volume({"run/latest": b"run/weight_v000012"})
    point = ResumePoint(8, 7, "run", "/trainer", "/rollout")
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
