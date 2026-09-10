from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cookbook.miles_disagg.resume import (
    ResumePoint,
    export_version,
    newest_complete_export,
    newest_persisted_export,
    prepare_attempt,
    read_trainer_call,
    record_trainer_call,
    resolve_resume_point,
    restore_boot_pointer,
    restore_resume_point,
    validate_resumable_config,
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
def test_validate_resumable_config(field: str, value: object, message: str) -> None:
    cfg = _Config()
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=message):
        validate_resumable_config(cfg)


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


def test_newest_complete_export_picks_the_newest_eligible(tmp_path) -> None:
    for iteration in (7, 19, 39):
        export = tmp_path / _Config.save_hf.format(rollout_id=iteration)
        export.mkdir(parents=True)
        (export / ".complete").touch()
    (tmp_path / "hf_checkpoints/weight_v000059").mkdir()  # saved, not complete
    (tmp_path / "hf_checkpoints/scratch").mkdir()  # not an export at all

    # v40 (iteration 39) is published ahead of latest, so v20 is the boot point.
    assert newest_complete_export(
        tmp_path, save_hf=_Config.save_hf, latest_version=25
    ) == (20, tmp_path / "hf_checkpoints/weight_v000019")


def test_newest_complete_export_is_none_before_the_first_save(tmp_path) -> None:
    assert (
        newest_complete_export(tmp_path, save_hf=_Config.save_hf, latest_version=9)
        is None
    )


def _staged_manifest(iteration=19, attempt="attempt"):
    root = f"old/attempts/{attempt}/snapshots/{iteration:07d}"
    return {
        "schema_version": 1,
        "run_id": "old",
        "attempt_id": attempt,
        "iteration": iteration,
        "version": iteration + 1,
        "completed_at_ns": iteration,
        "checkpoint_root": f"{root}/checkpoints",
        "hf_directory": f"{root}/hf_checkpoints/weight_v{iteration:06d}",
    }


def test_staged_resume_ignores_partial_uploads_and_requires_matching_publication():
    checkpoint_volume = _Volume(
        {
            "old/attempts/partial/snapshots/0000039/hf_checkpoints/weight_v000039/.complete": b"",
            "old/completed/0000019-attempt.json": json.dumps(
                _staged_manifest()
            ).encode(),
            "old/completed/0000029-attempt.json": json.dumps(
                _staged_manifest(29)
            ).encode(),
        }
    )
    run_volume = _Volume({**_published(20), "old/latest": b"old/weight_v000025"})
    point = prepare_attempt(
        run_volume,
        run_id="old",
        save_hf=_Config.save_hf,
        checkpoint_volume=checkpoint_volume,
    )
    assert point.staged
    assert point.iteration == 19
    assert (
        point.trainer_checkpoint
        == "/training-checkpoints/old/attempts/attempt/snapshots/0000019/checkpoints"
    )
    assert point.rollout_checkpoint.endswith("/hf_checkpoints/weight_v000019")
    assert run_volume.files["old/latest"] == b"old/weight_v000020"
    assert "old/checkpoints/latest_checkpointed_iteration.txt" not in run_volume.files


def test_staged_resume_falls_back_to_legacy_checkpoint_before_first_upload():
    volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"19",
            "old/checkpoints/iter_0000019/state": b"state",
            "old/hf_checkpoints/weight_v000019/.complete": b"",
            **_published(20),
        }
    )
    point = resolve_resume_point(
        volume,
        source_run_id="old",
        save_hf=_Config.save_hf,
        checkpoint_volume=_Volume({}),
    )
    assert not point.staged
    assert point.version == 20


def test_fresh_run_handles_modal_missing_completed_directory():
    from modal.exception import NotFoundError

    class EmptyCheckpointVolume:
        def iterdir(self, *args, **kwargs):
            raise NotFoundError('path "/old/completed" does not exist')

    assert (
        prepare_attempt(
            _Volume({}),
            run_id="old",
            save_hf=_Config.save_hf,
            checkpoint_volume=EmptyCheckpointVolume(),
        )
        is None
    )


def test_boot_selects_persisted_manifest_at_or_below_served_version(tmp_path):
    run_dir = tmp_path / "old"
    completed = run_dir / "completed"
    completed.mkdir(parents=True)
    for iteration in (19, 29):
        (completed / f"{iteration:07d}-attempt.json").write_text(
            json.dumps(_staged_manifest(iteration))
        )
    version, path = newest_persisted_export(run_dir, latest_version=25)
    assert version == 20
    assert path == tmp_path / _staged_manifest()["hf_directory"]


def test_staged_manifest_cannot_redirect_loader_outside_snapshot(tmp_path):
    run_dir = tmp_path / "old"
    (run_dir / "completed").mkdir(parents=True)
    manifest = {**_staged_manifest(), "hf_directory": "old/../../another-run"}
    (run_dir / "completed/0000019-attempt.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="relative path"):
        newest_persisted_export(run_dir, latest_version=20)


@pytest.mark.parametrize("staged_iteration,expected", [(19, 29), (39, 39)])
def test_resume_selects_newest_published_checkpoint_across_both_volumes(
    staged_iteration, expected
):
    run_volume = _Volume(
        {
            "old/checkpoints/latest_checkpointed_iteration.txt": b"29",
            "old/checkpoints/iter_0000029/state": b"state",
            "old/hf_checkpoints/weight_v000029/.complete": b"",
            **_published(30),
            **_published(staged_iteration + 1),
        }
    )
    checkpoint_volume = _Volume(
        {
            f"old/completed/{staged_iteration:07d}-attempt.json": json.dumps(
                _staged_manifest(staged_iteration)
            ).encode(),
        }
    )
    point = resolve_resume_point(
        run_volume,
        source_run_id="old",
        save_hf=_Config.save_hf,
        checkpoint_volume=checkpoint_volume,
    )
    assert point.iteration == expected
    assert point.staged == (expected == staged_iteration)


@pytest.mark.parametrize("checkpoint_kind", ["staged", "legacy", "boot"])
def test_prepare_attempt_refreshes_mounted_state_before_claim(
    tmp_path, checkpoint_kind
):
    from stitch.publish import claim_run
    from stitch.stores.modal_volume import ModalVolumeStore

    class MountedVolume(_Volume):
        def reload(self):
            for name, value in self.files.items():
                path = tmp_path / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)

    files = {"old/latest": b"old/weight_v000042"}
    checkpoint_volume = _Volume({})
    if checkpoint_kind == "staged":
        files.update(_published(30))
        checkpoint_volume.files["old/completed/0000029-attempt.json"] = json.dumps(
            _staged_manifest(29)
        ).encode()
    elif checkpoint_kind == "legacy":
        files.update(
            {
                "old/checkpoints/latest_checkpointed_iteration.txt": b"39",
                "old/checkpoints/iter_0000029/state": b"state",
                "old/hf_checkpoints/weight_v000029/.complete": b"",
                **_published(30),
            }
        )
    volume = MountedVolume(files)
    volume.reload()
    store = ModalVolumeStore(tmp_path / "old", run_id="old")
    assert store.read_pointer().version == 42

    point = prepare_attempt(
        volume,
        run_id="old",
        save_hf=_Config.save_hf,
        checkpoint_volume=checkpoint_volume,
    )
    boot_version = point.version if point else 0
    claim_run(store, None, "old", boot_version=boot_version)
    assert store.read_pointer().version == boot_version
    if checkpoint_kind == "legacy":
        assert (
            tmp_path / "old/checkpoints/latest_checkpointed_iteration.txt"
        ).read_text() == "29"
