"""Resolve and restore a saved Miles checkpoint for one Stitch run."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from cookbook.common.constants import STITCH_PATH, TRAINING_CHECKPOINTS_PATH
from cookbook.miles_disagg.checkpoint import relative_path
from stitch.types import WEIGHT_PREFIX, VersionRef

TRAINER_CALL_FILE = "trainer_call_id"


class ResumePointNotFound(ValueError):
    """The run has no complete checkpoint pair that can be resumed."""


@dataclass(frozen=True)
class ResumePoint:
    """One paired trainer/rollout checkpoint produced by a run: the export's
    published ``version`` and the Megatron ``iteration`` that produced it."""

    version: int
    iteration: int
    source_run_id: str
    trainer_checkpoint: str
    rollout_checkpoint: str
    staged: bool = False


def export_version(iteration: int) -> int:
    """The published weight version of the export saved at ``iteration``.

    A save at N precedes the publication of vN+1, and the runtime Miles patch
    keeps a resumed counter there, so this holds for a run's whole lifetime.
    """
    return iteration + 1


def validate_resumable_config(cfg: Any) -> None:
    """Require the checkpoint policy a trainer retry needs to resume a run."""
    validate_resume_config(cfg)
    if (interval := getattr(cfg, "save_interval", None)) is None or int(interval) <= 0:
        raise ValueError("resume requires a positive save_interval")
    if getattr(cfg, "no_save_optim", False):
        raise ValueError("resume requires optimizer checkpointing")
    if getattr(cfg, "no_save_rng", False):
        raise ValueError("resume requires RNG checkpointing")


def validate_resume_config(cfg: Any) -> None:
    """Require a complete saved trainer state and matching rollout checkpoint."""
    _validate_save_hf_template(getattr(cfg, "save_hf", None))
    # TODO: define the checkpoint-to-weight-version mapping for larger intervals.
    if int(getattr(cfg, "update_weights_interval", 1)) != 1:
        raise ValueError("resume currently requires update_weights_interval == 1")
    if getattr(cfg, "no_load_optim", False):
        raise ValueError("resume requires loading optimizer state")
    if getattr(cfg, "no_load_rng", False):
        raise ValueError("resume requires loading RNG state")


def resolve_resume_point(
    volume: Any,
    *,
    source_run_id: str,
    save_hf: str | None,
    checkpoint_volume: Any = None,
) -> ResumePoint:
    """Resolve the newest complete Megatron/HF checkpoint pair for a run."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", source_run_id) is None:
        raise ValueError(f"invalid resume run id: {source_run_id!r}")

    point = (
        _resolve_staged_resume_point(volume, checkpoint_volume, source_run_id)
        if checkpoint_volume is not None
        else None
    )
    try:
        legacy = _resolve_legacy_resume_point(volume, source_run_id, save_hf)
    except ResumePointNotFound:
        if point is not None:
            return point
        raise
    if point is not None and point.iteration >= legacy.iteration:
        return point
    return legacy


def _resolve_legacy_resume_point(
    volume: Any, source_run_id: str, save_hf: str | None
) -> ResumePoint:
    run_root = PurePosixPath(source_run_id)
    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    try:
        tracker_value = _read_volume_file(volume, str(tracker)).decode().strip()
    except FileNotFoundError as exc:
        raise ResumePointNotFound(
            f"run {source_run_id!r} has no saved Megatron checkpoint"
        ) from exc
    try:
        tracked_iteration = int(tracker_value)
    except ValueError as exc:
        raise ValueError(
            f"invalid checkpoint tracker {tracker}: {tracker_value!r}"
        ) from exc
    if tracked_iteration < 0:
        raise ValueError(
            f"invalid checkpoint iteration {tracked_iteration} in {tracker}"
        )

    iterations = []
    for entry in volume.iterdir(str(checkpoint_root), recursive=False):
        if match := re.fullmatch(r"iter_(\d+)", PurePosixPath(entry.path).name):
            iteration = int(match.group(1))
            # A fresh actor also reports iteration 0, so it never resumes.
            if 0 < iteration <= tracked_iteration:
                iterations.append(iteration)

    save_hf = _validate_save_hf_template(save_hf)
    for iteration in sorted(iterations, reverse=True):
        relative_hf = save_hf.format(rollout_id=iteration)
        hf_root = run_root / relative_hf
        try:
            _read_volume_file(volume, str(hf_root / ".complete"))
        except FileNotFoundError:
            continue
        # A crash between save and publish falls back one save interval.
        try:
            _check_published_version(
                volume, run_root, version=export_version(iteration)
            )
        except FileNotFoundError:
            continue
        break
    else:
        raise ResumePointNotFound(
            f"run {source_run_id!r} has no complete Megatron/HF checkpoint pair "
            f"with a published export at or before iteration {tracked_iteration}"
        )

    return ResumePoint(
        version=export_version(iteration),
        iteration=iteration,
        source_run_id=source_run_id,
        trainer_checkpoint=str(STITCH_PATH / checkpoint_root),
        rollout_checkpoint=str(STITCH_PATH / hf_root),
    )


def _check_published_version(
    volume: Any, run_root: PurePosixPath, *, version: int
) -> None:
    """Require ``updates/weight_vNNNNNN`` to exist and identify itself as ``version``."""
    index_path = (
        run_root
        / "updates"
        / f"{WEIGHT_PREFIX}{version:06d}"
        / "model.safetensors.index.json"
    )
    index = json.loads(_read_volume_file(volume, str(index_path)))
    published = int((index.get("metadata") or {})["version"])
    if published != version:
        raise ValueError(f"{index_path} identifies v{published}, not v{version}")


def prepare_attempt(
    volume: Any, *, run_id: str, save_hf: str | None, checkpoint_volume: Any = None
) -> ResumePoint | None:
    """Restore and return one trainer attempt's resume point, or rewind to the
    boot version and return None when the run has no complete pair yet.

    Every attempt calls this with identical inputs: that is what makes the
    first attempt, a retry, and a manual re-spawn one path.
    """
    try:
        point = resolve_resume_point(
            volume,
            source_run_id=run_id,
            save_hf=save_hf,
            checkpoint_volume=checkpoint_volume,
        )
    except ResumePointNotFound:
        restore_boot_pointer(volume, run_id)
        return None
    restore_resume_point(volume, point)
    return point


def restore_boot_pointer(volume: Any, run_id: str) -> None:
    """Rewind ``latest`` to the boot version when a run has nothing to resume."""
    pointer_path = f"{run_id}/latest"
    try:
        current = VersionRef.parse(
            _read_volume_file(volume, pointer_path).decode().strip()
        )
    except FileNotFoundError:
        return  # nothing claimed yet; the attempt's own claim writes v0
    if current.run_id != run_id:
        raise ValueError(f"latest belongs to run {current.run_id!r}, not {run_id!r}")
    if current.version == 0:
        return
    with volume.batch_upload(force=True) as upload:
        upload.put_file(BytesIO(VersionRef(run_id, 0).identity.encode()), pointer_path)


def record_trainer_call(volume: Any, run_id: str, call_id: str) -> None:
    """Record the run's active trainer call, so a takeover can cancel it."""
    with volume.batch_upload(force=True) as upload:
        upload.put_file(BytesIO(call_id.encode()), f"{run_id}/{TRAINER_CALL_FILE}")


def read_trainer_call(volume: Any, run_id: str) -> str | None:
    """Return the run's recorded trainer call id, or None before the first spawn."""
    try:
        return (
            _read_volume_file(volume, f"{run_id}/{TRAINER_CALL_FILE}").decode().strip()
        )
    except FileNotFoundError:
        return None


def restore_resume_point(volume: Any, point: ResumePoint) -> VersionRef:
    """Restore the trainer tracker and Stitch pointer to one checkpoint pair."""
    target = VersionRef(point.source_run_id, point.version)
    pointer_path = f"{point.source_run_id}/latest"
    try:
        current = VersionRef.parse(
            _read_volume_file(volume, pointer_path).decode().strip()
        )
    except FileNotFoundError as exc:
        raise ResumePointNotFound(
            f"run {point.source_run_id!r} has no Stitch latest pointer"
        ) from exc
    if current.run_id != target.run_id:
        raise ValueError(
            f"cannot restore {target.identity!r} from {current.identity!r}"
        )
    # One ahead is a publish interrupted before its pointer advance.
    if target.version > current.version + 1:
        raise ValueError(
            f"resume checkpoint v{target.version} is newer than latest v{current.version}"
        )

    with volume.batch_upload(force=True) as upload:
        upload.put_file(BytesIO(target.identity.encode()), pointer_path)
        if not point.staged:
            upload.put_file(
                BytesIO(str(point.iteration).encode()),
                f"{point.source_run_id}/checkpoints/latest_checkpointed_iteration.txt",
            )
    return target


def _validate_checkpoint_manifest(manifest: dict, run_id: str) -> None:
    """Validate the immutable checkpoint address before giving it to a loader."""
    iteration = manifest["iteration"]
    attempt = manifest["attempt_id"]
    if manifest["schema_version"] != 1 or manifest["run_id"] != run_id:
        raise ValueError("checkpoint manifest schema or run identity mismatch")
    if (
        not isinstance(iteration, int)
        or iteration < 0
        or manifest["version"] != export_version(iteration)
    ):
        raise ValueError("checkpoint manifest iteration/version mismatch")
    if "/" in relative_path(attempt):
        raise ValueError("invalid checkpoint attempt ID")
    root = f"{run_id}/attempts/{attempt}/snapshots/{iteration:07d}"
    if manifest["checkpoint_root"] != f"{root}/checkpoints":
        raise ValueError("checkpoint manifest points outside its snapshot")
    if hf := manifest["hf_directory"]:
        if not relative_path(hf).startswith(root + "/"):
            raise ValueError("HF checkpoint points outside its snapshot")


def _resolve_staged_resume_point(
    run_volume: Any, checkpoint_volume: Any, run_id: str
) -> ResumePoint | None:
    from modal.exception import NotFoundError

    try:
        entries = list(
            checkpoint_volume.iterdir(f"{run_id}/completed", recursive=False)
        )
    except (FileNotFoundError, NotFoundError):
        return None
    manifests = []
    for entry in entries:
        if PurePosixPath(entry.path).suffix != ".json":
            continue
        manifest = json.loads(_read_volume_file(checkpoint_volume, entry.path))
        _validate_checkpoint_manifest(manifest, run_id)
        if manifest["iteration"] > 0 and manifest["hf_directory"]:
            manifests.append(manifest)
    manifests.sort(
        key=lambda item: (item["iteration"], item["completed_at_ns"]), reverse=True
    )
    for manifest in manifests:
        try:
            _check_published_version(
                run_volume, PurePosixPath(run_id), version=manifest["version"]
            )
        except FileNotFoundError:
            continue
        return ResumePoint(
            version=manifest["version"],
            iteration=manifest["iteration"],
            source_run_id=run_id,
            trainer_checkpoint=str(
                TRAINING_CHECKPOINTS_PATH / manifest["checkpoint_root"]
            ),
            rollout_checkpoint=str(
                TRAINING_CHECKPOINTS_PATH / manifest["hf_directory"]
            ),
            staged=True,
        )
    return None


def newest_persisted_export(
    run_dir: Path, *, latest_version: int
) -> tuple[int, Path] | None:
    """Select a fully committed checkpoint from one snapshot of the mounted Volume."""
    manifests = []
    for path in (run_dir / "completed").glob("*.json"):
        manifest = json.loads(path.read_text())
        _validate_checkpoint_manifest(manifest, run_dir.name)
        if manifest["hf_directory"] and manifest["version"] <= latest_version:
            manifests.append(manifest)
    if not manifests:
        return None
    latest = max(manifests, key=lambda item: (item["version"], item["completed_at_ns"]))
    return latest["version"], run_dir.parent / latest["hf_directory"]


def newest_complete_export(
    run_dir: Path, *, save_hf: str, latest_version: int
) -> tuple[int, Path] | None:
    """The newest complete export at or below ``latest_version``, from a mounted
    run directory — the checkpoint a booting replica should load.

    Returns its published version and directory, or None before the first save.
    """
    exports = []
    export_root = run_dir / PurePosixPath(_validate_save_hf_template(save_hf)).parent
    for marker in export_root.glob("*/.complete"):
        export = marker.parent
        try:
            iteration = VersionRef.parse(export.name).version
        except ValueError:
            continue
        if export != run_dir / save_hf.format(rollout_id=iteration):
            continue
        if (version := export_version(iteration)) <= latest_version:
            exports.append((version, export))
    return max(exports) if exports else None


def _validate_save_hf_template(value: str | None) -> str:
    if not value:
        raise ValueError("resume requires save_hf")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("save_hf must be a run-relative path")
    try:
        formatted = value.format(rollout_id=0)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError("save_hf must be formattable with rollout_id") from exc
    if formatted == value:
        raise ValueError("save_hf must include a rollout_id format field")
    return value


def _read_volume_file(volume: Any, path: str) -> bytes:
    return b"".join(volume.read_file(path))
