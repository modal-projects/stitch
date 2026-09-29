"""Resolve and restore a saved Miles checkpoint for one Stitch run."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from cookbook.common.constants import STITCH_PATH
from stitch.types import WEIGHT_PREFIX, VersionRef

TRAINER_CALL_FILE = "trainer_call_id"


class ResumePointNotFound(ValueError):
    """The run has no complete checkpoint pair that can be resumed."""


@dataclass(frozen=True)
class ResumePoint:
    """One saved trainer state and its published rollout version.

    A single-view run also carries the matching full rollout checkpoint;
    multi-view runs recover from their selected delta lineages instead.
    """

    version: int
    iteration: int
    source_run_id: str
    trainer_checkpoint: str
    rollout_checkpoint: str | None


def export_version(iteration: int) -> int:
    """The published weight version of the export saved at ``iteration``.

    A save at N precedes the publication of vN+1, and the runtime Miles patch
    keeps a resumed counter there, so this holds for a run's whole lifetime.
    """
    return iteration + 1


def validate_resumable_config(cfg: Any, *, weight_views: Iterable[str] = ()) -> None:
    """Require the checkpoint policy a trainer retry needs to resume a run."""
    validate_resume_config(cfg, weight_views=weight_views)
    if (interval := getattr(cfg, "save_interval", None)) is None or int(interval) <= 0:
        raise ValueError("resume requires a positive save_interval")
    if getattr(cfg, "no_save_optim", False):
        raise ValueError("resume requires optimizer checkpointing")
    if getattr(cfg, "no_save_rng", False):
        raise ValueError("resume requires RNG checkpointing")


def validate_resume_config(cfg: Any, *, weight_views: Iterable[str] = ()) -> None:
    """Require the state needed to restore both trainer and selected rollout views."""
    if not tuple(weight_views):
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
    weight_views: Iterable[str] = (),
) -> ResumePoint:
    """Resolve the newest trainer checkpoint represented by the rollout state."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", source_run_id) is None:
        raise ValueError(f"invalid resume run id: {source_run_id!r}")

    run_root = PurePosixPath(source_run_id)
    views = tuple(weight_views)
    pointers = _read_pointers(volume, run_root, views)

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

    save_hf = None if views else _validate_save_hf_template(save_hf)
    for iteration in sorted(iterations, reverse=True):
        version = export_version(iteration)
        if any(version > pointer.version + 1 for pointer in pointers.values()):
            continue
        hf_root = None
        if save_hf is not None:
            relative_hf = save_hf.format(rollout_id=iteration)
            hf_root = run_root / relative_hf
            try:
                _read_volume_file(volume, str(hf_root / ".complete"))
            except FileNotFoundError:
                continue
        # Each pointer is a durable publication record. Only a one-ahead view
        # needs its delta checked: publication completed but pointer advancement
        # did not. Views already at or beyond this checkpoint committed it.
        try:
            for view, pointer in pointers.items():
                if version == pointer.version + 1:
                    _check_published_version(
                        volume, run_root, version=version, weight_view=view
                    )
        except FileNotFoundError:
            continue
        break
    else:
        checkpoint_kind = (
            "trainer checkpoint represented by every selected rollout view"
            if views
            else "Megatron/HF checkpoint pair"
        )
        raise ResumePointNotFound(
            f"run {source_run_id!r} has no complete {checkpoint_kind} "
            f"with a published export at or before iteration {tracked_iteration}"
        )

    return ResumePoint(
        version=version,
        iteration=iteration,
        source_run_id=source_run_id,
        trainer_checkpoint=str(STITCH_PATH / checkpoint_root),
        rollout_checkpoint=str(STITCH_PATH / hf_root) if hf_root is not None else None,
    )


def _check_published_version(
    volume: Any,
    run_root: PurePosixPath,
    *,
    version: int,
    weight_view: str | None = None,
) -> None:
    """Require ``updates/weight_vNNNNNN`` to exist and identify itself as ``version``."""
    update_root = run_root / "updates"
    if weight_view is not None:
        update_root /= weight_view
    index_path = update_root / f"{WEIGHT_PREFIX}{version:06d}"
    index_path /= "model.safetensors.index.json"
    index = json.loads(_read_volume_file(volume, str(index_path)))
    published = int((index.get("metadata") or {})["version"])
    if published != version:
        raise ValueError(f"{index_path} identifies v{published}, not v{version}")


def prepare_attempt(
    volume: Any,
    *,
    run_id: str,
    save_hf: str | None,
    weight_views: Iterable[str] = (),
) -> ResumePoint | None:
    """Restore and return one trainer attempt's resume point, or rewind to the
    boot version and return None when the run has no complete pair yet.

    Every attempt calls this with identical inputs: that is what makes the
    first attempt, a retry, and a manual re-spawn one path.
    """
    views = tuple(weight_views)
    try:
        point = resolve_resume_point(
            volume,
            source_run_id=run_id,
            save_hf=save_hf,
            weight_views=views,
        )
    except ResumePointNotFound:
        point = None
    if point is None:
        restore_boot_pointer(volume, run_id, weight_views=views)
    else:
        restore_resume_point(volume, point, weight_views=views)
    # API uploads do not refresh the mounted pointer or Megatron tracker.
    # The subsequent claim and checkpoint loader read those mounted files.
    volume.reload()
    return point


def restore_boot_pointer(
    volume: Any, run_id: str, *, weight_views: Iterable[str] = ()
) -> None:
    """Rewind ``latest`` to the boot version when a run has nothing to resume."""
    views = tuple(weight_views)
    paths = _pointer_paths(PurePosixPath(run_id), views)
    existing = {}
    for view, path in paths.items():
        try:
            current = VersionRef.parse(
                _read_volume_file(volume, str(path)).decode().strip()
            )
        except FileNotFoundError:
            continue
        if current.run_id != run_id:
            raise ValueError(
                f"latest belongs to run {current.run_id!r}, not {run_id!r}"
            )
        if current.version:
            existing[view] = path
    if not existing:
        return
    with volume.batch_upload(force=True) as upload:
        for path in existing.values():
            upload.put_file(BytesIO(VersionRef(run_id, 0).identity.encode()), str(path))


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


def restore_resume_point(
    volume: Any,
    point: ResumePoint,
    *,
    weight_views: Iterable[str] = (),
) -> VersionRef:
    """Restore the trainer tracker and Stitch pointer to one checkpoint pair."""
    target = VersionRef(point.source_run_id, point.version)
    views = tuple(weight_views)
    run_root = PurePosixPath(point.source_run_id)
    pointers = _read_pointers(volume, run_root, views)
    for current in pointers.values():
        # One ahead is a publish interrupted before its pointer advance.
        if target.version > current.version + 1:
            raise ValueError(
                f"resume checkpoint v{target.version} is newer than latest "
                f"v{current.version}"
            )

    with volume.batch_upload(force=True) as upload:
        for pointer_path in _pointer_paths(run_root, views).values():
            upload.put_file(BytesIO(target.identity.encode()), str(pointer_path))
        upload.put_file(
            BytesIO(str(point.iteration).encode()),
            f"{point.source_run_id}/checkpoints/latest_checkpointed_iteration.txt",
        )
    return target


def _pointer_paths(
    run_root: PurePosixPath, weight_views: tuple[str, ...]
) -> dict[str | None, PurePosixPath]:
    if not weight_views:
        return {None: run_root / "latest"}
    return {view: run_root / "latest" / view for view in weight_views}


def _read_pointers(
    volume: Any, run_root: PurePosixPath, weight_views: tuple[str, ...]
) -> dict[str | None, VersionRef]:
    pointers = {}
    missing = []
    for view, path in _pointer_paths(run_root, weight_views).items():
        try:
            pointer = VersionRef.parse(
                _read_volume_file(volume, str(path)).decode().strip()
            )
        except FileNotFoundError:
            missing.append(view)
            continue
        if pointer.run_id != run_root.name:
            raise ValueError(
                f"latest belongs to run {pointer.run_id!r}, not {run_root.name!r}"
            )
        pointers[view] = pointer
    if not pointers:
        raise ResumePointNotFound(f"run {run_root.name!r} has no Stitch latest pointer")
    if missing:
        if all(pointer.version == 0 for pointer in pointers.values()):
            pointers.update({view: VersionRef(run_root.name, 0) for view in missing})
            return pointers
        names = ", ".join(repr(view) for view in missing)
        raise ValueError(
            f"run {run_root.name!r} has no latest pointer for views: {names}"
        )
    return pointers


def newest_complete_export(
    run_dir: Path,
    *,
    save_hf: str,
    latest_version: int,
    weight_view: str | None = None,
) -> tuple[int, Path] | None:
    """The newest complete export at or below ``latest_version``, from a mounted
    run directory — the checkpoint a booting replica should load.

    Returns its published version and directory, or None before the first save.
    """
    exports = []
    export_root = run_dir / PurePosixPath(_validate_save_hf_template(save_hf)).parent
    marker_pattern = (
        "*/.complete" if weight_view is None else f"*/{weight_view}/.complete"
    )
    for marker in export_root.glob(marker_pattern):
        checkpoint = marker.parent
        export = checkpoint if weight_view is None else checkpoint.parent
        try:
            iteration = VersionRef.parse(export.name).version
        except ValueError:
            continue
        if export != run_dir / save_hf.format(rollout_id=iteration):
            continue
        if (version := export_version(iteration)) <= latest_version:
            exports.append((version, checkpoint))
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
