"""Resolve and restore a saved Miles checkpoint for one Stitch run."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

from cookbook.common.constants import STITCH_PATH
from stitch.types import WEIGHT_PREFIX, VersionRef

RESUME_POINT_ENV = "STITCH_RESUME_POINT"
TRAINER_CALL_FILE = "trainer_call_id"


class ResumePointNotFound(ValueError):
    """The run has no complete checkpoint pair that can be resumed."""


@dataclass(frozen=True)
class ResumePoint:
    """One paired trainer/rollout checkpoint produced by a run.

    ``version`` is the export's published weight version; ``iteration`` is the
    Megatron iteration that produced it (``version == iteration + 1``).
    """

    version: int
    iteration: int
    source_run_id: str
    trainer_checkpoint: str
    rollout_checkpoint: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> ResumePoint:
        data = json.loads(value)
        return cls(
            version=int(data["version"]),
            iteration=int(data["iteration"]),
            source_run_id=str(data["source_run_id"]),
            trainer_checkpoint=str(data["trainer_checkpoint"]),
            rollout_checkpoint=str(data["rollout_checkpoint"]),
        )


def export_version(iteration: int) -> int:
    """Return the published weight version of the export saved at ``iteration``.

    A save at iteration N precedes the publication of vN+1, and a resumed
    trainer continues the version counter from there (the runtime Miles patch),
    so this mapping holds for a run's whole lifetime: version numbers are never
    relabeled across attempts.
    """
    return iteration + 1


def validate_auto_resume_config(cfg: Any) -> None:
    """Require an explicit, resumable checkpoint policy before automatic resume."""
    validate_resume_config(cfg)
    if (interval := getattr(cfg, "save_interval", None)) is None or int(interval) <= 0:
        raise ValueError("--auto-resume requires a positive save_interval")
    if getattr(cfg, "no_save_optim", False):
        raise ValueError("--auto-resume requires optimizer checkpointing")
    if getattr(cfg, "no_save_rng", False):
        raise ValueError("--auto-resume requires RNG checkpointing")


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
) -> ResumePoint:
    """Resolve the newest complete Megatron/HF checkpoint pair for a run."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", source_run_id) is None:
        raise ValueError(f"invalid resume run id: {source_run_id!r}")

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
        tracked_version = int(tracker_value)
    except ValueError as exc:
        raise ValueError(
            f"invalid checkpoint tracker {tracker}: {tracker_value!r}"
        ) from exc
    if tracked_version < 0:
        raise ValueError(f"invalid checkpoint version {tracked_version} in {tracker}")

    iterations = []
    for entry in volume.iterdir(str(checkpoint_root), recursive=False):
        if match := re.fullmatch(r"iter_(\d+)", PurePosixPath(entry.path).name):
            iteration = int(match.group(1))
            # Iteration 0 is never a resume point: a fresh actor also reports
            # iteration 0, which is how the version counter tells fresh from
            # resumed (the runtime Miles patch).
            if 0 < iteration <= tracked_version:
                iterations.append(iteration)

    save_hf = _validate_save_hf_template(save_hf)
    for iteration in sorted(iterations, reverse=True):
        relative_hf = save_hf.format(rollout_id=iteration)
        hf_root = run_root / relative_hf
        try:
            _read_volume_file(volume, str(hf_root / ".complete"))
        except FileNotFoundError:
            continue
        # A resumed run republishes the abandoned suffix in place, so the resume
        # point's own publication must exist and identify itself — a crash
        # between save and publish falls back one save interval instead.
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
            f"with a published export at or before iteration {tracked_version}"
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
    volume: Any, *, run_id: str, save_hf: str | None
) -> ResumePoint | None:
    """Derive one trainer attempt's resume state from the run volume alone.

    Every attempt — the first, a Modal retry, or a manual re-spawn — calls this
    with identical inputs: the newest complete checkpoint pair is restored and
    returned, and a run that crashed before its first pair restarts from
    scratch with the pointer rewound to the boot version, so replicas abandon
    the unsaved suffix either way.
    """
    try:
        point = resolve_resume_point(volume, source_run_id=run_id, save_hf=save_hf)
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
    """Record the run's active trainer call; a newer spawn supersedes the old."""
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
    # One ahead is a publish interrupted between its verified files and the
    # pointer advance; restoring forward by one completes it.
    if target.version > current.version + 1:
        raise ValueError(
            f"resume checkpoint v{target.version} is newer than latest v{current.version}"
        )

    with volume.batch_upload(force=True) as upload:
        upload.put_file(BytesIO(target.identity.encode()), pointer_path)
        upload.put_file(
            BytesIO(str(point.iteration).encode()),
            f"{point.source_run_id}/checkpoints/latest_checkpointed_iteration.txt",
        )
    return target


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
