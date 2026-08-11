"""Resolve a saved Miles checkpoint into a fresh Stitch run."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any

from cookbook.common.constants import STITCH_PATH

RESUME_POINT_ENV = "STITCH_RESUME_POINT"


class ResumePointNotFound(ValueError):
    """The run has no complete checkpoint pair that can be resumed."""


@dataclass(frozen=True)
class ResumePoint:
    """One paired trainer/rollout checkpoint produced by a previous run."""

    version: int
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
            source_run_id=str(data["source_run_id"]),
            trainer_checkpoint=str(data["trainer_checkpoint"]),
            rollout_checkpoint=str(data["rollout_checkpoint"]),
        )


def saved_checkpoint_version(rollout_id: int, *, resumed: bool) -> int:
    """Return the Stitch version stored by a Miles ``save_hf`` checkpoint.

    A fresh trainer starts rollout 0 from Stitch v0, so save N precedes the
    publication of vN+1. A resumed trainer starts rollout N+1 from Stitch vN,
    so subsequent save IDs and Stitch versions are equal.
    """
    return rollout_id if resumed else rollout_id + 1


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
    """Resolve Miles' latest Megatron tracker and matching complete HF export."""
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
        version = int(tracker_value)
    except ValueError as exc:
        raise ValueError(
            f"invalid checkpoint tracker {tracker}: {tracker_value!r}"
        ) from exc
    if version < 0:
        raise ValueError(f"invalid checkpoint version {version} in {tracker}")

    relative_hf = _validate_save_hf_template(save_hf).format(rollout_id=version)
    hf_root = run_root / relative_hf
    complete_marker = hf_root / ".complete"
    try:
        _read_volume_file(volume, str(complete_marker))
    except FileNotFoundError as exc:
        raise ResumePointNotFound(
            f"run {source_run_id!r} checkpoint v{version} has no complete HF export"
        ) from exc

    return ResumePoint(
        version=version,
        source_run_id=source_run_id,
        trainer_checkpoint=str(STITCH_PATH / checkpoint_root),
        rollout_checkpoint=str(STITCH_PATH / hf_root),
    )


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
