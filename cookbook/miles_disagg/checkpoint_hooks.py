"""Publish completed Miles checkpoints after every host's Volume commit."""

from pathlib import Path
from typing import Any

from cookbook.common import process
from cookbook.miles_disagg.resume import CHECKPOINT_COMPLETE_MARKER

COMPLETION_HOOK = "cookbook.miles_disagg.checkpoint_hooks.commit_checkpoint"


def commit_checkpoint(
    args: Any, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str | None
) -> None:
    """Called on all actor ranks after native, HF, and sampler writes close."""
    volume = _volume(args)
    error = None
    if process.dist_is_container_leader():
        try:
            volume.commit()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    failures = [error for error in process.dist_all_gather_object(error) if error]
    if failures:
        raise RuntimeError("Checkpoint Volume commit failed: " + "; ".join(failures))
    if process.dist_rank() in (None, 0):
        if (
            hf_checkpoint_dir is None
            or not (Path(hf_checkpoint_dir) / ".complete").is_file()
        ):
            raise RuntimeError(f"Checkpoint {rollout_id} has no completed HF export")
        marker = Path(checkpoint_dir) / CHECKPOINT_COMPLETE_MARKER
        marker.touch()
        volume.commit()


def _volume(args: Any) -> Any:
    import modal

    return modal.Volume.from_name(args.experiment_volume_name, version=2)
