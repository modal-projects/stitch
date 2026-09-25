"""``ModalConfig`` — the shared Modal-infrastructure half of an experiment config.

Training arguments remain in each trainer integration; GPU selection, region, rollout-pool
sizing, and preparation topology live here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

# ``"X+"`` is Modal's tier floor: that class or better (e.g. "B200+" = B200 or B300).
GPUType = Literal["H100", "H200", "B200", "B200+", "B300", "A100"]


@dataclass(kw_only=True)
class ModalConfig:
    """Modal infrastructure: GPU model, region, rollout-pool sizing, prep topology."""

    gpu: GPUType = "B200"
    # Rollout-pool GPU; defaults to ``gpu``. Either a single type or a list of
    # acceptable types in preference order (Modal falls back down the list) — e.g.
    # ["B200", "B300"] schedules engines on whichever pool has capacity.
    rollout_gpu: GPUType | list[GPUType] | None = None
    rollout_cpu: float | None = None
    trainer_cpu: float | None = None
    trainer_memory_mib: tuple[int, int] | None = None
    cloud: str | None = None
    region: str | None = None
    draft_volume: str | None = None
    draft_volume_env: str | None = None
    kernel_cache_volume: str = "kernel-cache"
    rollout_min_containers: int = 2
    rollout_max_containers: int | None = None
    # Flash autoscaler target: keep well below sglang engine concurrency so Flash adds
    # containers instead of packing requests until KV saturates.
    rollout_target_inputs: int | None = None
    routing_region: str = "us-east"
    rollout_ephemeral_disk_mib: int | None = None
    rollout_memory_mib: tuple[int, int] | None = None
    torch_dist_prep_nodes: int = 2
    torch_dist_prep_gpus_per_node: int = 8
    torch_dist_convert_extra_args: str = ""
    torch_dist_prep_ephemeral_disk_mib: int | None = None
    trainer_ephemeral_disk_mib: int | None = None

    def rollout_gpus(self, per_engine: int) -> str | list[str]:
        """GPU request for one rollout engine: ``rollout_gpu`` falling back to ``gpu``,
        with the per-engine count attached (a list when multiple types are acceptable)."""
        spec = self.gpu if self.rollout_gpu is None else self.rollout_gpu
        if isinstance(spec, str):
            return f"{spec}:{per_engine}"
        return [f"{gpu_type}:{per_engine}" for gpu_type in spec]


def validate_serving_config(recipe: Any, *, gpus_per_engine: int) -> None:
    """Check the recipe contracts shared by preparation and rollout deployment."""
    model = getattr(recipe, "SOURCE_MODEL", None)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("SOURCE_MODEL must name a checkpoint repository")
    revision = getattr(recipe, "SOURCE_REVISION", None)
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
    ):
        raise ValueError("SOURCE_REVISION must pin a full checkpoint commit hash")

    args = recipe.SGLANG_SERVER_ARGS
    if gpus_per_engine <= 0 or int(args.get("--tp", 1)) != gpus_per_engine:
        raise ValueError(
            "SGLANG_SERVER_ARGS --tp must match the GPU count per rollout engine"
        )
    mode = getattr(recipe, "SGLANG_DELTA_UPDATE_MODE", None)
    if mode not in {"cpu", "disk"}:
        raise ValueError(f"Unsupported SGLANG_DELTA_UPDATE_MODE: {mode!r}")
    local_checkpoint = getattr(recipe, "LOCAL_CHECKPOINT_PATH", None)
    if local_checkpoint is not None and (
        not isinstance(local_checkpoint, str) or not local_checkpoint.strip()
    ):
        raise ValueError("LOCAL_CHECKPOINT_PATH must be a non-empty path or None")
    if mode == "disk" and local_checkpoint is None:
        raise ValueError("disk delta updates require LOCAL_CHECKPOINT_PATH")

    managed_args = {
        "--weight-update-staging",
        "--weight-update-local-checkpoint-dir",
        "--weight-version",
    }
    configured = managed_args.intersection(args)
    if configured:
        raise ValueError(
            "Stitch configures SGLang's staged-update lifecycle; remove managed "
            f"server arguments: {', '.join(sorted(configured))}"
        )
