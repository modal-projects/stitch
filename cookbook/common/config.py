"""``ModalConfig`` — the shared Modal-infrastructure half of an experiment config.

Training arguments remain in each trainer integration; GPU selection, region, rollout-pool
sizing, and preparation topology live here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# ``"X+"`` is Modal's tier floor: that class or better (e.g. "B200+" = B200 or B300).
GPUType = Literal["H100", "H200", "B200", "B200+", "B300", "A100"]


@dataclass(frozen=True, kw_only=True)
class RolloutPoolConfig:
    """One independently scaled rollout-engine configuration."""

    name: str
    gpu: GPUType | list[GPUType]
    gpus_per_engine: int
    target_inputs: int
    sglang_args: dict[str, str]
    min_containers: int = 1
    max_containers: int | None = None
    memory_mib: tuple[int, int] | None = None
    ephemeral_disk_mib: int | None = None
    cloud: str | None = None
    region: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def gpu_request(self) -> str | list[str]:
        if isinstance(self.gpu, str):
            return f"{self.gpu}:{self.gpus_per_engine}"
        return [f"{gpu}:{self.gpus_per_engine}" for gpu in self.gpu]


@dataclass(kw_only=True)
class ModalConfig:
    """Modal infrastructure: GPU model, region, rollout-pool sizing, prep topology."""

    gpu: GPUType = "B200"
    # Rollout-pool GPU; defaults to ``gpu``. Either a single type or a list of
    # acceptable types in preference order (Modal falls back down the list) — e.g.
    # ["B200", "B300"] schedules engines on whichever pool has capacity.
    rollout_gpu: GPUType | list[GPUType] | None = None
    # Separate pools guarantee a heterogeneous fleet. A ``rollout_gpu`` list is
    # only a per-container fallback preference and does not provide that guarantee.
    rollout_pools: tuple[RolloutPoolConfig, ...] = ()
    rollout_cpu: float | tuple[float, float] | None = None
    trainer_cpu: float | tuple[float, float] | None = None
    trainer_memory_mib: int | tuple[int, int] | None = None
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
    rollout_memory_mib: int | tuple[int, int] | None = None
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

    def resolved_rollout_pools(
        self,
        *,
        default_gpus_per_engine: int,
        default_target_inputs: int,
        default_sglang_args: dict[str, str],
    ) -> tuple[RolloutPoolConfig, ...]:
        """Return complete per-pool engine configurations.

        Existing single-pool recipes are lifted into the same representation at the
        deployment boundary. Explicit pools are already self-contained.
        """
        if self.rollout_pools:
            if self.rollout_gpu is not None:
                raise ValueError("rollout_pools and rollout_gpu are mutually exclusive")
            names = [pool.name for pool in self.rollout_pools]
            if len(names) != len(set(names)):
                raise ValueError("rollout pool names must be unique")
            if any(not name.isidentifier() for name in names):
                raise ValueError("rollout pool names must be valid Python identifiers")
            if any(not isinstance(pool.gpu, str) for pool in self.rollout_pools):
                raise ValueError(
                    "each explicit rollout pool must select one exact GPU type"
                )
            if any(pool.min_containers < 1 for pool in self.rollout_pools):
                raise ValueError("each rollout pool must keep at least one container")
            if any(pool.gpus_per_engine < 1 for pool in self.rollout_pools):
                raise ValueError("each rollout pool must use at least one GPU")
            if any(pool.target_inputs < 1 for pool in self.rollout_pools):
                raise ValueError("each rollout pool target_inputs must be positive")
            if any(
                pool.max_containers is not None
                and pool.max_containers < pool.min_containers
                for pool in self.rollout_pools
            ):
                raise ValueError(
                    "each rollout pool max_containers must cover min_containers"
                )
            return self.rollout_pools

        spec = self.gpu if self.rollout_gpu is None else self.rollout_gpu
        return (
            RolloutPoolConfig(
                name="Server",
                gpu=spec,
                gpus_per_engine=default_gpus_per_engine,
                target_inputs=default_target_inputs,
                sglang_args=dict(default_sglang_args),
                min_containers=self.rollout_min_containers,
                max_containers=self.rollout_max_containers,
            ),
        )

    @property
    def rollout_replica_floor(self) -> int:
        if self.rollout_pools:
            return sum(pool.min_containers for pool in self.rollout_pools)
        return self.rollout_min_containers


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

    explicit_pools = recipe.modal.rollout_pools
    serving_configs = (
        tuple(
            (pool.name, pool.gpus_per_engine, pool.target_inputs, pool.sglang_args)
            for pool in explicit_pools
        )
        if explicit_pools
        else (
            (
                "Server",
                gpus_per_engine,
                recipe.modal.rollout_target_inputs or 1,
                recipe.SGLANG_SERVER_ARGS,
            ),
        )
    )
    managed_args = {
        "--weight-update-staging",
        "--weight-update-local-checkpoint-dir",
        "--weight-version",
    }
    for name, pool_gpus, target_inputs, args in serving_configs:
        if pool_gpus <= 0 or int(args.get("--tp", 1)) != pool_gpus:
            raise ValueError(
                f"{name} SGLANG --tp must match its GPU count per rollout engine"
            )
        if explicit_pools:
            try:
                max_running = int(args["--max-running-requests"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must set an integer --max-running-requests"
                ) from exc
            if target_inputs > max_running:
                raise ValueError(
                    f"{name} target_inputs cannot exceed --max-running-requests"
                )
        configured = managed_args.intersection(args)
        if configured:
            raise ValueError(
                "Stitch configures SGLang's staged-update lifecycle; remove managed "
                f"server arguments from {name}: {', '.join(sorted(configured))}"
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
