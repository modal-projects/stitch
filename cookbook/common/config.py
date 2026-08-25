"""``ModalConfig`` — the shared Modal-infrastructure half of an experiment config.

Training arguments remain in each trainer integration; GPU selection, region, rollout-pool
sizing, and preparation topology live here.
"""

from __future__ import annotations

from typing import Any, Literal

# ``"X+"`` is Modal's tier floor: that class or better (e.g. "B200+" = B200 or B300).
GPUType = Literal["H100", "H200", "B200", "B200+", "B300", "A100"]


class ModalConfig:
    """Modal infrastructure: GPU model, region, rollout-pool sizing, prep topology."""

    gpu: GPUType = "B200"
    # Rollout-pool GPU; defaults to ``gpu``. Either a single type or a list of
    # acceptable types in preference order (Modal falls back down the list) — e.g.
    # ["B200", "B300"] schedules engines on whichever pool has capacity.
    rollout_gpu: GPUType | list[GPUType] | None = None
    trainer_memory_mib: tuple[int, int] | None = None
    cloud: str | None = None
    region: str | None = None
    draft_volume: str | None = None
    draft_volume_env: str | None = None
    rollout_min_containers: int = 2
    rollout_max_containers: int | None = None
    # Flash autoscaler target: keep well below sglang engine concurrency so Flash adds
    # containers instead of packing requests until KV saturates.
    rollout_target_inputs: int | None = None
    routing_region: str = "us-east"
    # Session-routing LB deployed alongside the pool (cookbook/common/router.py):
    # registry replica floor, proxy replica floor, and the proxy's autoscale target.
    router_registry_min_containers: int = 2
    router_min_containers: int = 2
    router_target_concurrency: int = 50
    rollout_ephemeral_disk_mib: int | None = None
    rollout_memory_mib: tuple[int, int] | None = None
    # Version of the shared "huggingface-cache" Volume to mount. v2 is the current
    # layout; older environments may still have a v1 volume (set 1 there).
    hf_cache_volume_version: int = 2
    torch_dist_prep_nodes: int = 2
    torch_dist_prep_gpus_per_node: int = 8
    torch_dist_convert_extra_args: str = ""
    torch_dist_prep_ephemeral_disk_mib: int | None = None
    trainer_ephemeral_disk_mib: int | None = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def rollout_gpus(self, per_engine: int) -> str | list[str]:
        """GPU request for one rollout engine: ``rollout_gpu`` falling back to ``gpu``,
        with the per-engine count attached (a list when multiple types are acceptable)."""
        spec = self.gpu if self.rollout_gpu is None else self.rollout_gpu
        if isinstance(spec, str):
            return f"{spec}:{per_engine}"
        return [f"{gpu_type}:{per_engine}" for gpu_type in spec]
