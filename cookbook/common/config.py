"""``ModalConfig`` — the Modal-infrastructure half of an experiment config, shared by
every recipe (GPU model, region, rollout-pool sizing, prep topology).

The framework *training* config (``MilesConfig`` / ``SlimeConfig``) is self-contained in
each framework's subdir; only this infra grouping is common.
"""

from __future__ import annotations

from typing import Any, Literal

GPUType = Literal["H100", "H200", "B200", "B300", "A100"]


class ModalConfig:
    """Modal infrastructure: GPU model, region, rollout-pool sizing, prep topology."""

    gpu: GPUType = "B200"
    memory: tuple[int, int] | None = None
    cloud: str | None = None
    region: str | None = None
    rollout_min_containers: int = 2
    rollout_max_containers: int | None = None
    # Flash autoscaler target: keep well below the sglang engine concurrency so Flash
    # adds containers instead of packing requests until KV saturates and they stall.
    rollout_target_inputs: int | None = None
    proxy_regions: list[str] = ["us-west"]
    rollout_ephemeral_disk_mib: int | None = None
    rollout_memory_mib: int | None = None
    # Rollout router (stitch.routing): when enabled, a single always-on Router
    # container fronts the pool and the trainer's rollout_endpoint_url points at
    # it instead of the Flash gateway. The policy names come from the gorgo
    # package's registry ("session-affinity" reproduces the gateway's pinning;
    # "gorgo" adds prefix-cache/load/RTT-aware scoring).
    router_enabled: bool = False
    router_policy: str = "session-affinity"
    router_hyperparameters: dict[str, float] | None = None  # gorgo weight overrides
    router_tuner: dict[str, Any] | None = None  # POST /router/tune-shaped startup config
    router_tokenizer: bool = True  # tokenize prompts with the served model's HF tokenizer
    torch_dist_prep_nodes: int = 2
    torch_dist_prep_gpus_per_node: int = 8
    torch_dist_convert_extra_args: str = ""
    torch_dist_prep_ephemeral_disk_mib: int | None = None
    trainer_ephemeral_disk_mib: int | None = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
