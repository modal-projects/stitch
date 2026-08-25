"""GLM-5.2 FP8 inference pool — 3-replica mixed-version fleet, no trainer.

Clone of glm5_2_fp8.py with version leases, a 3-replica floor, and a
run volume isolated from production.
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import STITCH_LEASE_HEADER
from cookbook.inference_only.configs.glm5_2_fp8 import (
    CHECKPOINT_VOLUME_NAME,
    LOCAL_CHECKPOINT_PATH,
    ROLLOUT_CHECKPOINT_PATH,
    ROLLOUT_GPUS_PER_ENGINE,
    SERVED_MODEL_NAME,
    SGLANG_DELTA_UPDATE_MODE,
    SGLANG_SERVER_ARGS,
    SIDECAR_COMMIT_MODE,
    SIDECAR_FLUSH_CACHE_ON_COMMIT,
    SOURCE_MODEL,
    SOURCE_REVISION,
)

# Re-exported base-config fields (consumed via exp.*); listed so ruff F401
# treats the imports as used.
__all__ = [
    "CHECKPOINT_VOLUME_NAME",
    "LOCAL_CHECKPOINT_PATH",
    "ROLLOUT_CHECKPOINT_PATH",
    "ROLLOUT_GPUS_PER_ENGINE",
    "SERVED_MODEL_NAME",
    "SGLANG_DELTA_UPDATE_MODE",
    "SGLANG_SERVER_ARGS",
    "SIDECAR_COMMIT_MODE",
    "SIDECAR_FLUSH_CACHE_ON_COMMIT",
    "SOURCE_MODEL",
    "SOURCE_REVISION",
]

APP_NAME = "stitch-glm5-2-fp8-poc"
EXPERIMENT_VOLUME_NAME = "stitch-pp-poc-glm"
SIDECAR_VERSION_LEASE_TTL = 120  # seconds; straggler windows must exceed this
SIDECAR_LEASE_HEADER = STITCH_LEASE_HEADER

modal = ModalConfig(
    gpu="B300",
    routing_region="us-west",
    rollout_min_containers=3,
    rollout_max_containers=3,
    rollout_target_inputs=16,
    # Disk mode reconstructs the ~756 GB FP8 target on local storage.
    rollout_ephemeral_disk_mib=2 * 1024 * 1024,
    rollout_memory_mib=(512 * 1024, 2 * 1024 * 1024),
    router_registry_min_containers=1,
    router_min_containers=1,
    # some environments' huggingface-cache volume predates volume v2
    hf_cache_volume_version=1,
)
