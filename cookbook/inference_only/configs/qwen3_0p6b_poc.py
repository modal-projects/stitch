"""Qwen 3-0.6B inference pool — small openly-available model, 3×L40S.

Distinct volume namespace so nothing collides with real runs.
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH, STITCH_LEASE_HEADER

APP_NAME = "stitch-qwen3-0p6b-poc"
EXPERIMENT_VOLUME_NAME = "stitch-pp-poc"
CHECKPOINT_VOLUME_NAME = "inference-checkpoints"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = True
SIDECAR_VERSION_LEASE_TTL = 120  # 2 minutes: short enough for eval boundaries to exercise eviction
SIDECAR_LEASE_HEADER = STITCH_LEASE_HEADER
SGLANG_DELTA_UPDATE_MODE = "disk"

SOURCE_MODEL = "Qwen/Qwen3-0.6B"
SOURCE_REVISION = "main"
ROLLOUT_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-0p6b-poc"
SERVED_MODEL_NAME = "qwen3-0p6b-poc"
ROLLOUT_GPUS_PER_ENGINE = 1

SGLANG_SERVER_ARGS = {
    "--tp": "1",
    "--load-format": "auto",
    "--dtype": "auto",
    "--context-length": "8192",
}

modal = ModalConfig(
    gpu="L40S",
    routing_region="us-west",
    rollout_min_containers=3,
    rollout_max_containers=3,
    rollout_target_inputs=8,
    # Disk mode for updates. 512GiB is Modal's minimum ephemeral-disk allocation.
    rollout_ephemeral_disk_mib=512 * 1024,
    rollout_memory_mib=(64 * 1024, 128 * 1024),
    router_registry_min_containers=1,
    router_min_containers=1,
    # some environments' huggingface-cache volume predates volume v2
    hf_cache_volume_version=1,
)
