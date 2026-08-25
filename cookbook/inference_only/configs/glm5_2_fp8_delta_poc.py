"""GLM-5.2 FP8 inference pool — cpu-mode deltas.

cpu mode rejects FULL publications, so a run on this config must be born
at v0 and only ever see delta publishes (eval_client --delta --base-version 0
on a fresh run volume).
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import MINUTES, STITCH_LEASE_HEADER
from cookbook.inference_only.configs.glm5_2_fp8 import (
    CHECKPOINT_VOLUME_NAME,
    ROLLOUT_CHECKPOINT_PATH,
    ROLLOUT_GPUS_PER_ENGINE,
    SERVED_MODEL_NAME,
    SIDECAR_COMMIT_MODE,
    SIDECAR_FLUSH_CACHE_ON_COMMIT,
    SOURCE_MODEL,
    SOURCE_REVISION,
)
from cookbook.inference_only.configs.glm5_2_fp8 import (
    SGLANG_SERVER_ARGS as _BASE_SERVER_ARGS,
)

# Re-exported base-config fields (consumed via exp.*); listed so ruff F401
# treats the imports as used.
__all__ = [
    "CHECKPOINT_VOLUME_NAME",
    "ROLLOUT_CHECKPOINT_PATH",
    "ROLLOUT_GPUS_PER_ENGINE",
    "SERVED_MODEL_NAME",
    "SIDECAR_COMMIT_MODE",
    "SIDECAR_FLUSH_CACHE_ON_COMMIT",
    "SOURCE_MODEL",
    "SOURCE_REVISION",
]

APP_NAME = "stitch-glm5-2-fp8-poc-delta"
EXPERIMENT_VOLUME_NAME = "stitch-pp-poc-glm"
# cpu mode keeps the canonical checkpoint in host RAM, not on local disk.
LOCAL_CHECKPOINT_PATH = None
SIDECAR_VERSION_LEASE_TTL = 120  # seconds; straggler windows must exceed this
SIDECAR_LEASE_HEADER = STITCH_LEASE_HEADER
SGLANG_DELTA_UPDATE_MODE = "cpu"
# cpu-cache boot (weight load + DeepGEMM) exceeds the shared 60 min Flash floor.
SERVER_STARTUP_TIMEOUT = 120 * MINUTES

SGLANG_SERVER_ARGS = {
    **_BASE_SERVER_ARGS,
    # cpu delta-update mode requires the CPU weight cache (coupling enforced by
    # cookbook/common/server.py: cache enabled exactly when mode == 'cpu').
    "--enable-cpu-weight-cache": "",
    "--cpu-weight-cache-max-compile-group-gb": "8",
}

modal = ModalConfig(
    gpu="B300",
    routing_region="us-west",
    rollout_min_containers=3,
    rollout_max_containers=3,
    rollout_target_inputs=16,
    # CPU updates use ephemeral disk only for runtime scratch and spill (miles
    # precedent: 512GiB, vs. the 2TiB the disk-mode config requests).
    rollout_ephemeral_disk_mib=524_288,
    # CPU mode retains the ~756GB FP8 canonical checkpoint and TP rank images in
    # memory (miles GLM precedent: request 1TiB, cap 3TiB).
    rollout_memory_mib=(1024 * 1024, 3 * 1024 * 1024),
    router_registry_min_containers=1,
    router_min_containers=1,
    # some environments' huggingface-cache volume predates volume v2
    hf_cache_volume_version=1,
)
