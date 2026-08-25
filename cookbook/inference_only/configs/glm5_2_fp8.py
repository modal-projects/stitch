"""Open-weight GLM-5.2 FP8 inference pool — no trainer.

Disk updates so a later publisher can land full checkpoints or deltas without
the CPU-cache RAM budget. Flush on commit so evals do not reuse KV from a
previous version.
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH

APP_NAME = "stitch-glm5-2-fp8"
EXPERIMENT_VOLUME_NAME = "stitch-inference-glm5-2-fp8"
CHECKPOINT_VOLUME_NAME = "inference-checkpoints"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = True
SGLANG_DELTA_UPDATE_MODE = "disk"

SOURCE_MODEL = "zai-org/GLM-5.2-FP8"
SOURCE_REVISION = "ba978f7d347eaf65d22f1a86833408afdb953541"
ROLLOUT_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-2-fp8"
SERVED_MODEL_NAME = "glm-5-2-fp8"
ROLLOUT_GPUS_PER_ENGINE = 4

SGLANG_SERVER_ARGS = {
    "--tp": "4",
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--weight-loader-drop-cache-after-load": "",
    "--dtype": "auto",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
    "--context-length": "32768",
    "--attention-backend": "dsa",
    "--dsa-prefill-backend": "flashmla_sparse",
    "--dsa-decode-backend": "flashmla_kv",
    "--dsa-topk-backend": "flashinfer",
    "--page-size": "64",
    "--ep-size": str(ROLLOUT_GPUS_PER_ENGINE),
    "--moe-dense-tp-size": "1",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--mem-fraction-static": "0.80",
    "--chunked-prefill-size": "16384",
    "--skip-server-warmup": "",
}

modal = ModalConfig(
    gpu="B300",
    routing_region="us-west",
    rollout_min_containers=1,
    rollout_max_containers=1,
    rollout_target_inputs=16,
    # Disk mode reconstructs the ~756 GB FP8 target on local storage.
    rollout_ephemeral_disk_mib=2 * 1024 * 1024,
    rollout_memory_mib=(512 * 1024, 2 * 1024 * 1024),
    router_registry_min_containers=1,
    router_min_containers=1,
)
