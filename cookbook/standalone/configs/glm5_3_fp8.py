"""GLM-5.3 FP8 serving on eight B300 GPUs per replica, without a draft model."""

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH

APP_NAME = "stitch-standalone-glm5-3-fp8"
EXPERIMENT_VOLUME_NAME = "stitch-standalone-glm5-3-fp8"
CHECKPOINT_VOLUME_NAME = "miles-checkpoints"
LOCAL_CHECKPOINT_PATH = None

SOURCE_MODEL = "zai-org/GLM-5.3"
SOURCE_REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
BASE_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-aca966e4-fp8"

ROLLOUT_GPUS_PER_ENGINE = 8
ROLLOUT_INPUTS_PER_ENGINE = 16
ROLLOUT_MAX_RUNNING_REQUESTS = 32
MAX_SEQ_LEN = 65_536

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
# Let SGLang's 20-second generation-health deadline return its own verdict.
SIDECAR_ENGINE_HEALTH_TIMEOUT = 30.0
SGLANG_DELTA_UPDATE_MODE = "cpu"

SGLANG_SERVER_ENV = {
    "SGLANG_DSA_FUSE_TOPK": "1",
    # DeepGEMM PDL launch attributes race between serving and background staging.
    "SGLANG_DEEPGEMM_PDL": "0",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}

SGLANG_SERVER_ARGS = {
    "--tp": str(ROLLOUT_GPUS_PER_ENGINE),
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--enable-cpu-weight-cache": "",
    "--cpu-weight-cache-max-compile-group-gb": "8",
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
    "--served-model-name": SOURCE_MODEL,
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--context-length": str(MAX_SEQ_LEN),
    "--dsa-prefill-backend": "trtllm",
    "--dsa-decode-backend": "trtllm",
    "--dsa-topk-backend": "sgl-kernel",
    "--kv-cache-dtype": "bfloat16",
    "--fp8-gemm-backend": "deep_gemm",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--mem-fraction-static": "0.80",
    "--chunked-prefill-size": "65536",
    "--max-running-requests": str(ROLLOUT_MAX_RUNNING_REQUESTS),
    "--max-queued-requests": "4",
    "--cuda-graph-max-bs-decode": str(ROLLOUT_MAX_RUNNING_REQUESTS),
    "--disable-cuda-graph-padding": "",
    "--enable-metrics": "",
    "--enable-metrics-for-all-schedulers": "",
    "--decode-log-interval": "1000",
    "--log-level-http": "warning",
    "--enable-return-routed-experts": "",
    "--sampling-mask-max-tokens": "8192",
}

modal = ModalConfig(
    gpu="B300",
    rollout_gpu="B300",
    # CPU staging retains both the canonical checkpoint and TP rank images.
    rollout_memory_mib=(1024 * 1024, 3 * 1024 * 1024),
    rollout_min_containers=2,
    rollout_target_inputs=ROLLOUT_INPUTS_PER_ENGINE,
    routing_region="us-west",
    rollout_ephemeral_disk_mib=524_288,
)
