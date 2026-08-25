"""Standalone GLM-5.2-FP8 rollout pool for SWE-bench-Pro-style agentic RL."""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH, DRAFT_PATH

APP_NAME = "stitch-standalone-glm5-2-fp8"
EXPERIMENT_VOLUME_NAME = "stitch-standalone-glm5-2-fp8"
CHECKPOINT_VOLUME_NAME = "miles-checkpoints"
LOCAL_CHECKPOINT_PATH = None

SOURCE_MODEL = "zai-org/GLM-5.2-FP8"
SOURCE_REVISION = "ba978f7d347eaf65d22f1a86833408afdb953541"
BASE_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-2-fp8"
DFLASH_VOLUME = "dflash-checkpoints"
DFLASH_CHECKPOINT_PATH = DRAFT_PATH / "zai-org/GLM-5.2/dflash/draft-step-103000"

ROLLOUT_GPUS_PER_ENGINE = 8
ROLLOUT_INPUTS_PER_ENGINE = 16
# Engine admission stays at the decode-graph capacity; the Flash target keeps
# headroom below it so the autoscaler adds containers before KV saturates.
ROLLOUT_MAX_RUNNING_REQUESTS = 32
# Bound scheduler-poll and routing skew to the headroom above the routing target.
# Sustained excess load gets a retryable 503 instead of an unbounded engine queue.
ROLLOUT_MAX_QUEUED_REQUESTS = 4
MAX_SEQ_LEN = 65_536
# SGLang's request boundary needs a small physical-context margin. The harness still
# truncates every trainable session to MAX_SEQ_LEN.
SGLANG_CONTEXT_LENGTH = MAX_SEQ_LEN + 8

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"

SGLANG_SERVER_ENV = {
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_NSA_FORCE_MLA": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}

DFLASH_SERVER_ARGS = {
    "--speculative-algorithm": "DFLASH",
    "--speculative-attention-mode": "decode",
    "--speculative-dflash-block-size": "8",
    "--speculative-num-draft-tokens": "8",
    "--speculative-num-steps": "1",
    "--speculative-eagle-topk": "1",
    "--speculative-draft-attention-backend": "flashinfer",
    "--speculative-draft-load-format": "fastsafetensors",
    "--speculative-draft-model-path": str(DFLASH_CHECKPOINT_PATH),
    "--speculative-draft-model-quantization": "unquant",
    "--speculative-draft-window-size": "4096",
}

SGLANG_SERVER_ARGS = {
    "--tp": str(ROLLOUT_GPUS_PER_ENGINE),
    # loading / elastic refit
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--enable-cpu-weight-cache": "",
    "--cpu-weight-cache-max-compile-group-gb": "8",
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
    # model
    "--served-model-name": SOURCE_MODEL,
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--context-length": str(SGLANG_CONTEXT_LENGTH),
    # attention / quant (from the smoked recipe)
    "--dsa-prefill-backend": "trtllm",
    "--dsa-decode-backend": "trtllm",
    "--dsa-topk-backend": "sgl-kernel",
    "--kv-cache-dtype": "bfloat16",
    "--fp8-gemm-backend": "deep_gemm",
    # MoE runs the routed runner so per-token expert choices can be returned
    # and replayed; shared-experts fusion would fold them out of reach.
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    # DFlash drafts from a fixed external drafter. The target-model optimizer
    # never updates it, so the draft stays fixed across weight updates (CPU
    # staging covers the target model only).
    **DFLASH_SERVER_ARGS,
    # memory / batching
    "--mem-fraction-static": "0.80",
    "--chunked-prefill-size": "65536",
    "--max-running-requests": str(ROLLOUT_MAX_RUNNING_REQUESTS),
    "--max-queued-requests": str(ROLLOUT_MAX_QUEUED_REQUESTS),
    "--cuda-graph-max-bs": "32",
    "--disable-cuda-graph-padding": "",
    # observability
    "--enable-metrics": "",
    "--enable-metrics-for-all-schedulers": "",
    "--decode-log-interval": "1000",
    "--log-level-http": "warning",
    # RL: the trainer replays per-token routed experts and receives the realized
    # top-p sampling mask with each response.
    "--enable-return-routed-experts": "",
    "--sampling-mask-max-tokens": "8192",
}

modal = ModalConfig(
    gpu="B300",
    rollout_gpu="B300",
    # CPU staging keeps the canonical checkpoint and TP rank images in host RAM
    # (1.63 TB measured for GLM-5.2 FP8 with both in memory).
    rollout_memory_mib=(1024 * 1024, 3 * 1024 * 1024),
    rollout_min_containers=2,
    rollout_max_containers=None,
    rollout_target_inputs=ROLLOUT_INPUTS_PER_ENGINE,
    draft_volume=DFLASH_VOLUME,
    routing_region="us-west",
    # CPU updates use ephemeral disk only for runtime scratch and spill.
    rollout_ephemeral_disk_mib=524_288,
)
