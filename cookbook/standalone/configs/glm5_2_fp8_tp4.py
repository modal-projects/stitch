"""Standalone GLM-5.2-FP8 rollout pool on 4xB300 engines.

Denser sibling of ``glm5_2_fp8``: TP4 engines with BF16 target KV. One engine
holds 552,704 KV tokens. On the pinned agentic mix, 12 concurrent users peaked
at 475 output tok/s with 72% active-KV occupancy and 1.48s p50 latency; 16 users
reached 95% active KV and throughput collapsed. Engine admission therefore
stops at 16 while the router targets the measured sweet spot of 12.
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH, DRAFT_PATH

APP_NAME = "stitch-standalone-glm5-2-fp8-tp4"
EXPERIMENT_VOLUME_NAME = "stitch-standalone-glm5-2-fp8-tp4"
CHECKPOINT_VOLUME_NAME = "miles-checkpoints"
LOCAL_CHECKPOINT_PATH = None

SOURCE_MODEL = "zai-org/GLM-5.2-FP8"
SOURCE_REVISION = "ba978f7d347eaf65d22f1a86833408afdb953541"
BASE_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-2-fp8"
DFLASH_VOLUME = "dflash-checkpoints"
DFLASH_CHECKPOINT_PATH = DRAFT_PATH / "zai-org/GLM-5.2/dflash/draft-step-103000"

ROLLOUT_GPUS_PER_ENGINE = 4
ROLLOUT_INPUTS_PER_ENGINE = 12
ROLLOUT_MAX_RUNNING_REQUESTS = 16
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

# Serving path follows autoinference recipe rec-1911384c55b6
# (glm-5-2-fp8-dflash-bf16kv-s103k-b300x4); the Stitch sidecar variant is
# measured with DFLASH block 8 and CUDA graphs capped at batch 16 above.
SGLANG_SERVER_ENV = {
    "SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA": "1",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_NSA_FORCE_MLA": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}

# The benchmarked recipe pins only the block size; the remaining draft
# parameters ride the engine defaults it was measured with.
DFLASH_SERVER_ARGS = {
    "--speculative-algorithm": "DFLASH",
    "--speculative-dflash-block-size": "8",
    "--speculative-draft-model-path": str(DFLASH_CHECKPOINT_PATH),
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
    # attention / quant (from the benchmarked recipe)
    "--dsa-prefill-backend": "trtllm",
    "--dsa-decode-backend": "trtllm",
    "--dsa-topk-backend": "sgl-kernel",
    "--kv-cache-dtype": "bf16",
    "--fp8-gemm-backend": "deep_gemm",
    # MoE runs the routed runner so per-token expert choices can be returned
    # and replayed; shared-experts fusion would fold them out of reach.
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    # DFlash drafts from a fixed external drafter. The target-model optimizer
    # never updates it, so the draft stays fixed across weight updates (CPU
    # staging covers the target model only).
    **DFLASH_SERVER_ARGS,
    # memory / batching: 0.90 static leaves ~25.5 GB/GPU after CUDA graphs,
    # which still fits the 8 GB weight-staging compile groups.
    "--mem-fraction-static": "0.90",
    "--chunked-prefill-size": "16384",
    "--max-running-requests": str(ROLLOUT_MAX_RUNNING_REQUESTS),
    "--max-queued-requests": str(ROLLOUT_MAX_QUEUED_REQUESTS),
    "--cuda-graph-max-bs": "16",
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
    # (755.6 GB + 189.4 GB x 4 measured for GLM-5.2 FP8 at TP4).
    rollout_memory_mib=(1024 * 1024, 3 * 1024 * 1024),
    rollout_min_containers=2,
    rollout_max_containers=None,
    rollout_target_inputs=ROLLOUT_INPUTS_PER_ENGINE,
    draft_volume=DFLASH_VOLUME,
    routing_region="us-west",
    # CPU updates use ephemeral disk only for runtime scratch and spill.
    rollout_ephemeral_disk_mib=524_288,
)
