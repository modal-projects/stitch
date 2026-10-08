"""Qwen3.6-35B-A3B BF16 code-agent training on MiMo-V2.6-RL-oss."""

import math
from dataclasses import replace

from cookbook.common.constants import DATA_PATH
from cookbook.miles_disagg import mimo_v2_6, swebench_config
from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_swebench_pro as base

APP_NAME = "stitch-qwen3-6-35b-mimo-code"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-6-35b-mimo-code"

SOURCE_MODEL = base.SOURCE_MODEL
SOURCE_REVISION = base.SOURCE_REVISION
BF16_CHECKPOINT_PATH = base.BF16_CHECKPOINT_PATH
ROLLOUT_CHECKPOINT_PATH = base.ROLLOUT_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = base.TORCH_DIST_CHECKPOINT_PATH
SERVED_CHECKPOINT_FORMAT = base.SERVED_CHECKPOINT_FORMAT
CHECKPOINT_PREP_REQUIRES_GPU = base.CHECKPOINT_PREP_REQUIRES_GPU
UNPACK_FUSED_EXPERTS = base.UNPACK_FUSED_EXPERTS
LOCAL_CHECKPOINT_PATH = base.LOCAL_CHECKPOINT_PATH
TRAINER_EXTRA_PIP_PACKAGES = base.TRAINER_EXTRA_PIP_PACKAGES
PREP_ENV = base.PREP_ENV

SIDECAR_COMMIT_MODE = base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_DELTA_UPDATE_MODE = base.SGLANG_DELTA_UPDATE_MODE

DATASET_PATH = DATA_PATH / mimo_v2_6.DATA_DIRNAME
MAX_SEQ_LEN = 262_144
AGENT_PROCESSES = 64
AGENT_THREADS_PER_PROCESS = 32
ROLLOUT_CONCURRENT_SAMPLES = AGENT_PROCESSES * AGENT_THREADS_PER_PROCESS
ROLLOUT_TARGET_INPUTS = {
    "H100": 16,
    "H200": 8,
    "B200": 16,
    "B300": 16,
}
ROLLOUT_SESSIONS_PER_POOL = ROLLOUT_CONCURRENT_SAMPLES // len(base.modal.rollout_pools)

SGLANG_SERVER_ARGS = {
    **base.ROLLOUT_SERVER_ARGS,
    "--context-length": str(MAX_SEQ_LEN),
    "--kv-cache-dtype": "fp8_e4m3",
    "--max-running-requests": "24",
    "--cuda-graph-max-bs-decode": "24",
}


def mimo_rollout_pool(pool):
    target_inputs = ROLLOUT_TARGET_INPUTS[pool.gpu]
    min_containers = math.ceil(ROLLOUT_SESSIONS_PER_POOL / target_inputs)
    return replace(
        pool,
        min_containers=min_containers,
        max_containers=min_containers * 3 // 2,
        target_inputs=target_inputs,
        sglang_args={
            **pool.sglang_args,
            "--context-length": str(MAX_SEQ_LEN),
            "--kv-cache-dtype": "fp8_e4m3",
            "--max-running-requests": "24",
            "--cuda-graph-max-bs-decode": "24",
        },
    )


modal = replace(
    base.modal,
    rollout_pools=tuple(mimo_rollout_pool(pool) for pool in base.modal.rollout_pools),
)


class _Miles(base._Miles):
    sglang_server_concurrency = ROLLOUT_CONCURRENT_SAMPLES
    async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES
    session_server_workers = 64
    session_samples_timeout = 600
    miles_router_timeout = 3600

    num_rollout = 10
    rollout_batch_size = 64
    n_samples_per_prompt = 8
    global_batch_size = rollout_batch_size * n_samples_per_prompt
    rollout_top_p = 0.95
    rollout_top_k = 1024
    rollout_max_response_len = 32_768
    max_seq_len = MAX_SEQ_LEN

    context_parallel_size = 2
    max_tokens_per_gpu = 16_384
    log_probs_max_tokens_per_gpu = 16_384

    wandb_group = "qwen3-6-35b-mimo-v2-6-code"
    prometheus_run_name = wandb_group

    environment = {
        **swebench_config.environment(
            sandbox_app="qwen3-6-35b-mimo-code-sandbox",
            processes=AGENT_PROCESSES,
            threads_per_process=AGENT_THREADS_PER_PROCESS,
        ),
        "MODAL_SWE_TASKS_DIR": f"{DATASET_PATH}/tasks/code",
        "MODAL_SWE_AGENT_PROFILE": "mimo-code-bash",
        "MODAL_SWE_MAX_STEPS": "500",
        "MODAL_SWE_EPISODE_TIMEOUT": "4800",
        "MODAL_SWE_MODEL_REQUEST_TIMEOUT": "3600",
        "MODAL_SWE_EXEC_TIMEOUT": "300",
        "MODAL_SWE_MEMORY_MIB": "8192",
    }

    def prepare_data(self) -> None:
        mimo_v2_6.prepare_mimo_v2_6(DATASET_PATH)


arguments = swebench_config.arguments()
arguments["prompt_data"] = f"{DATASET_PATH}/code.jsonl"
miles = _Miles(**arguments)
