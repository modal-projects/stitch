"""Fully asynchronous Qwen3.6-35B-A3B GRPO on SWE-bench Pro."""

from copy import deepcopy

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH
from cookbook.miles_disagg.configs import glm47_flash_swebench_pro as base

APP_NAME = "stitch-qwen3-6-35b-swebench-pro"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-6-35b-swebench-pro"
SOURCE_MODEL = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-bf16-unpacked"
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = (
    CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-torch-dist-tp2-ep8"
)
SERVED_CHECKPOINT_FORMAT = "bf16"
CHECKPOINT_PREP_REQUIRES_GPU = False
UNPACK_FUSED_EXPERTS = True
LOCAL_CHECKPOINT_PATH = None
PREP_ENV = dict(base.PREP_ENV)
TRAINER_EXTRA_PIP_PACKAGES = base.TRAINER_EXTRA_PIP_PACKAGES
MEGATRON_RUNTIME_PATCHES = [
    "/root/cookbook/miles_disagg/patches/megatron-r3-dispatch.patch",
    *base.MEGATRON_RUNTIME_PATCHES,
]

MAX_SEQ_LEN = 65_536
AGENT_PROCESSES = 16
AGENT_THREADS_PER_PROCESS = 16
ROLLOUT_CONCURRENT_SAMPLES = AGENT_PROCESSES * AGENT_THREADS_PER_PROCESS
SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"
SGLANG_SERVER_ENV = dict(base.SGLANG_SERVER_ENV)
SGLANG_SERVER_ARGS = {
    **base.SGLANG_SERVER_ARGS,
    "--reasoning-parser": "qwen3",
    "--tool-call-parser": "qwen3_coder",
    "--context-length": str(MAX_SEQ_LEN + 8),
    "--max-running-requests": "16",
    "--max-queued-requests": "4",
    "--cuda-graph-max-bs-decode": "16",
    "--moe-runner-backend": "triton",
}

modal = ModalConfig(
    gpu="B300",
    rollout_gpu="B200+",
    trainer_memory_mib=(262_144, 786_432),
    rollout_memory_mib=(262_144, 524_288),
    rollout_min_containers=16,
    rollout_max_containers=None,
    rollout_target_inputs=8,
    rollout_ephemeral_disk_mib=524_288,
    trainer_ephemeral_disk_mib=524_288,
    torch_dist_prep_nodes=1,
    torch_dist_prep_gpus_per_node=8,
    torch_dist_convert_extra_args=(
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--sequence-parallel "
        "--moe-token-dispatcher-type alltoall"
    ),
    torch_dist_prep_ephemeral_disk_mib=524_288,
)

miles = deepcopy(base.miles)
miles.megatron_model_type = "qwen3.6-35B-A3B"
miles.hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
miles.ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
miles.actor_num_nodes = 2
miles.actor_num_gpus_per_node = 8
miles.num_gpus_per_node = 8
miles.rollout_num_gpus_per_engine = 1
miles.sglang_server_concurrency = ROLLOUT_CONCURRENT_SAMPLES
miles.session_server_port = [30000, 30008]
miles.session_server_startup_timeout_seconds = 600
miles.tito_model = "qwen35"
miles.num_rollout = 500
miles.save_interval = 20
miles.rollout_batch_size = 32
miles.n_samples_per_prompt = 8
miles.global_batch_size = miles.rollout_batch_size * miles.n_samples_per_prompt
miles.max_seq_len = MAX_SEQ_LEN
miles.rollout_max_response_len = 8192
miles.rollout_top_k = 20
miles.context_parallel_size = 1
miles.max_tokens_per_gpu = 4096
miles.log_probs_max_tokens_per_gpu = 4096
miles.async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES
miles.async_data_buffer_capacity_factor = 2.0
miles.wandb_group = "qwen3-6-35b-swebench-pro"
miles.prometheus_run_name = miles.wandb_group
miles.environment = {
    **base.miles.environment,
    "MODAL_SWE_SANDBOX_APP": "qwen3-6-35b-swebench-pro-sandbox",
    "MODAL_SWE_MAX_STEPS": "256",
    "MODAL_SWE_EPISODE_TIMEOUT": "7200",
    "MODAL_SWE_AGENT_PROCESSES": str(AGENT_PROCESSES),
    "MODAL_SWE_AGENT_THREADS_PER_PROCESS": str(AGENT_THREADS_PER_PROCESS),
}
