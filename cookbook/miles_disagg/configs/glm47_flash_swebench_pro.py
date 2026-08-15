"""Fully async GLM-4.7-Flash GRPO on SWE-bench Pro."""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH, DATA_PATH
from cookbook.miles_disagg.config import MilesConfig
from cookbook.miles_disagg.swebench_pro import prepare_swebench_pro

APP_NAME = "stitch-glm47-flash-swebench-pro"
EXPERIMENT_VOLUME_NAME = "stitch-miles-glm47-flash-swebench-pro"
LOCAL_CHECKPOINT_PATH = None

SOURCE_MODEL = "zai-org/GLM-4.7-Flash"
SOURCE_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm4-7-flash-bf16"
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm4-7-flash-torch-dist"
SWEBENCH_PRO_PATH = DATA_PATH / "swebench-pro"
SERVED_CHECKPOINT_FORMAT = "bf16"
PREP_ENV = {
    "CONVERT_KEEP_PP1": "1",  # Miles otherwise expands PP1 to the torchrun world.
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",  # Required by Megatron TP/CP.
}

TRAINER_EXTRA_PIP_PACKAGES = (
    "harbor[modal,huggingface]==0.20.0",
    "mini-swe-agent==2.4.5",
    "swebench==4.1.0",
    "modal==1.5.3",
)
MEGATRON_RUNTIME_PATCHES = [
    "/root/cookbook/miles_disagg/patches/megatron-hdo-dp-reshardable-step.patch",
]

TRAINER_NODES = 4
GPUS_PER_TRAINER_NODE = 8
ROLLOUT_GPUS_PER_ENGINE = 1
ROLLOUT_INPUTS_PER_ENGINE = 20
ROLLOUT_CONCURRENT_SAMPLES = 640  # saturates the 40-worker rollout harness
ROLLOUT_MIN_CONTAINERS = 48  # 48 rollout GPUs at one GPU per engine
ROLLOUT_MAX_RUNNING_REQUESTS = 28
ROLLOUT_MAX_QUEUED_REQUESTS = 4  # backpressure
MAX_SEQ_LEN = 65_536

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"
SGLANG_SERVER_ENV = {
    "SGLANG_SANITIZE_NAN_LOGITS": "true",
}
SGLANG_SERVER_ARGS = {
    "--load-format": "auto",
    "--enable-cpu-weight-cache": "",
    "--cpu-weight-cache-max-compile-group-gb": "8",
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "900",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--context-length": str(MAX_SEQ_LEN + 8),
    "--mem-fraction-static": "0.85",
    "--chunked-prefill-size": "8192",
    "--max-running-requests": str(ROLLOUT_MAX_RUNNING_REQUESTS),
    "--max-queued-requests": str(ROLLOUT_MAX_QUEUED_REQUESTS),
    "--cuda-graph-max-bs-decode": str(ROLLOUT_MAX_RUNNING_REQUESTS),
    "--enable-metrics": "",
    "--enable-metrics-for-all-schedulers": "",
    "--decode-log-interval": "1000",
    "--log-level-http": "warning",
    "--enable-return-routed-experts": "",
    "--sampling-mask-max-tokens": "8192",
}

modal = ModalConfig(
    gpu="H200",
    rollout_gpu="H200",
    cloud="aws",
    trainer_memory_mib=(1024 * 1024, 3 * 1024 * 1024),
    rollout_memory_mib=(512 * 1024, 1024 * 1024),
    rollout_min_containers=ROLLOUT_MIN_CONTAINERS,
    rollout_max_containers=None,
    rollout_target_inputs=ROLLOUT_INPUTS_PER_ENGINE,
    rollout_ephemeral_disk_mib=524_288,
    trainer_ephemeral_disk_mib=1_048_576,
    torch_dist_prep_nodes=1,
    torch_dist_prep_gpus_per_node=8,
    torch_dist_convert_extra_args=(
        "--tensor-model-parallel-size 2 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 4 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--sequence-parallel "
        "--moe-token-dispatcher-type alltoall"
    ),
    torch_dist_prep_ephemeral_disk_mib=524_288,
)


class _Miles(MilesConfig):
    megatron_model_type = "glm4.7-flash"
    async_mode = True

    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    megatron_to_hf_mode = "raw"

    actor_num_nodes = TRAINER_NODES
    actor_num_gpus_per_node = GPUS_PER_TRAINER_NODE
    num_gpus_per_node = GPUS_PER_TRAINER_NODE
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = ROLLOUT_GPUS_PER_ENGINE
    rollout_endpoint_url = None
    sglang_server_concurrency = ROLLOUT_CONCURRENT_SAMPLES

    custom_rollout_request_hook_path = (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        "rollout_request_retry_attempts": 1200,
        "rollout_request_retry_sleep": 1.0,
        "rollout_request_timeout_secs": 300,
    }

    update_weights_interval = 1
    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_buffer_size = 2 * 1024**3
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"

    prompt_data = f"{SWEBENCH_PRO_PATH}/test.jsonl"
    input_key = "prompt"
    metadata_key = "metadata"
    apply_chat_template = False
    rollout_shuffle = True
    balance_data = True

    fully_async = True
    pause_generation_mode = "in_place"
    rollout_submission_granularity = "sample"
    custom_rollout_log_function_path = "modal_swe_metrics.log_rollout_data"
    custom_generate_function_path = (
        "miles.rollout.generate_hub.agentic_tool_call.generate"
    )
    custom_agent_function_path = "modal_swe_agent_function.run"
    custom_rm_path = "modal_swe_agent_function.reward_func"
    tito_model = "glm47"
    use_session_server = "v2"
    session_sample_picker_path = "modal_swe_agent_function.pick_latest_leaf"
    session_sample_postprocessor_path = "modal_swe_agent_function.postprocess_samples"
    session_server_port = [30000, 30064]
    session_server_startup_timeout_seconds = 180

    num_rollout = 500
    save_interval = 20
    save_hf = "hf_checkpoints/weight_v{rollout_id:06d}"
    rollout_batch_size = 32
    n_samples_per_prompt = 8
    global_batch_size = 256
    rollout_temperature = 1.0
    rollout_top_p = 0.95
    rollout_top_k = 8192
    rollout_max_response_len = 8192
    max_seq_len = MAX_SEQ_LEN
    max_weight_staleness = 6
    async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES
    async_data_buffer_capacity_factor = (
        0.5  # at most half a batch of completed R3 state
    )
    async_unused_samples_handler = "drop"
    eval_interval = None

    use_rollout_routing_replay = True
    use_fault_tolerance = True

    tensor_model_parallel_size = 2
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 4
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    distributed_timeout_minutes = 60
    moe_token_dispatcher_type = "alltoall"
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 16384
    log_probs_max_tokens_per_gpu = 16384
    log_probs_chunk_size = 8192
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    train_backend = "megatron"
    grad_reduce_in_bf16 = True
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    advantage_estimator = "grpo"
    use_rollout_logprobs = False
    use_tis = True
    get_mismatch_metrics = True
    custom_tis_function_path = (
        "miles.backends.training_utils.loss_hub.corrections.icepop_function"
    )
    tis_clip_low = 0.5
    tis_clip = 5.0
    kl_coef = 0.0
    use_kl_loss = False
    kl_loss_coef = None
    kl_loss_type = None
    observe_training_entropy = True
    entropy_coef = 0.0
    eps_clip = 0.2
    eps_clip_high = 0.28

    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    use_wandb = True
    wandb_project = "fully-async-rl-modal"
    wandb_group = "glm47-flash-swebench-pro"
    disable_wandb_random_suffix = True
    use_prometheus = True
    prometheus_port = 9090
    prometheus_run_name = "glm47-flash-swebench-pro"

    environment = {
        "PYTHONPATH": (
            "/root/Megatron-LM:/root/miles:/root/miles/examples/swe-agent:"
            "/root/miles/examples/experimental/modal-swe"
        ),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "RAY_health_check_timeout_ms": "60000",
        "RAY_health_check_failure_threshold": "30",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_MODEL_NAME": "model",
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
        "LITELLM_LOG": "ERROR",
        "MODAL_SWE_TASKS_DIR": f"{SWEBENCH_PRO_PATH}/tasks",
        "MODAL_SWE_SANDBOX_APP": "glm47-flash-swebench-pro-sandbox",
        "MODAL_SWE_MAX_STEPS": "256",
        "MODAL_SWE_EPISODE_TIMEOUT": "7200",
        "MODAL_SWE_MODEL_REQUEST_TIMEOUT": "1800",
        "MODAL_SWE_EXEC_TIMEOUT": "120",
        "MODAL_SWE_SETUP_TIMEOUT": "240",
        "MODAL_SWE_OUTPUT_HARD_LIMIT_BYTES": str(16 * 1024 * 1024),
        "MODAL_SWE_VERIFY_TIMEOUT": "3600",
        "MODAL_SWE_INJECT_PYTEST_REPORTER": "0",
        "MODAL_SWE_CPUS": "2",
        "MODAL_SWE_MEMORY_MIB": "16384",
        "MODAL_SWE_AGENT_PROCESSES": "40",
        "MODAL_SWE_AGENT_THREADS_PER_PROCESS": "16",
        "MODAL_SWE_SANDBOX_BOOT_CONCURRENCY_PER_PROCESS": "4",
    }

    def prepare_data(self) -> None:
        prepare_swebench_pro(SWEBENCH_PRO_PATH)


miles = _Miles()
