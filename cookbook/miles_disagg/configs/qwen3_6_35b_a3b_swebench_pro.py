"""Qwen3.6-35B-A3B BF16 GRPO on SWE-bench Pro."""

from cookbook.common.config import ModalConfig, RolloutPoolConfig
from cookbook.common.constants import CHECKPOINTS_PATH
from cookbook.miles_disagg import swebench_config
from cookbook.miles_disagg.config import MilesConfig
from cookbook.miles_disagg.swebench_pro import prepare_swebench_pro

APP_NAME = "stitch-qwen3-6-35b-swebench-pro"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-6-35b-swebench-pro"
SOURCE_MODEL = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
BF16_CHECKPOINT_PATH = (
    CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-bf16-unpacked-native"
)
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = (
    CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-torch-dist-bf16-tp2-ep8"
)
SERVED_CHECKPOINT_FORMAT = "bf16"
CHECKPOINT_PREP_REQUIRES_GPU = False
UNPACK_FUSED_EXPERTS = True
LOCAL_CHECKPOINT_PATH = None
TRAINER_EXTRA_PIP_PACKAGES = swebench_config.TRAINER_PACKAGES
PREP_ENV = {
    "CONVERT_KEEP_PP1": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
}

MAX_SEQ_LEN = 65_536
AGENT_PROCESSES = 64
AGENT_THREADS_PER_PROCESS = 16
ROLLOUT_CONCURRENT_SAMPLES = AGENT_PROCESSES * AGENT_THREADS_PER_PROCESS
GPUS_PER_NODE = 8
ROLLOUT_MIN_NODES_PER_POOL = 4
ROLLOUT_MAX_NODES_PER_POOL = 8
ROLLOUT_MIN_GPUS_PER_POOL = ROLLOUT_MIN_NODES_PER_POOL * GPUS_PER_NODE
ROLLOUT_MAX_GPUS_PER_POOL = ROLLOUT_MAX_NODES_PER_POOL * GPUS_PER_NODE

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"
ROLLOUT_SERVER_ARGS = {
    "--dtype": "bfloat16",
    "--load-format": "safetensors",
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "900",
    "--reasoning-parser": "qwen3",
    "--tool-call-parser": "qwen3_coder",
    "--context-length": str(MAX_SEQ_LEN + 8),
    "--linear-attn-prefill-backend": "flashinfer",
    "--linear-attn-decode-backend": "flashinfer",
    "--mamba-ssm-dtype": "bfloat16",
    "--mamba-radix-cache-strategy": "extra_buffer",
    "--moe-runner-backend": "triton",
    "--kv-cache-dtype": "bfloat16",
    "--chunked-prefill-size": "8192",
    "--max-running-requests": "16",
    "--cuda-graph-max-bs-decode": "16",
    "--enable-metrics": "",
    "--enable-metrics-for-all-schedulers": "",
    "--decode-log-interval": "1000",
    "--log-level-http": "warning",
    "--enable-return-routed-experts": "",
    "--sampling-mask-max-tokens": "8192",
    "--weight-update-max-compile-group-gb": "2",
}


def rollout_server_args(
    *, tp: int, attention_backend: str, mem_fraction_static: float
) -> dict[str, str]:
    return {
        **ROLLOUT_SERVER_ARGS,
        "--tp": str(tp),
        "--attention-backend": attention_backend,
        "--mem-fraction-static": str(mem_fraction_static),
    }


modal = ModalConfig(
    gpu="B300",
    rollout_pools=(
        RolloutPoolConfig(
            name="ServerH100TP2",
            gpu="H100",
            gpus_per_engine=2,
            target_inputs=8,
            sglang_args=rollout_server_args(
                tp=2, attention_backend="fa3", mem_fraction_static=0.7
            ),
            min_containers=ROLLOUT_MIN_GPUS_PER_POOL // 2,
            max_containers=ROLLOUT_MAX_GPUS_PER_POOL // 2,
        ),
        RolloutPoolConfig(
            name="ServerH200TP1",
            gpu="H200",
            gpus_per_engine=1,
            target_inputs=8,
            sglang_args=rollout_server_args(
                tp=1, attention_backend="fa3", mem_fraction_static=0.7
            ),
            min_containers=ROLLOUT_MIN_GPUS_PER_POOL,
            max_containers=ROLLOUT_MAX_GPUS_PER_POOL,
        ),
        RolloutPoolConfig(
            name="ServerB200TP1",
            gpu="B200",
            gpus_per_engine=1,
            target_inputs=8,
            sglang_args=rollout_server_args(
                tp=1,
                attention_backend="trtllm_mha",
                mem_fraction_static=0.8,
            ),
            min_containers=ROLLOUT_MIN_GPUS_PER_POOL,
            max_containers=ROLLOUT_MAX_GPUS_PER_POOL,
        ),
        RolloutPoolConfig(
            name="ServerB300TP1",
            gpu="B300",
            gpus_per_engine=1,
            target_inputs=8,
            sglang_args=rollout_server_args(
                tp=1,
                attention_backend="trtllm_mha",
                mem_fraction_static=0.85,
            ),
            min_containers=ROLLOUT_MIN_GPUS_PER_POOL,
            max_containers=ROLLOUT_MAX_GPUS_PER_POOL,
        ),
    ),
    trainer_memory_mib=(262_144, 786_432),
    rollout_memory_mib=(262_144, 524_288),
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
        "--mtp-num-layers 0 "
        "--sequence-parallel "
        "--moe-token-dispatcher-type alltoall"
    ),
    torch_dist_prep_ephemeral_disk_mib=524_288,
)


class _Miles(MilesConfig):
    megatron_model_type = "qwen3.6-35B-A3B"
    async_mode = True

    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    megatron_to_hf_mode = "raw"
    model_name = "qwen3_6"
    mtp_num_layers = 0
    hf_export_source_tensor_prefixes = ["model.visual.", "mtp."]
    transformer_impl = "transformer_engine"
    bf16 = True

    actor_num_nodes = 2
    actor_num_gpus_per_node = 8
    num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1
    rollout_endpoint_url = None
    sglang_server_concurrency = ROLLOUT_CONCURRENT_SAMPLES

    custom_rollout_request_hook_path = (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    custom_rollout_request_hook_args = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        "rollout_request_max_attempts": 1200,
        "rollout_request_retry_interval": 1.0,
    }
    miles_router_timeout = 1800

    update_weights_interval = 1
    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_buffer_size = 2 * 1024**3
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"

    tito_model = "qwen36"
    session_server_port = 30000
    session_server_workers = 8

    num_rollout = 500
    save_interval = 20
    save_hf = "hf_checkpoints/weight_v{rollout_id:06d}"
    rollout_batch_size = 32
    n_samples_per_prompt = 8
    global_batch_size = rollout_batch_size * n_samples_per_prompt
    rollout_temperature = 1.0
    rollout_top_p = 0.95
    rollout_top_k = 4096
    rollout_max_response_len = 8192
    max_seq_len = MAX_SEQ_LEN
    max_weight_staleness = None
    async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES
    async_data_buffer_capacity_factor = 2.0
    async_unused_samples_handler = "drop"
    keep_partial_groups_on_abort = True
    eval_interval = None

    use_rollout_routing_replay = True
    use_fault_tolerance = True
    rollout_health_check_first_wait = 600

    tensor_model_parallel_size = 2
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    distributed_timeout_minutes = 60
    moe_token_dispatcher_type = "alltoall"
    use_dynamic_batch_size = True
    use_dynamic_global_batch_size = True
    max_tokens_per_gpu = 4096
    log_probs_max_tokens_per_gpu = 4096
    log_probs_chunk_size = 8192
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    train_backend = "megatron"
    accumulate_allreduce_grads_in_fp32 = True
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    advantage_estimator = "grpo"
    skip_actor_forward_only = True
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
    wandb_group = "qwen3-6-35b-swebench-pro"
    disable_wandb_random_suffix = True
    use_prometheus = True
    prometheus_port = 9090
    prometheus_run_name = wandb_group

    environment = swebench_config.environment(
        sandbox_app="qwen3-6-35b-swebench-pro-sandbox",
        processes=AGENT_PROCESSES,
        threads_per_process=AGENT_THREADS_PER_PROCESS,
    )

    def prepare_data(self) -> None:
        prepare_swebench_pro(swebench_config.DATASET_PATH)


miles = _Miles(**swebench_config.arguments())
