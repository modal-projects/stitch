"""Qwen3.6-35B-A3B SWE-bench Pro training with humans& NVFP4 routed experts."""

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH
from cookbook.miles_disagg import nvfp4, swebench_config
from cookbook.miles_disagg.config import MilesConfig
from cookbook.miles_disagg.swebench_pro import prepare_swebench_pro

APP_NAME = "stitch-qwen3-6-35b-nvfp4"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-6-35b-nvfp4"
SOURCE_MODEL = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-bf16-unpacked-v4"
ROLLOUT_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-nvfp4-mse-v4"
TORCH_DIST_CHECKPOINT_PATH = (
    CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-torch-dist-tp2-ep8-v5"
)
SERVED_CHECKPOINT_FORMAT = "nvfp4"
CHECKPOINT_PREP_REQUIRES_GPU = True
UNPACK_FUSED_EXPERTS = True
LOCAL_CHECKPOINT_PATH = None
TRAINER_EXTRA_PIP_PACKAGES = swebench_config.TRAINER_PACKAGES
NVFP4_TRAINING_ENV, NVFP4_SERVING_ENV = nvfp4.environments(error_mode="MSE")
PREP_ENV = {
    **NVFP4_TRAINING_ENV,
    "CONVERT_KEEP_PP1": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
}
SGLANG_SERVER_ENV = dict(NVFP4_SERVING_ENV)
MAX_SEQ_LEN = 65_536
AGENT_PROCESSES = 16
AGENT_THREADS_PER_PROCESS = 16
ROLLOUT_CONCURRENT_SAMPLES = AGENT_PROCESSES * AGENT_THREADS_PER_PROCESS
SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"
SGLANG_SERVER_ARGS = {
    "--tp": "1",
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "900",
    "--quantization": "modelopt_fp4",
    "--reasoning-parser": "qwen3",
    "--tool-call-parser": "qwen3_coder",
    "--context-length": str(MAX_SEQ_LEN + 8),
    "--attention-backend": "trtllm_mha",
    "--linear-attn-prefill-backend": "flashinfer",
    "--linear-attn-decode-backend": "flashinfer",
    "--mamba-ssm-dtype": "bfloat16",
    "--mamba-radix-cache-strategy": "extra_buffer",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--kv-cache-dtype": "bfloat16",
    "--mem-fraction-static": "0.85",
    "--chunked-prefill-size": "8192",
    "--max-running-requests": "16",
    "--max-queued-requests": "4",
    "--cuda-graph-max-bs-decode": "16",
    "--enable-metrics": "",
    "--enable-metrics-for-all-schedulers": "",
    "--decode-log-interval": "1000",
    "--log-level-http": "warning",
    "--enable-return-routed-experts": "",
    "--sampling-mask-max-tokens": "8192",
}

modal = ModalConfig(
    gpu="B300",
    rollout_gpu="B200+",
    trainer_memory_mib=(262_144, 786_432),
    rollout_memory_mib=(262_144, 524_288),
    rollout_min_containers=16,
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
    # This recipe serves and trains only the target model; the source checkpoint's
    # optional MTP block is neither part of the rollout model nor the RL objective.
    mtp_num_layers = 0
    extra_high_precision_layers_hf = [".shared_expert."]
    extra_high_precision_layers_megatron = [
        ".shared_experts.linear_fc1",
        ".shared_experts.linear_fc2",
    ]
    transformer_impl = "transformer_engine"
    bf16 = True
    fp4_format = "e2m1"
    fp4_recipe = "nvfp4"
    first_last_layers_bf16 = True
    # Keep shared experts and the final 15% of 40 layers in BF16.
    num_layers_at_start_in_bf16 = 0
    num_layers_at_end_in_bf16 = 6
    te_precision_config_file = nvfp4.routed_expert_precision()

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
    miles_router_timeout = 300
    # Preserve source-owned modules that the target-only trainer does not instantiate.
    hf_export_source_tensor_prefixes = ["model.visual.", "mtp."]

    update_weights_interval = 1
    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_buffer_size = 2 * 1024**3
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"

    tito_model = "qwen35"
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
    rollout_top_k = 20
    rollout_max_response_len = 8192
    max_seq_len = MAX_SEQ_LEN
    max_weight_staleness = 6
    async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES
    # Buffer at most two completed learner batches while training is busy.
    async_data_buffer_capacity_factor = 2.0
    async_unused_samples_handler = "drop"
    eval_interval = None

    use_rollout_routing_replay = True
    use_fault_tolerance = True
    # Allow cold-start generation to settle before the first health probe.
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
    # FP32 A_log parameters require FP32 gradient buffers for uniform norm inputs.
    grad_reduce_in_bf16 = False
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    advantage_estimator = "grpo"
    # Reuse detached log-probs from the single training forward as the PPO
    # denominator instead of running a redundant forward-only actor pass.
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
    wandb_group = "qwen3-6-35b-nvfp4-swebench-pro"
    disable_wandb_random_suffix = True
    use_prometheus = True
    prometheus_port = 9090
    prometheus_run_name = wandb_group

    environment = {
        **swebench_config.environment(
            sandbox_app="qwen3-6-35b-nvfp4-sandbox",
            processes=AGENT_PROCESSES,
            threads_per_process=AGENT_THREADS_PER_PROCESS,
        ),
        **NVFP4_TRAINING_ENV,
    }

    def prepare_data(self) -> None:
        prepare_swebench_pro(swebench_config.DATASET_PATH)


miles = _Miles(**swebench_config.arguments())
