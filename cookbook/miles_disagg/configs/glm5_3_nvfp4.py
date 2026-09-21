"""Fully async GLM-5.3 NVFP4 GRPO on SWE-bench Pro."""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH, DRAFT_PATH
from cookbook.miles_disagg import nvfp4, swebench_config
from cookbook.miles_disagg.config import MilesConfig
from cookbook.miles_disagg.swebench_pro import prepare_swebench_pro

APP_NAME = "stitch-glm5-3-nvfp4"
EXPERIMENT_VOLUME_NAME = "stitch-miles-glm5-3-nvfp4"
LOCAL_CHECKPOINT_PATH = None
TRAINER_EXTRA_PIP_PACKAGES = swebench_config.TRAINER_PACKAGES
TRAINER_IMAGE_RUN_COMMANDS = (
    "uv pip install --system --break-system-packages flashinfer-python==0.6.15.post1",
    "uv pip install --system --break-system-packages --no-deps "
    "--index-url https://flashinfer.ai/whl "
    "flashinfer-cubin==0.6.15.post1",
    "uv pip install --system --break-system-packages --no-deps "
    "--index-url https://flashinfer.ai/whl/cu130 "
    "flashinfer-jit-cache==0.6.15.post1+cu130",
)
MEGATRON_RUNTIME_PATCHES = [
    "/root/cookbook/miles_disagg/patches/megatron-hdo-dp-reshardable-step.patch",
    "/root/cookbook/miles_disagg/patches/megatron-r3-dispatch.patch",
]

SOURCE_MODEL = "zai-org/GLM-5.3-BF16"
SOURCE_REVISION = "9d2398f478cab2de883137db3a36ad2c96205e24"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-9d2398f4-bf16"
ROLLOUT_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-9d2398f4-nvfp4-mse"
TORCH_DIST_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-9d2398f4-torch-dist-pp4-ep8"
DFLASH_VOLUME = "dflash_data"
DFLASH_CHECKPOINT_PATH = (
    DRAFT_PATH / "train/glm53_nvfp4_6l_swa_k3taps_rope2m_32x/trainer-0/draft-step-22500"
)
SERVED_CHECKPOINT_FORMAT = "nvfp4"
CHECKPOINT_PREP_REQUIRES_GPU = True
MATERIALIZE_BF16_MASTERS = False
USE_MODAL_TORCH_DIST_WRAPPER = True

TRAINER_NODES = 32
GPUS_PER_TRAINER_NODE = 8
ROLLOUT_GPUS_PER_ENGINE = 4
ROLLOUT_BATCH_SIZE = 64
N_SAMPLES_PER_PROMPT = 8
ROLLOUT_INPUTS_PER_ENGINE = 16
ROLLOUT_MAX_RUNNING_REQUESTS = 24
# Bound scheduler-poll and routing skew to the headroom above the routing target.
# Sustained excess load gets a retryable 503 instead of an unbounded engine queue.
ROLLOUT_MAX_QUEUED_REQUESTS = 4
ROLLOUT_ENGINES = 16
ROLLOUT_CONCURRENT_SAMPLES = ROLLOUT_ENGINES * ROLLOUT_INPUTS_PER_ENGINE
MAX_SEQ_LEN = 65_536
# SGLang's request boundary needs a small physical-context margin. Miles still
# truncates every trainable session to MAX_SEQ_LEN.
SGLANG_CONTEXT_LENGTH = MAX_SEQ_LEN + 8
# This metric changes exported NVFP4 bytes, so prep, training, and serving must agree.
NVFP4_TRAINING_ENV, NVFP4_SERVING_ENV = nvfp4.environments(error_mode="MSE")

PREP_ENV = dict(NVFP4_TRAINING_ENV)
SGLANG_SERVER_ENV = {
    **NVFP4_SERVING_ENV,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "NVSHMEM_DISABLE_NCCL": "1",
}

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"

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
    "--tp": "4",
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
    "--quantization": "modelopt_fp4",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--context-length": str(SGLANG_CONTEXT_LENGTH),
    "--attention-backend": "dsa",
    "--kv-cache-dtype": "fp8_e4m3",
    "--dsa-prefill-backend": "flashmla_sparse",
    "--dsa-decode-backend": "flashmla_kv",
    "--dsa-topk-backend": "flashinfer",
    "--page-size": "64",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--mem-fraction-static": "0.8",
    "--chunked-prefill-size": "16384",
    "--max-running-requests": str(ROLLOUT_MAX_RUNNING_REQUESTS),
    "--max-queued-requests": str(ROLLOUT_MAX_QUEUED_REQUESTS),
    "--schedule-conservativeness": "1.0",
    "--schedule-policy": "lpm",
    "--cuda-graph-config": (
        '{"decode":{"backend":"full","max_bs":24,'
        '"bs":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]}}'
    ),
    **DFLASH_SERVER_ARGS,
    "--enable-return-routed-experts": "",
}

modal = ModalConfig(
    gpu="B300",
    rollout_gpu="B300",
    # CPU-cache preparation is host-memory-bandwidth sensitive. For repeatable
    # latency, profile a B300 pool and pin its cloud and concrete region.
    region="us",
    trainer_memory_mib=(1024 * 1024, 3 * 1024 * 1024),
    # CPU mode retains the canonical checkpoint and TP rank images in memory.
    rollout_memory_mib=(1024 * 1024, 3 * 1024 * 1024),
    rollout_min_containers=ROLLOUT_ENGINES,
    rollout_target_inputs=ROLLOUT_INPUTS_PER_ENGINE,
    draft_volume=DFLASH_VOLUME,
    draft_volume_env="glm-bringup",
    routing_region="us-east",
    # CPU updates use ephemeral disk only for runtime scratch and spill.
    rollout_ephemeral_disk_mib=524_288,
    trainer_ephemeral_disk_mib=2_097_152,
    torch_dist_prep_nodes=4,
    torch_dist_prep_gpus_per_node=8,
    torch_dist_convert_extra_args=(
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 4 "
        "--expert-model-parallel-size 8 "
        "--decoder-first-pipeline-num-layers 18 "
        "--decoder-last-pipeline-num-layers 20"
    ),
    torch_dist_prep_ephemeral_disk_mib=2_097_152,
)


class _Miles(MilesConfig):
    # GLM-5.3 shares the GLM-5.2 architecture preset.
    megatron_model_type = "glm5.2-744B-A40B"
    train_backend = "megatron"
    async_mode = True

    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    megatron_to_hf_mode = "raw"
    model_name = "glm_moe_dsa"
    extra_high_precision_layers_hf = [".shared_experts."]
    extra_high_precision_layers_megatron = [
        ".shared_experts.linear_fc1",
        ".shared_experts.linear_fc2",
    ]

    actor_num_nodes = TRAINER_NODES
    actor_num_gpus_per_node = GPUS_PER_TRAINER_NODE
    num_gpus_per_node = GPUS_PER_TRAINER_NODE
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = ROLLOUT_GPUS_PER_ENGINE
    # An opaque endpoint has one Miles-side semaphore for the complete fleet.
    # A saturated engine's 503 retry holds its slot, feeding backpressure into
    # session generation instead of admitting another trajectory.
    sglang_server_concurrency = ROLLOUT_CONCURRENT_SAMPLES
    sglang_speculative_algorithm = "DFLASH"
    rollout_endpoint_url = None

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

    transformer_impl = "transformer_engine"
    bf16 = True
    fp4_format = "e2m1"
    fp4_recipe = "nvfp4"
    first_last_layers_bf16 = True
    # GLM-5.3 has three dense layers; keep those and the final 15% (12/78) in BF16.
    num_layers_at_start_in_bf16 = 3
    num_layers_at_end_in_bf16 = 12
    te_precision_config_file = nvfp4.routed_expert_precision()

    rollout_health_check_first_wait = 600
    tito_model = "glm47"
    session_server_port = [30000, 30064]
    session_server_startup_timeout_seconds = 600

    num_rollout = 500
    save_interval = 10
    save_hf = "hf_checkpoints/weight_v{rollout_id:06d}"
    rollout_batch_size = ROLLOUT_BATCH_SIZE
    n_samples_per_prompt = N_SAMPLES_PER_PROMPT
    global_batch_size = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT
    rollout_temperature = 1.0
    rollout_top_p = 0.95
    rollout_top_k = 4096
    rollout_max_response_len = 8192
    max_seq_len = MAX_SEQ_LEN
    use_dynamic_global_batch_size = True
    max_weight_staleness = 6
    async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES
    async_data_buffer_capacity_factor = 3.0
    async_unused_samples_handler = "drop"
    eval_interval = None

    use_rollout_routing_replay = True
    use_fault_tolerance = True

    # 256 GPUs: TP4 * PP4 * CP8 * DP2. Four balanced DSA-valid stages start on
    # layers 1, 19, 39, and 59; CP8 keeps long-context activations sharded.
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 4
    decoder_first_pipeline_num_layers = 18
    decoder_last_pipeline_num_layers = 20
    context_parallel_size = 8
    expert_model_parallel_size = 32
    expert_tensor_parallel_size = 1
    # Large clustered initialization and rollout-data materialization can exceed
    # the default process-group timeout.
    distributed_timeout_minutes = 60
    allgather_cp = True
    moe_token_dispatcher_type = "alltoall"
    use_dynamic_batch_size = True
    # CP8 shards the 64K sequence budget across eight ranks.
    max_tokens_per_gpu = 8192
    # Forward-only scoring retains no backward activations, so it can pack
    # twice the training budget without changing the backward memory bound.
    log_probs_max_tokens_per_gpu = 16384
    data_pad_size_multiplier = 1024
    log_probs_chunk_size = 16384
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    miles_dsa_topk_backend = "flashinfer"

    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = False
    overlap_cpu_optimizer_d2h_h2d = False
    use_precision_aware_optimizer = True

    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_rollout_logprobs = False
    skip_actor_forward_only = True
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

    use_wandb = True
    wandb_project = "fully-async-rl-modal"
    wandb_group = "glm5-3-nvfp4-swebench-pro"
    disable_wandb_random_suffix = True
    use_prometheus = True
    prometheus_port = 9090
    prometheus_run_name = "glm5-3-nvfp4-swebench-pro"

    environment = {
        **swebench_config.environment(
            sandbox_app="glm5-3-nvfp4-swebench-pro-sandbox",
            processes=48,
            boot_concurrency_per_process=10,
        ),
        "NCCL_NVLS_ENABLE": "1",
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
        **NVFP4_TRAINING_ENV,
    }

    def prepare_data(self) -> None:
        prepare_swebench_pro(swebench_config.DATASET_PATH)


miles = _Miles(**swebench_config.arguments())
