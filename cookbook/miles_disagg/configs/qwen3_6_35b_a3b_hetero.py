"""Qwen3.6-35B-A3B RL on MiMo code tasks with a heterogeneous rollout fleet.

Each GPU type serves the precision and kernels that suit it: FP8 on H100 and H200,
NVFP4 (W4A16) on B200 and B300, BF16 on A100, RTX PRO 6000, H100 and H200. Every pool
therefore samples from its own approximation of the BF16 trainer's policy. Every
request still asks its replica for the latest published weights of its precision,
but no sample is dropped for its age.

This base holds the fleet, the trainer and the data shared by every recipe: sampling
with top-p 0.95 / top-k 64 whose support the trainer replays, and group-centered
advantages without std normalization. Each algorithm recipe inherits it and names
its own app, volume and W&B group:

- ``qwen3_6_35b_a3b_hetero_grpo``: naive GRPO.
- ``qwen3_6_35b_a3b_hetero_icepop``: IcePop, prompt-mean loss, frozen MoE router.
- ``qwen3_6_35b_a3b_hetero_score_centering``: score centering.
- ``qwen3_6_35b_a3b_hetero_score_centering_mis``: score centering with IcePop's weights.
"""

from cookbook.common.config import ModalConfig, RolloutPoolConfig
from cookbook.common.constants import CHECKPOINTS_PATH, DATA_PATH
from cookbook.miles_disagg import mimo_v2_6, swebench_config
from cookbook.miles_disagg.config import MilesConfig

SOURCE_MODEL = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
CHECKPOINT_ROOT = CHECKPOINTS_PATH / "qwen3-6-35b-a3b"
BF16_CHECKPOINT_PATH = CHECKPOINT_ROOT / "bf16"
FP8_CHECKPOINT_PATH = CHECKPOINT_ROOT / "fp8"
NVFP4_CHECKPOINT_PATH = CHECKPOINT_ROOT / "nvfp4-w4a16"
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
ROLLOUT_WEIGHT_VIEWS = {
    "bf16": BF16_CHECKPOINT_PATH,
    "fp8": FP8_CHECKPOINT_PATH,
    "nvfp4": NVFP4_CHECKPOINT_PATH,
}
TORCH_DIST_CHECKPOINT_PATH = CHECKPOINT_ROOT / "torch-dist-bf16-tp2-ep8"
SERVED_CHECKPOINT_FORMAT = "bf16"
CHECKPOINT_PREP_REQUIRES_GPU = True
UNPACK_FUSED_EXPERTS = True
LOCAL_CHECKPOINT_PATH = None
TRAINER_EXTRA_PIP_PACKAGES = swebench_config.TRAINER_PACKAGES
NVFP4_ENCODING_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
}
PREP_ENV = {
    "CONVERT_KEEP_PP1": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    **NVFP4_ENCODING_ENV,
    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
}
SGLANG_SERVER_ENV = {}

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"

DATASET_PATH = DATA_PATH / "mimo-v2-6-rl-oss"
SANDBOX_APP_NAME = "stitch-qwen36-hetero-sandbox"
MAX_SEQ_LEN = 262_144
AGENT_PROCESSES = 64
AGENT_THREADS_PER_PROCESS = 21
ROLLOUT_CONCURRENT_SAMPLES = AGENT_PROCESSES * AGENT_THREADS_PER_PROCESS
ROLLOUT_MIN_GPUS = {
    "A100-80GB": 8,
    "H100!": 8,
    "H200": 8,
    "B200": 8,
    "B300": 8,
    "RTX-PRO-6000": 8,
}
# Sessions per engine. Picked so per-request decode speed and turn latency
# match H200 FP8 at 32 (~90 tok/s per request) under MiMo-calibrated agent traffic.
ROLLOUT_TARGET_INPUTS = {
    "A100-80GB": 12,
    "H100!": 16,
    "H200": 32,
    "B200": 32,
    "B300": 48,
    "RTX-PRO-6000": 12,
}
ROLLOUT_MAX_RUNNING_REQUESTS = {
    "A100-80GB": 16,
    "H100!": 24,
    "H200": 32,
    "B200": 32,
    "B300": 64,
    "RTX-PRO-6000": 16,
}
# Host RAM per engine: measured steady state (CPU staging holds the canonical
# checkpoint plus rank images) + 50%. BF16 TP2 peaked at 150 GiB during a v1->v9
# catch-up. The 512 GiB limit leaves room to burst.
ROLLOUT_MEMORY_REQUEST_GIB = {
    ("bf16", 1): 216,
    ("bf16", 2): 224,
    ("fp8", 1): 120,
    ("nvfp4", 1): 88,
}
ROLLOUT_MEMORY_LIMIT_GIB = 512
# Weight staging uses ~5 cores even with 64 available, and catch-up time matched
# at 10 and 64 cores. Ten cores per engine lets every host class pack a full node.
ROLLOUT_CPU = 10.0


def _server_args(
    *,
    tp: int,
    attention_backend: str,
    max_running_requests: int,
    linear_attention_backend: str = "flashinfer",
    kv_cache_dtype: str = "fp8_e4m3",
    mem_fraction_static: float = 0.7,
    quantization: str | None = None,
    moe_runner_backend: str | None = None,
) -> dict[str, str]:
    args = {
        "--tp": str(tp),
        "--dtype": "bfloat16",
        "--load-format": "safetensors",
        "--weight-loader-drop-cache-after-load": "",
        "--dist-timeout": "3600",
        "--watchdog-timeout": "900",
        "--reasoning-parser": "qwen3",
        "--tool-call-parser": "qwen3_coder",
        "--context-length": str(MAX_SEQ_LEN),
        "--attention-backend": attention_backend,
        "--linear-attn-prefill-backend": linear_attention_backend,
        "--linear-attn-decode-backend": linear_attention_backend,
        "--mamba-ssm-dtype": "bfloat16",
        "--mamba-radix-cache-strategy": "extra_buffer",
        "--kv-cache-dtype": kv_cache_dtype,
        "--mem-fraction-static": str(mem_fraction_static),
        "--chunked-prefill-size": "8192",
        "--max-running-requests": str(max_running_requests),
        "--cuda-graph-max-bs-decode": str(max_running_requests),
        "--enable-metrics": "",
        "--enable-metrics-for-all-schedulers": "",
        "--decode-log-interval": "1000",
        "--log-level-http": "warning",
        "--weight-update-max-compile-group-gb": "2",
    }
    if quantization is not None:
        args["--quantization"] = quantization
    if moe_runner_backend is not None:
        args["--moe-runner-backend"] = moe_runner_backend
    return args


def _pool(
    *,
    name: str,
    gpu: str,
    weight_view: str,
    attention_backend: str,
    gpus_per_engine: int = 1,
    target_inputs: int | None = None,
    max_running_requests: int | None = None,
    linear_attention_backend: str = "flashinfer",
    kv_cache_dtype: str = "fp8_e4m3",
    mem_fraction_static: float = 0.7,
    quantization: str | None = None,
    moe_runner_backend: str | None = None,
    environment: dict[str, str] | None = None,
) -> RolloutPoolConfig:
    target_inputs = target_inputs or ROLLOUT_TARGET_INPUTS[gpu]
    max_running_requests = max_running_requests or ROLLOUT_MAX_RUNNING_REQUESTS[gpu]
    min_containers = ROLLOUT_MIN_GPUS[gpu] // gpus_per_engine
    return RolloutPoolConfig(
        name=name,
        gpu=gpu,
        gpus_per_engine=gpus_per_engine,
        target_inputs=target_inputs,
        sglang_args=_server_args(
            tp=gpus_per_engine,
            attention_backend=attention_backend,
            max_running_requests=max_running_requests,
            linear_attention_backend=linear_attention_backend,
            kv_cache_dtype=kv_cache_dtype,
            mem_fraction_static=mem_fraction_static,
            quantization=quantization,
            moe_runner_backend=moe_runner_backend,
        ),
        weight_view=weight_view,
        memory_mib=(
            ROLLOUT_MEMORY_REQUEST_GIB[weight_view, gpus_per_engine] * 1024,
            ROLLOUT_MEMORY_LIMIT_GIB * 1024,
        ),
        environment=environment or {},
        min_containers=min_containers,
        # A fixed fleet: scaling past the floor and back down kills live sessions,
        # and router weights are fixed at startup. Resize with update_autoscaler.
        max_containers=min_containers,
    )


modal = ModalConfig(
    gpu="B300",
    rollout_cpu=ROLLOUT_CPU,
    rollout_pools=(
        _pool(
            name="ServerH100FP8",
            gpu="H100!",
            weight_view="fp8",
            attention_backend="fa3",
        ),
        _pool(
            name="ServerH200FP8",
            gpu="H200",
            weight_view="fp8",
            attention_backend="fa3",
        ),
        _pool(
            name="ServerB200NVFP4W4A16",
            gpu="B200",
            weight_view="nvfp4",
            attention_backend="trtllm_mha",
            quantization="modelopt_fp4",
            moe_runner_backend="flashinfer_cutedsl",
            environment={"SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1"},
        ),
        _pool(
            name="ServerB300NVFP4W4A16",
            gpu="B300",
            weight_view="nvfp4",
            attention_backend="trtllm_mha",
            quantization="modelopt_fp4",
            moe_runner_backend="flashinfer_cutedsl",
            environment={"SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1"},
        ),
        # BF16 reads twice the FP8 expert bytes per decode step, so each BF16 GPU
        # takes half the routing load of its low-precision sibling. The 71.8 GB
        # checkpoint needs TP2 below 141 GB. Triton is the MoE runner validated for
        # BF16 staged updates.
        _pool(
            name="ServerA100BF16TP2",
            gpu="A100-80GB",
            gpus_per_engine=2,
            weight_view="bf16",
            attention_backend="flashinfer",
            # FlashInfer's linear attention needs SM90+.
            linear_attention_backend="triton",
            mem_fraction_static=0.65,
            moe_runner_backend="triton",
        ),
        _pool(
            name="ServerRTXPRO6000BF16TP2",
            gpu="RTX-PRO-6000",
            gpus_per_engine=2,
            weight_view="bf16",
            # trtllm_mha on SM120 rejects fp8 KV; flashinfer supports it.
            attention_backend="flashinfer",
            moe_runner_backend="triton",
        ),
        _pool(
            name="ServerH100BF16TP2",
            gpu="H100!",
            gpus_per_engine=2,
            weight_view="bf16",
            attention_backend="fa3",
            target_inputs=32,
            max_running_requests=48,
            moe_runner_backend="triton",
        ),
        _pool(
            name="ServerH200BF16",
            gpu="H200",
            weight_view="bf16",
            attention_backend="fa3",
            target_inputs=16,
            max_running_requests=24,
            moe_runner_backend="triton",
        ),
    ),
    trainer_cpu=(64.0, 256.0),
    trainer_memory_mib=(1_048_576, 3_145_728),
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


class HeteroMiles(MilesConfig):
    """Naive GRPO on the heterogeneous fleet; recipes override only what they change."""

    megatron_model_type = "qwen3.6-35B-A3B"
    async_mode = True

    hf_checkpoint = str(BF16_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    megatron_to_hf_mode = "raw"
    model_name = "qwen3_6"
    mtp_num_layers = 0
    hf_export_source_tensor_prefixes = ["model.visual.", "mtp."]
    transformer_impl = "transformer_engine"
    bf16 = True

    actor_num_nodes = 4
    actor_num_gpus_per_node = 8
    num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1
    rollout_endpoint_url = None
    sglang_server_concurrency = ROLLOUT_CONCURRENT_SAMPLES

    # Every request asks its replica for the latest published weights of the
    # replica's precision: the hook sends each view's latest version and the
    # router pins it as the request's minimum, so a replica still catching up
    # answers 409 and the request retries.
    custom_rollout_request_hook_path = (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    custom_rollout_request_hook_args = {"rollout_request_weight_version_mode": "min"}
    rollout_request_max_attempts = 1200
    rollout_request_retry_interval = 1.0
    miles_router_timeout = 3600

    update_weights_interval = 1
    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_buffer_size = 2 * 1024**3
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"

    tito_model = "qwen36"
    session_server_port = 30000
    session_server_workers = 64
    session_samples_timeout = 600

    num_rollout = 500
    save_interval = 10
    save_hf = "hf_checkpoints/weight_v{rollout_id:06d}"
    rollout_batch_size = 128
    n_samples_per_prompt = 8
    global_batch_size = rollout_batch_size * n_samples_per_prompt
    rollout_temperature = 1.0
    # The trainer replays this support; SGLang returns up to 4096 support tokens
    # per sampled token by default, well above top-k.
    rollout_top_p = 0.95
    rollout_top_k = 64
    rollout_max_response_len = 32_768
    max_seq_len = MAX_SEQ_LEN
    # Unbounded staleness: no sample is dropped for its age.
    max_weight_staleness = None
    async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES
    async_data_buffer_capacity_factor = 2.0
    # Convert the next batch while the three weight views publish.
    fully_async_drain_during_weight_update = True
    async_unused_samples_handler = "drop"
    keep_partial_groups_on_abort = True
    eval_interval = None

    tensor_model_parallel_size = 2
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    # Dynamic batching cannot split one sample, so CP alone bounds the tokens a
    # GPU holds for the longest episode: 262K / 4 = 64K. CP=2 OOMed at ~194K.
    context_parallel_size = 4
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    distributed_timeout_minutes = 60
    moe_token_dispatcher_type = "alltoall"
    use_dynamic_batch_size = True
    use_dynamic_global_batch_size = True
    max_tokens_per_gpu = 16_384
    log_probs_max_tokens_per_gpu = 16_384
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

    # One optimizer step per rollout batch against the trainer's own detached
    # log-probs: ratio 1, no off-policy correction.
    advantage_estimator = "grpo"
    disable_grpo_std_normalization = True
    calculate_per_token_loss = True
    skip_actor_forward_only = True
    use_rollout_logprobs = False
    use_tis = False
    use_rollout_routing_replay = False
    kl_coef = 0.0
    use_kl_loss = False
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
    disable_wandb_random_suffix = True
    use_prometheus = True
    prometheus_port = 9090

    environment = {
        **swebench_config.environment(
            sandbox_app=SANDBOX_APP_NAME,
            processes=AGENT_PROCESSES,
            threads_per_process=AGENT_THREADS_PER_PROCESS,
        ),
        **NVFP4_ENCODING_ENV,
        "OMP_NUM_THREADS": "1",
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        # Long episodes leave the B300 caching allocator fragmented.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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


def arguments() -> dict:
    values = swebench_config.arguments()
    values["prompt_data"] = f"{DATASET_PATH}/code.jsonl"
    return values
