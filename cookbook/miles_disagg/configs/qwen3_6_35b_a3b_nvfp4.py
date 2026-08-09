"""Fully-async Qwen3.6 NVFP4 DFlash run on SWE-bench Pro.

This keeps the GLM-5.2 experiment's RL behavior—NVFP4 4/6 training and
serving, routed-expert replay, and exact top-p support replay—but uses the
single-GPU Qwen3.6 DFlash serving recipe. The run uses one 8-GPU trainer node,
sixteen rollout GPUs, and remains bounded to ten steps.
"""

from __future__ import annotations

from pathlib import Path

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH, DATA_PATH
from cookbook.miles_disagg.config import MilesConfig
from cookbook.miles_disagg.swebench_pro import prepare_swebench_pro

APP_NAME = "stitch-qwen3-6-35b-a3b-nvfp4"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-6-35b-a3b-nvfp4"
MILES_REPO_REF = "b1020b5961657ef1bb8c9f56bda49bc12899fa57"
QWEN36_NVFP4_DFLASH_MILES_PATCH = (
    Path(__file__).resolve().parents[1] / "patches" / "qwen3_6_nvfp4_dflash.patch"
)
MILES_SOURCE_PATCHES = (QWEN36_NVFP4_DFLASH_MILES_PATCH,)
LOCAL_CHECKPOINT_PATH = None
TRAINER_EXTRA_PIP_PACKAGES = (
    "harbor[modal,huggingface]==0.20.0",
    "mini-swe-agent==2.4.5",
    "swebench==4.1.0",
    "modal==1.5.3",
)
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

SOURCE_MODEL = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
DFLASH_MODEL = "modal-labs/Qwen3.6-35B-A3B-DFlash"
DFLASH_REVISION = "45197228fd8152743a4566620c7aa4014d35f773"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-6-35b-a3b-bf16"
ROLLOUT_CHECKPOINT_PATH = (
    CHECKPOINTS_PATH / "qwen3-6-35b-a3b-nvfp4-routed-bf16-alog-fp32"
)
TORCH_DIST_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-6-35b-a3b-torch-dist"
SWEBENCH_PRO_PATH = DATA_PATH / "swebench-pro"
SERVED_CHECKPOINT_FORMAT = "nvfp4"
CHECKPOINT_PREP_REQUIRES_GPU = True
MATERIALIZE_BF16_MASTERS = False

TRAINER_NODES = 1
GPUS_PER_TRAINER_NODE = 8
ROLLOUT_GPUS_PER_ENGINE = 1
ROLLOUT_BATCH_SIZE = 32
N_SAMPLES_PER_PROMPT = 8
GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT
ROLLOUT_INPUTS_PER_ENGINE = 8
ROLLOUT_MAX_RUNNING_REQUESTS = 32
ROLLOUT_MAX_QUEUED_REQUESTS = 8
ROLLOUT_ENGINES = 16
ROLLOUT_CONCURRENT_SAMPLES = ROLLOUT_ENGINES * ROLLOUT_INPUTS_PER_ENGINE
MAX_SEQ_LEN = 16_384
SGLANG_CONTEXT_LENGTH = MAX_SEQ_LEN + 8

# This metric changes the packed weights, so prep, live export, and serving
# must use one contract. It deliberately matches the GLM-5.2 branch recipe.
NVFP4_4OVER6_ERR_MODE = "MSE"
NVFP4_TRAINING_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
    "NVTE_NVFP4_4OVER6_ERR_MODE": NVFP4_4OVER6_ERR_MODE,
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
}
NVFP4_SERVING_ENV = {
    "FLASHINFER_NVFP4_4OVER6": "1",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": NVFP4_4OVER6_ERR_MODE,
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
}
# Miles' offline converter otherwise promotes PP1 to WORLD_SIZE. Keep PP1 so
# the one-node conversion can use all eight ranks for expert parallelism.
PREP_ENV = {
    **NVFP4_TRAINING_ENV,
    "CONVERT_KEEP_PP1": "1",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
}
SGLANG_SERVER_ENV = {
    **NVFP4_SERVING_ENV,
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_RELOAD_LOAD_PLAN": "1",
    "SGLANG_SANITIZE_NAN_LOGITS": "true",
}

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"

SGLANG_SERVER_ARGS = {
    # Loading and elastic weight refits.
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--enable-cpu-weight-cache": "",
    "--cpu-weight-cache-max-compile-group-gb": "8",
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
    # Qwen3.6 and its hybrid attention stack.
    "--quantization": "modelopt_fp4",
    "--reasoning-parser": "qwen3",
    "--tool-call-parser": "qwen3_coder",
    "--context-length": str(SGLANG_CONTEXT_LENGTH),
    "--attention-backend": "trtllm_mha",
    "--linear-attn-prefill-backend": "flashinfer",
    "--linear-attn-decode-backend": "flashinfer",
    "--mamba-ssm-dtype": "bfloat16",
    "--mamba-radix-cache-strategy": "extra_buffer",
    # Shared expert remains BF16; routed experts use NVFP4 and emit R3 metadata.
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--enable-return-routed-experts": "",
    # The public draft's single-GPU DFlash recipe.
    "--speculative-algorithm": "DFLASH",
    "--speculative-draft-model-path": DFLASH_MODEL,
    "--speculative-draft-model-revision": DFLASH_REVISION,
    "--speculative-draft-model-quantization": "unquant",
    "--speculative-dflash-block-size": "8",
    "--speculative-draft-attention-backend": "fa4",
    # Bounded rollout scheduling and graph capture.
    "--mem-fraction-static": "0.65",
    "--chunked-prefill-size": "8192",
    "--max-running-requests": str(ROLLOUT_MAX_RUNNING_REQUESTS),
    "--max-queued-requests": str(ROLLOUT_MAX_QUEUED_REQUESTS),
    "--schedule-conservativeness": "1.0",
    "--schedule-policy": "lpm",
    "--cuda-graph-max-bs-decode": "32",
    "--cuda-graph-backend-prefill": "tc_piecewise",
    "--enable-flashinfer-allreduce-fusion": "",
    "--enable-metrics": "",
    "--enable-metrics-for-all-schedulers": "",
    "--decode-log-interval": "1000",
    "--log-level-http": "warning",
}

modal = ModalConfig(
    gpu="B200",
    rollout_gpu="B200",
    region="us",
    trainer_memory_mib=(512 * 1024, 1024 * 1024),
    rollout_memory_mib=(128 * 1024, 512 * 1024),
    rollout_min_containers=ROLLOUT_ENGINES,
    rollout_max_containers=ROLLOUT_ENGINES,
    rollout_target_inputs=ROLLOUT_INPUTS_PER_ENGINE,
    routing_region="us-west",
    rollout_ephemeral_disk_mib=524_288,
    trainer_ephemeral_disk_mib=524_288,
    torch_dist_prep_nodes=1,
    torch_dist_prep_gpus_per_node=8,
    torch_dist_convert_extra_args=(
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--expert-model-parallel-size 8"
    ),
    torch_dist_prep_ephemeral_disk_mib=524_288,
)


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/qwen3.6-35B-A3B.sh"
    async_mode = True

    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    megatron_to_hf_mode = "raw"
    model_name = "qwen3_6"
    extra_high_precision_layers_hf = [".shared_expert."]
    extra_high_precision_layers_megatron = [
        ".shared_experts.linear_fc1",
        ".shared_experts.linear_fc2",
    ]

    actor_num_nodes = TRAINER_NODES
    actor_num_gpus_per_node = GPUS_PER_TRAINER_NODE
    num_gpus_per_node = GPUS_PER_TRAINER_NODE
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = ROLLOUT_GPUS_PER_ENGINE
    sglang_server_concurrency = ROLLOUT_CONCURRENT_SAMPLES
    sglang_speculative_algorithm = "DFLASH"
    sglang_disaggregation_sampling_mask_max_tokens = 4096
    rollout_endpoint_url = None

    custom_rollout_request_hook_path = (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        "rollout_request_retry_attempts": 1200,
        "rollout_request_retry_sleep": 1.0,
        "rollout_session_affinity_header": "Modal-Session-ID",
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
    fp4_param_gather = False
    first_last_layers_bf16 = True
    num_layers_at_start_in_bf16 = 0
    # Keep the final 15% (6/40) in BF16, plus the shared expert above.
    num_layers_at_end_in_bf16 = 6
    te_precision_config_file = {
        "configs": {
            "nvfp4": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {"fp4_quantization_recipe": "nvfp4"},
            },
            "bf16": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {},
            },
        },
        "matchers": {
            "routed_experts_fc1_nvfp4": {
                "type": "glob",
                "enabled": True,
                "pattern": "*.mlp.experts.linear_fc1",
                "config": "nvfp4",
            },
            "routed_experts_fc2_nvfp4": {
                "type": "glob",
                "enabled": True,
                "pattern": "*.mlp.experts.linear_fc2",
                "config": "nvfp4",
            },
            "default_bf16": {
                "type": "glob",
                "enabled": True,
                "pattern": "*",
                "config": "bf16",
            },
        },
    }

    prompt_data = f"{SWEBENCH_PRO_PATH}/test.jsonl"
    input_key = "prompt"
    metadata_key = "metadata"
    apply_chat_template = False
    rollout_shuffle = True
    balance_data = True

    fully_async = True
    rollout_sample_completion_backfill = True
    custom_rollout_log_function_path = "modal_swe_metrics.log_rollout_data"
    custom_generate_function_path = (
        "miles.rollout.generate_hub.agentic_tool_call.generate"
    )
    custom_agent_function_path = "modal_swe_agent_function.run"
    custom_rm_path = "modal_swe_agent_function.reward_func"
    tito_model = "qwen35"
    use_session_server = True
    session_server_port = [30000, 30016]
    session_server_startup_timeout_seconds = 180
    tito_session_mismatch_sample_rate = 0.125

    num_rollout = 10
    save_interval = None
    rollout_batch_size = ROLLOUT_BATCH_SIZE
    n_samples_per_prompt = N_SAMPLES_PER_PROMPT
    global_batch_size = GLOBAL_BATCH_SIZE
    rollout_temperature = 1.0
    rollout_top_p = 0.95
    rollout_max_response_len = 8192
    max_seq_len = MAX_SEQ_LEN
    use_dynamic_global_batch_size = True
    max_weight_staleness = 2
    async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES
    eval_interval = None

    use_rollout_routing_replay = True
    use_fault_tolerance = True

    # TP1/PP1/CP1 with one trainer GPU for each of the eight expert shards.
    tensor_model_parallel_size = 1
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    distributed_timeout_minutes = 60
    moe_token_dispatcher_type = "alltoall"
    use_dynamic_batch_size = True
    max_tokens_per_gpu = MAX_SEQ_LEN
    log_probs_max_tokens_per_gpu = MAX_SEQ_LEN
    data_pad_size_multiplier = 1024
    log_probs_chunk_size = MAX_SEQ_LEN
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    attention_backend = "flash"
    train_backend = "megatron"

    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
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

    use_wandb = True
    wandb_project = "fully-async-rl-modal"
    wandb_group = "qwen3-6-35b-a3b-nvfp4-swebench-pro-smoke"
    disable_wandb_random_suffix = True
    use_prometheus = True
    prometheus_port = 9090
    prometheus_run_name = "qwen3-6-35b-a3b-nvfp4-swebench-pro-smoke"

    environment = {
        "PYTHONPATH": (
            "/root/Megatron-LM:/root/miles:/root/miles/examples/swe-agent:"
            "/root/miles/examples/experimental/modal-swe"
        ),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NVSHMEM_DISABLE_NCCL": "1",
        "RAY_health_check_timeout_ms": "60000",
        "RAY_health_check_failure_threshold": "30",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_MODEL_NAME": "model",
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
        "LITELLM_LOG": "ERROR",
        "MODAL_SWE_TASKS_DIR": f"{SWEBENCH_PRO_PATH}/tasks",
        # The sandbox runtime is model-independent; reuse the existing deployed app.
        "MODAL_SWE_SANDBOX_APP": "glm5-2-nvfp4-swebench-pro-sandbox",
        "MODAL_SWE_MAX_STEPS": "128",
        "MODAL_SWE_EPISODE_TIMEOUT": "7200",
        "MODAL_SWE_MODEL_REQUEST_TIMEOUT": "1800",
        "MODAL_SWE_EXEC_TIMEOUT": "120",
        "MODAL_SWE_OUTPUT_HARD_LIMIT_BYTES": str(16 * 1024 * 1024),
        "MODAL_SWE_SETUP_TIMEOUT": "240",
        "MODAL_SWE_VERIFY_TIMEOUT": "3600",
        "MODAL_SWE_INJECT_PYTEST_REPORTER": "0",
        "MODAL_SWE_CPUS": "2",
        "MODAL_SWE_MEMORY_MIB": "16384",
        "MODAL_SWE_AGENT_PROCESSES": "8",
        "MODAL_SWE_AGENT_THREADS_PER_PROCESS": "8",
        "MODAL_SWE_SANDBOX_BOOT_CONCURRENCY_PER_PROCESS": "2",
        **NVFP4_TRAINING_ENV,
    }

    def prepare_data(self) -> None:
        prepare_swebench_pro(SWEBENCH_PRO_PATH)


miles = _Miles()
