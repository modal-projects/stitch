"""Fully async GLM-5.2 NVFP4 GRPO on SWE-bench Pro."""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH, DATA_PATH
from cookbook.miles_disagg.config import MilesConfig
from cookbook.miles_disagg.swebench_pro import (
    DATASET_REVISION,
    prepare_swebench_pro,
)

APP_NAME = "stitch-glm5-2-nvfp4"
EXPERIMENT_VOLUME_NAME = "stitch-miles-glm5-2-nvfp4"
# This experiment layers Stitch integration onto Miles' fully-async SWE runtime.
# Other cookbook experiments keep the standard Miles pin in trainer_image.py.
MILES_REPO_REF = "7cdfcf78c8f7ab3f2111dafe9881aa99b716a0c5"
LOCAL_CHECKPOINT_PATH = None
TRAINER_EXTRA_PIP_PACKAGES = (
    "harbor[modal,huggingface]==0.20.0",
    "mini-swe-agent==2.4.5",
    "swebench==4.1.0",
    "modal==1.5.1",
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

SOURCE_MODEL = "zai-org/GLM-5.2"
SOURCE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-2-bf16"
ROLLOUT_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-2-nvfp4"
TORCH_DIST_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-2-torch-dist"
SWEBENCH_PRO_PATH = DATA_PATH / "datasets" / "swebench-pro" / DATASET_REVISION
SERVED_CHECKPOINT_FORMAT = "nvfp4"
CHECKPOINT_PREP_REQUIRES_GPU = True
MATERIALIZE_BF16_MASTERS = False
USE_MODAL_TORCH_DIST_WRAPPER = True

TRAINER_NODES = 16
GPUS_PER_TRAINER_NODE = 8
ROLLOUT_GPUS_PER_ENGINE = 4
ROLLOUT_TO_TRAINER_GPU_RATIO = 1
ROLLOUT_INPUTS_PER_ENGINE = 24
ROLLOUT_ENGINES = (
    TRAINER_NODES
    * GPUS_PER_TRAINER_NODE
    * ROLLOUT_TO_TRAINER_GPU_RATIO
    // ROLLOUT_GPUS_PER_ENGINE
)
ROLLOUT_CONCURRENT_SAMPLES = ROLLOUT_ENGINES * ROLLOUT_INPUTS_PER_ENGINE
MAX_SEQ_LEN = 65_536
# SGLang's request boundary needs a small physical-context margin. Miles still
# truncates every trainable session to MAX_SEQ_LEN.
SGLANG_CONTEXT_LENGTH = MAX_SEQ_LEN + 8

NVFP4_TRAINING_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
}
NVFP4_SERVING_ENV = {
    "FLASHINFER_NVFP4_4OVER6": "1",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
}
DSA_TOPK_ENV = {"SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large"}
PREP_ENV = NVFP4_TRAINING_ENV
SGLANG_SERVER_ENV = {
    **NVFP4_SERVING_ENV,
    **DSA_TOPK_ENV,
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
    "SGLANG_SANITIZE_NAN_LOGITS": "true",
}

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"

SGLANG_SERVER_ARGS = {
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--enable-cpu-weight-cache": "",
    "--cpu-weight-cache-max-compile-group-gb": "8",
    "--weight-loader-drop-cache-after-load": "",
    "--quantization": "modelopt_fp4",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--attention-backend": "dsa",
    "--dsa-decode-backend": "flashmla_kv",
    "--dsa-prefill-backend": "flashmla_sparse",
    "--dsa-topk-backend": "flashinfer",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--dist-timeout": "3600",
    "--kv-cache-dtype": "fp8_e4m3",
    "--page-size": "64",
    "--context-length": str(SGLANG_CONTEXT_LENGTH),
    "--mem-fraction-static": "0.8",
    "--chunked-prefill-size": "8192",
    "--schedule-conservativeness": "0.5",
    "--schedule-policy": "lpm",
    "--skip-server-warmup": "",
    "--enable-return-routed-experts": "",
}

modal = ModalConfig(
    gpu="B300",
    region="us",
    trainer_memory_mib=(1024 * 1024, 3 * 1024 * 1024),
    # CPU mode retains both the canonical checkpoint and TP rank images.
    rollout_memory_mib=(1024 * 1024, 3 * 1024 * 1024),
    rollout_min_containers=ROLLOUT_ENGINES,
    rollout_target_inputs=ROLLOUT_INPUTS_PER_ENGINE,
    routing_region="us-west",
    # cpu updates stage nothing local; disk is runtime + spill only.
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
    miles_model_script = "scripts/models/glm5.2-744B-A40B.sh"
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
    sglang_server_concurrency = ROLLOUT_CONCURRENT_SAMPLES
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
    first_last_layers_bf16 = True
    # GLM-5.2 has three dense layers; keep those and the final 15% (12/78) in BF16.
    num_layers_at_start_in_bf16 = 3
    num_layers_at_end_in_bf16 = 12
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
    tito_model = "glm47"
    use_session_server = True
    session_server_port = [30000, 30064]
    session_server_startup_timeout_seconds = 180
    tito_session_mismatch_sample_rate = 0.0625

    num_rollout = 300
    save_interval = 10
    rollout_batch_size = 32
    n_samples_per_prompt = 8
    global_batch_size = 256
    rollout_temperature = 1.0
    rollout_top_p = 1.0
    rollout_max_response_len = 8192
    max_seq_len = MAX_SEQ_LEN
    use_dynamic_global_batch_size = True
    max_weight_staleness = 6
    async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES

    use_rollout_routing_replay = True
    use_fault_tolerance = True

    # 128 GPUs: TP4 * PP8 * CP4, with one data-parallel replica. The uneven
    # split keeps every DSA pipeline stage on a layer that computes its index.
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 8
    decoder_first_pipeline_num_layers = 14
    decoder_last_pipeline_num_layers = 16
    context_parallel_size = 4
    expert_model_parallel_size = 16
    expert_tensor_parallel_size = 1
    allgather_cp = True
    moe_enable_deepep = True
    moe_token_dispatcher_type = "flex"
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 16384
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
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_rollout_logprobs = True
    get_mismatch_metrics = True
    custom_tis_function_path = (
        "examples.infra_features.train_infer_mismatch_helper.mis."
        "compute_mis_weights_with_cp"
    )
    entropy_coef = 0.0

    use_wandb = True
    wandb_project = "fully-async-rl-modal"
    wandb_group = "glm5-2-nvfp4-swebench-pro"
    disable_wandb_random_suffix = True
    use_prometheus = True
    prometheus_port = 9090
    prometheus_run_name = "glm5-2-nvfp4-swebench-pro"

    environment = {
        "PYTHONPATH": (
            "/root/Megatron-LM:/root/miles:/root/miles/examples/swe-agent:"
            "/root/miles/examples/experimental/modal-swe"
        ),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        # GLM-5 uses interleaved, not NeoX-style, rotary pairs in the DSA indexer.
        "INDEXER_ROPE_NEOX_STYLE": "0",
        "RAY_health_check_timeout_ms": "60000",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "AGENT_MODEL_NAME": "model",
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
        "LITELLM_LOG": "ERROR",
        "MODAL_SWE_TASKS_DIR": f"{SWEBENCH_PRO_PATH}/tasks",
        "MODAL_SWE_SANDBOX_APP": "glm5-2-nvfp4-swebench-pro-sandbox",
        "MODAL_SWE_MAX_STEPS": "256",
        "MODAL_SWE_EPISODE_TIMEOUT": "7200",
        "MODAL_SWE_MODEL_REQUEST_TIMEOUT": "1800",
        "MODAL_SWE_EXEC_TIMEOUT": "120",
        "MODAL_SWE_OUTPUT_HARD_LIMIT_BYTES": str(16 * 1024 * 1024),
        "MODAL_SWE_SETUP_TIMEOUT": "600",
        "MODAL_SWE_VERIFY_TIMEOUT": "3600",
        "MODAL_SWE_INJECT_PYTEST_REPORTER": "0",
        "MODAL_SWE_CPUS": "2",
        "MODAL_SWE_MEMORY_MIB": "16384",
        "MODAL_SWE_AGENT_PROCESSES": "48",
        "MODAL_SWE_AGENT_THREADS_PER_PROCESS": "16",
        **NVFP4_TRAINING_ENV,
        **DSA_TOPK_ENV,
    }

    def prepare_data(self) -> None:
        prepare_swebench_pro(SWEBENCH_PRO_PATH)


miles = _Miles()
