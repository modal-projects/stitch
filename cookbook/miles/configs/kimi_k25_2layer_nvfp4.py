"""Kimi-K2.5 2-layer NVFP4 disaggregated — the FAST iteration proxy for K2.6.

CharyZeng/Kimi-K2.5-2layer is the identical arch class as K2.6
(KimiK25ForConditionalGeneration, model_type "kimi_k25": MLA + DeepSeek-MoE, 384
experts, topk 8, first_k_dense_replace 1, compressed-tensors INT4) but only 2
layers. So it exercises the EXACT K2.6-specific path — INT4->bf16 dequant, NVFP4
convert, the KimiK25 mbridge import converter, raw-mode build, torch_dist load,
te-precision NVFP4 QAT, disk-delta publish, NVFP4 serving — on a SINGLE node in
minutes, instead of 16x8 B200 + ~40 min init. Shake out the init/converter path
here, then run the full kimi_k2_6_nvfp4 once it's clean.

    EXPERIMENT_CONFIG=kimi_k25_2layer_nvfp4 \
      uv run --extra modal modal run --detach -m cookbook.miles.app::prepare_checkpoints
    EXPERIMENT_CONFIG=kimi_k25_2layer_nvfp4 \
      uv run --extra modal modal run --detach -m cookbook.miles.app::prepare_torch_dist
    EXPERIMENT_CONFIG=kimi_k25_2layer_nvfp4 \
      uv run --extra modal modal deploy --strategy recreate -m cookbook.miles.app
    # then ::launch_train
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import DATA_PATH, PREP_PATH
from cookbook.miles.config import MilesConfig


APP_NAME = "stitch-kimi-k25-2layer-nvfp4"
DELTA_VOLUME_NAME = "stitch-delta-kimi-k25-2layer-nvfp4"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

SOURCE_MODEL = "CharyZeng/Kimi-K2.5-2layer"  # INT4, KimiK25 arch, 2 layers
MODEL_TAG = "kimi-k25-2layer-nvfp4"

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_DEBUG_REQUESTS = True
# R3 routing-replay needs the dropless Megatron dispatch fix at startup.
MEGATRON_RUNTIME_PATCHES = [
    "/root/cookbook/miles/patches/megatron-r3-dispatch.patch",
]


SGLANG_SERVER_ARGS = {
    "--tool-call-parser": "kimi_k2",
    "--reasoning-parser": "kimi_k2",
    "--dist-timeout": "3600",
    "--kv-cache-dtype": "fp8_e4m3",
    "--attention-backend": "tokenspeed_mla",
    "--context-length": "8192",
    "--mem-fraction-static": "0.85",
    "--skip-server-warmup": "",
    "--enable-return-routed-experts": "",
}

modal = ModalConfig(
    gpu="B200",
    region="us",
    rollout_min_containers=1,
    proxy_regions=["us-west"],
    # 2-layer model fits 1 GPU for the torch_dist conversion (pp=world_size <= 2).
    torch_dist_prep_nodes=1,
    torch_dist_prep_gpus_per_node=1,
)


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/kimi-k25_2layer.sh"

    hf_checkpoint = f"{PREP_PATH}/{MODEL_TAG}/nvfp4"
    ref_load = f"{PREP_PATH}/{MODEL_TAG}/torch_dist"
    megatron_to_hf_mode = "raw"
    model_name = "kimi_k25"  # KimiK25 mbridge import + convert_kimi_k25_to_hf export

    # Disaggregated publish-only rollout on a single node (B200:4 trainer).
    actor_num_nodes = 1
    actor_num_gpus_per_node = 4
    num_gpus_per_node = 4
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1  # B200:1 pool (2-layer base is tiny)
    rollout_endpoint_url = None
    use_miles_router = True

    custom_rollout_request_hook_path = "cookbook.common.hooks.gated_rollout_request_hook"
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        "rollout_request_retry_attempts": 240,
        "rollout_request_retry_sleep": 1.0,
        "rollout_session_affinity_header": "Modal-Session-ID",
    }

    async_mode = True
    update_weights_interval = 1

    # NVFP4 QAT — same canonical recipe as K2.6.
    fp4_format = "e2m1"
    fp4_recipe = "nvfp4"
    fp4_param_gather = False
    te_precision_config_file = {
        "configs": {
            "nvfp4": {
                "transformer_engine_config_type": "TEQuantizationParams",
                "training_recipe": {"fp4_quantization_recipe": "nvfp4"},
            },
            "bf16": {"transformer_engine_config_type": "TEQuantizationParams", "training_recipe": {}},
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
            "default_bf16": {"type": "glob", "enabled": True, "pattern": "*", "config": "bf16"},
        },
    }
    # 2 layers: layer 0 is dense (FIRST_K_DENSE_REPLACE=1) -> bf16; layer 1 is MoE
    # -> NVFP4. Carving the end too would leave nothing NVFP4, so end=0.
    num_layers_at_start_in_bf16 = 1
    num_layers_at_end_in_bf16 = 0

    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"
    eval_interval = None

    # Tiny smoke: a couple of rollout/train steps to close the loop fast.
    num_rollout = 2
    save_interval = 10
    rollout_batch_size = 16
    rollout_max_response_len = 2048
    rollout_temperature = 0.8
    n_samples_per_prompt = 4
    global_batch_size = 32
    use_dynamic_global_batch_size = True
    sglang_server_concurrency = 32

    use_rollout_routing_replay = True

    # Single-node parallelism (B200:4): TP2/SP/PP1/CP1/EP4 (2 layers -> PP1).
    tensor_model_parallel_size = 2
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    expert_model_parallel_size = 4
    expert_tensor_parallel_size = 1
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 8192
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    no_check_for_nan_in_loss_and_grad = True

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
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.0
    use_tis = True

    environment = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NVSHMEM_DISABLE_NCCL": "1",
        "NCCL_TIMEOUT_MS": "360000000",
        "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
        "NVTE_NVFP4_DISABLE_RHT": "1",
        "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
        "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
        "NVTE_BACKWARD_OVERRIDE": "high_precision",
        "NVTE_USE_FAST_MATH": "0",
    }

    def prepare_data(self) -> None:
        from datasets import load_dataset

        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(2000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


miles = _Miles()
