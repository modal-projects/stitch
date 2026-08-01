"""Qwen3-30B-A3B GRPO on Modal, disaggregated — the humans& NVFP4 "4-bitter" recipe.

The full recipe from https://humansand.ai/blog/nvfp4-rl (miles PR #1261), on the blog's
own model/dataset (Qwen3-30B-A3B, DAPO-Math-17k), end to end on B200:

  1. Per-token row-scaled NVFP4 forward on the MoE experts
     (NVTE_NVFP4_ROW_SCALED_ACTIVATION=1, RHT/stochastic-rounding/2D off).
  2. Dequantized backward — the BF16 backward differentiates DQ(Q(w)), the same quantized
     function the forward evaluated (NVTE_BACKWARD_OVERRIDE=dequantized; TE 2.17 + the
     TE#3141 save-original-input fix from the shared Miles image).
  3. Four-over-six adaptive block scaling on weights AND activations
     (NVTE_NVFP4_4OVER6=all / FLASHINFER_NVFP4_4OVER6=1), the trainer and sampler sides of
     the env contract pinned to identical conventions: MAE error mode, e4m3-max 256, fast
     math off.
  4. Selective precision: the last 15% of layers (7 of 48) stay BF16; Qwen3 MoE has no
     shared expert, so the layer carve-out is the whole of this axis.

WHY THE WEIGHT BYTES ARE EXACT BY CONSTRUCTION
----------------------------------------------
miles' export and tools/convert_hf_to_nvfp4.py both use the same TE-direct NVFP4Quantizer
(PR #1261), so served packing == export packing and the delta baseline aligns — the
FlashInfer weight quantizer is never on the served path. PREP_ENV pins the conversion to
the trainer's exact NVTE_* settings; a drift there fails loud on the first delta apply.
FlashInfer's runtime role is activation quantization only, where the FLASHINFER_* mirror
keeps rollout logprobs near the trainer's (train_rollout_logprob_abs_diff ~= 0.044 in the
validated smoke; PR #1261 reports 0.031).

STACK
-----
The shared Miles image and repository pin contain PR #1261, TransformerEngine 2.17,
and its dequantized-backward fix. The shared SGLang image contains FlashInfer 0.6.14,
which includes the 4/6 error-domain fix. This experiment adds no dependency overrides;
it only supplies the matching NVTE_* and FLASHINFER_* runtime settings.

GPU footprint: trainer 1x8 B200 + rollout pool 1-3x1 B200 (<= 11 concurrent).

Launch: EXPERIMENT_CONFIG=qwen3_30b_a3b_nvfp4_46 uv run --extra modal python -m cookbook.miles_disagg.launch
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH, DATA_PATH
from cookbook.miles_disagg.config import MilesConfig

APP_NAME = "stitch-qwen3-30b-nvfp4-46"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-30b-nvfp4-46"
LOCAL_CHECKPOINT_PATH = None

SOURCE_MODEL = "Qwen/Qwen3-30B-A3B"  # bf16 -> masters ARE the download
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-30b-a3b-bf16"
ROLLOUT_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-30b-a3b-nvfp4-46"

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"
# R3 routing-replay needs the dropless Megatron dispatch fix at startup.
MEGATRON_RUNTIME_PATCHES = [
    "/root/cookbook/miles_disagg/patches/megatron-r3-dispatch.patch",
]

# ── Trainer/sampler 4/6 + NVFP4 quantizer contract ────────────────────────────
# Both sides must agree on: 4/6 on, MAE candidate-error mode, e4m3-max 256 under 4/6,
# fast math off. The NVTE_* side drives TE (training, export, and the served-base
# conversion via PREP_ENV); the FLASHINFER_* side drives the pool's activation
# quantization kernels.
_NVTE_QUANT_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MAE",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
}
_FLASHINFER_QUANT_ENV = {
    "FLASHINFER_NVFP4_4OVER6": "1",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MAE",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
}

# prepare_checkpoints must quantize the served base under the trainer's exact settings.
PREP_ENV = dict(_NVTE_QUANT_ENV)

SGLANG_SERVER_ENV = {
    **_FLASHINFER_QUANT_ENV,
    "SGLANG_ENABLE_RELOAD_LOAD_PLAN": "1",
}

# Qwen3 MoE is GQA (no MLA): trtllm_mha attention + the routed FlashInfer TRTLLM MoE
# runner (emits per-token routed experts for R3 replay). NVFP4 comes from the served
# checkpoint's own quant config — no --quantization flag.
SGLANG_SERVER_ARGS = {
    "--weight-loader-prefetch-checkpoints": "",
    "--weight-loader-prefetch-num-threads": "8",
    "--enable-cpu-weight-cache": "",
    "--attention-backend": "trtllm_mha",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--kv-cache-dtype": "bfloat16",
    "--context-length": "16384",
    "--mem-fraction-static": "0.8",
    "--chunked-prefill-size": "4096",
    "--skip-server-warmup": "",
    "--enable-return-routed-experts": "",
}

modal = ModalConfig(
    gpu="B200",
    region="us",
    # Warm floor of 1 (the pool must be UP before the trainer sends rollouts); cap at 3
    # to bound the footprint at trainer 8 + pool 3 = 11 concurrent B200.
    rollout_min_containers=1,
    rollout_max_containers=3,
    # Per-container autoscaler target, well below the trainer's client concurrency: a
    # rollout wave (rollout_batch_size x n_samples_per_prompt = 256) must register as
    # queue pressure so Flash scales OUT to the container cap instead of one engine
    # absorbing the whole wave at its concurrency ceiling.
    rollout_target_inputs=24,
    # cpu persist ≈ 2x the 23.5 GiB canonical checkpoint; request covers staging + serving baseline.
    rollout_memory_mib=(128 * 1024, 512 * 1024),
    proxy_regions=["us-west"],
)


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/qwen3-30B-A3B.sh"

    # Bridge mode: ref_load is the bf16 HF masters directly (no torch_dist prep).
    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(BF16_CHECKPOINT_PATH)
    megatron_to_hf_mode = "bridge"
    model_name = "qwen3moe"  # megatron_to_hf export dispatch

    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    num_gpus_per_node = 8
    colocate = False  # disaggregated: rollouts live on the pool, not the trainer
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1  # 30B NVFP4 is ~17 GB packed; 1 B200 per engine
    rollout_endpoint_url = None
    use_miles_router = True

    custom_rollout_request_hook_path = (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        "rollout_request_retry_attempts": 240,
        "rollout_request_retry_sleep": 1.0,
        "rollout_session_affinity_header": "Modal-Session-ID",
    }

    async_mode = True
    update_weights_interval = 1

    # ── NVFP4 QAT: the recipe's trainer precision config ──
    fp4_format = "e2m1"
    fp4_recipe = "nvfp4"
    fp4_param_gather = False
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
    # Selective precision: last 15% of the 48 layers (7) stay BF16 (the blog's "spend
    # precision where it matters"; Qwen3 MoE has no shared expert). All layers are MoE
    # (FIRST_K_DENSE_REPLACE=0), so no start carve-out.
    num_layers_at_start_in_bf16 = 0
    num_layers_at_end_in_bf16 = 7

    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"

    # Data: DAPO-Math-17k — the blog's dataset.
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"
    eval_interval = None

    # Rollout (bring-up smoke length; scale num_rollout for a real run).
    num_rollout = 20
    # None (not just > num_rollout): miles still writes a final megatron save at the last
    # rollout, and the ~120 GB torch_dist save blows the trainer's default ephemeral disk
    # (the volume write cache lives there). None makes the app null out --load/--save so
    # no save path exists. Set a real save_interval (plus trainer_ephemeral_disk_mib) for
    # long runs.
    save_interval = None
    rollout_batch_size = 32
    # Long enough that math reasoning traces mostly terminate rather than truncate
    # (4096 clipped most responses, starving the reward signal).
    rollout_max_response_len = 12288
    rollout_temperature = 0.8
    n_samples_per_prompt = 8
    global_batch_size = 128
    use_dynamic_global_batch_size = True
    # Trainer-side client concurrency to the pool gateway: high enough to drive the
    # scaled-out pool (3 x 24 targets), not just one engine.
    sglang_server_concurrency = 128

    use_rollout_routing_replay = True

    # TE's cuDNN fused-attention backward fails on the dynamic sequence shapes emitted
    # by this recipe (CUDNN_STATUS_BAD_PARAM in the reshape operation). Attention stays
    # in BF16, so FlashAttention does not change the NVFP4 expert-layer recipe.
    attention_backend = "flash"

    # Trainer parallelism: the upstream Qwen3-30B recipe (TP4/SP/PP1/CP1) with EP = the
    # full node under no_colocate, as run_qwen3_30b_a3b.py does.
    tensor_model_parallel_size = 4
    sequence_parallel = True
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    expert_model_parallel_size = 8
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

    # Algorithm (GRPO + truncated importance sampling, per the blog's baseline).
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
        **_NVTE_QUANT_ENV,
    }

    def prepare_data(self) -> None:
        from datasets import load_dataset

        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


miles = _Miles()
