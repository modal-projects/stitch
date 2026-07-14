"""GLM-5.2 GRPO on Modal, disaggregated, native-NVFP4 end to end.

Modeled directly on ``kimi_k2_6_nvfp4`` — same machinery: NVFP4 (W4A4, experts-only)
QAT trainer on Blackwell, an elastic Flash B200 rollout pool serving the NVFP4 base,
and byte-exact XOR disk-deltas published against that base. sglang v0.5.15 carries the
GLM-5.2 NVFP4 optimization, and the stitch fork's O(delta) partial reload is validated
byte-identical on NVFP4 (cutlass + trtllm), so per-version reloads are ~O(delta).

Everything below that is NOT marked ``TODO(glm5.2)`` is shared with the Kimi NVFP4
recipe and correct as-is (the NVFP4 QAT recipe, the experts-only te_precision_config,
the disk-delta publish, the GRPO/optimizer/data blocks, the checkpoint prefetch, the
Modal B200 shape). The ``TODO(glm5.2)`` items are the model-specific values that must
be confirmed before deploy — see the checklist at the bottom.

Deploy as its own app:
    EXPERIMENT_CONFIG=glm5_2_nvfp4 \
      uv run --extra modal modal deploy --strategy recreate -m cookbook.miles_disagg.app
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import DATA_PATH, PREP_PATH
from cookbook.miles_disagg.config import MilesConfig


APP_NAME = "stitch-glm5-2-nvfp4"
DELTA_VOLUME_NAME = "stitch-delta-glm5-2-nvfp4"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

# GLM-5.2 ships bf16 (huggingface.co/zai-org/GLM-5.2) — unlike Kimi's INT4 — so the
# bf16 masters ARE the source (no dequant); the served NVFP4 base is generated from it
# with tools/convert_hf_to_nvfp4.py, same packing as the trainer's NVFP4 export.
SOURCE_MODEL = "zai-org/GLM-5.2"
MODEL_TAG = "glm5-2-nvfp4"

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_DEBUG_REQUESTS = True
# R3 routing-replay needs the dropless Megatron dispatch fix at startup.
MEGATRON_RUNTIME_PATCHES = [
    "/root/cookbook/miles_disagg/patches/megatron-r3-dispatch.patch",
]


# Serving args for the NVFP4 base on the B200 pool. Shared NVFP4 serving knobs +
# the checkpoint prefetch (cold-volume read ~3x faster). The trainer image injects
# --served-model-name / --dtype / --cuda-graph-max-bs / --trust-remote-code.
SGLANG_SERVER_ARGS = {
    "--weight-loader-prefetch-checkpoints": "",
    "--weight-loader-prefetch-num-threads": "8",
    # GLM 5.2 is glm_moe_dsa: MLA (q/kv-lora) MoE, DeepSeek-V3-arch — so MLA serving
    # like Kimi (tokenspeed_mla + fp8 KV), not GQA.
    # v0.5.15 has no glm5.2 parser: glm45 is the only GLM reasoning detector, glm47 the
    # newest tool-call detector (closest to 5.2). TODO(glm5.2): confirm 5.2's format.
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--dist-timeout": "3600",
    "--kv-cache-dtype": "fp8_e4m3",
    "--attention-backend": "tokenspeed_mla",
    "--context-length": "32768",
    "--mem-fraction-static": "0.85",
    "--chunked-prefill-size": "16384",
    "--schedule-conservativeness": "0.5",
    "--schedule-policy": "lpm",
    "--enable-hierarchical-cache": "",
    "--hicache-ratio": "2",
    "--hicache-io-backend": "kernel",
    "--hicache-mem-layout": "page_first",
    "--hicache-write-policy": "write_through",
    "--skip-server-warmup": "",
    "--enable-return-routed-experts": "",  # routing replay (DeepSeek-V3-arch MoE)
}

SGLANG_ENV = {"SGLANG_ENABLE_RELOAD_LOAD_PLAN": "1"}  # NVFP4: load-plan replay + O(delta) partial reload

modal = ModalConfig(
    gpu="B200",
    region="us",
    # TODO(glm5.2): size to the actual model. These are Kimi's (~1T) numbers — scale
    # down for a smaller GLM 5.2 (memory, ephemeral disk, node count).
    memory=1_650_688,
    rollout_min_containers=8,  # warm floor; Flash scales above under load
    rollout_target_inputs=32,
    proxy_regions=["us-west"],
    rollout_ephemeral_disk_mib=819_200,  # NVFP4 base copy + in-place delta headroom
    trainer_ephemeral_disk_mib=2_097_152,
    # TODO(glm5.2): torch_dist conversion parallelism — match the trainer EP/TP below.
    torch_dist_prep_nodes=4,
    torch_dist_prep_gpus_per_node=8,
    torch_dist_convert_extra_args=(
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 32"
    ),
    torch_dist_prep_ephemeral_disk_mib=2_097_152,
)


class _Miles(MilesConfig):
    # GLM 5.2 = 744B-A40B (256 experts, 8 active, 1 shared, 78 layers, MLA). The
    # _5layer / _lora variants exist for smoke / lora runs.
    miles_model_script = "scripts/models/glm5.2-744B-A40B.sh"

    hf_checkpoint = f"{PREP_PATH}/{MODEL_TAG}/nvfp4"      # served NVFP4 base
    ref_load = f"{PREP_PATH}/{MODEL_TAG}/torch_dist"      # trainer torch_dist ckpt
    # glm_moe_dsa routes to the DeepSeekV3 export converter (megatron_to_hf/__init__.py:
    # "glm_moe_dsa"/"glmmoedsa" -> convert_deepseekv3_to_hf). "raw" builds the GPTModel
    # from the model script's MODEL_ARGS (MLA MoE), like the Kimi / DeepSeek recipes.
    megatron_to_hf_mode = "raw"
    model_name = "glm_moe_dsa"

    # Disaggregated publish-only rollout; modal_train fills rollout_endpoint_url.
    # TODO(glm5.2): size actor_num_nodes / parallelism to the model (Kimi's 16x8=128).
    actor_num_nodes = 16
    actor_num_gpus_per_node = 8
    num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 4  # TODO(glm5.2): B200s per rollout engine (fit the base)
    rollout_endpoint_url = None
    use_miles_router = True

    custom_rollout_request_hook_path = (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        "rollout_request_retry_attempts": 1200,  # outlast a full cold pool load
        "rollout_request_retry_sleep": 1.0,
        "rollout_session_affinity_header": "Modal-Session-ID",
        "rollout_request_timeout_secs": 300,
    }

    async_mode = True
    update_weights_interval = 1

    # NVFP4 QAT — miles' canonical NVFP4 RL recipe (shared with Kimi; correct as-is).
    fp4_format = "e2m1"
    fp4_recipe = "nvfp4"
    fp4_param_gather = False
    # NVFP4 only on the routed expert GEMMs, everything else bf16 — matches the
    # experts-only served base. (GLM MoE expert module path is the same
    # *.mlp.experts.linear_fc{1,2} in Megatron.)
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
    # Keep ALL routed experts NVFP4 (reload-safe: sglang's fused-MoE reload loader
    # allocates NVFP4 for every expert layer; a bf16 carve-out layer can't reload).
    # first_k_dense_replace=3: the first 3 layers are dense (no routed experts).
    num_layers_at_start_in_bf16 = 3
    num_layers_at_end_in_bf16 = 0

    # Disk-delta publish-only over the Modal Volume bulletin board.
    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"

    # Data: dapo-math-17k.
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"
    eval_interval = None

    # Rollout (bring-up smoke length; scale num_rollout / batch for a real run).
    num_rollout = 3
    save_interval = 20
    rollout_batch_size = 32
    rollout_max_response_len = 4096
    rollout_temperature = 0.8
    n_samples_per_prompt = 8
    global_batch_size = 256
    use_dynamic_global_batch_size = True
    sglang_server_concurrency = 256

    use_rollout_routing_replay = True

    # TODO(glm5.2): trainer parallelism — size to the model + actor_num_nodes above.
    # These mirror Kimi's 128-GPU layout; adjust TP/PP/CP/EP + decoder_last_pipeline
    # to GLM 5.2's layer/expert counts.
    tensor_model_parallel_size = 8
    sequence_parallel = True
    pipeline_model_parallel_size = 8
    context_parallel_size = 2
    expert_model_parallel_size = 16
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 5
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 16384
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    no_check_for_nan_in_loss_and_grad = True

    # Optimizer.
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    # Algorithm (GRPO + truncated importance sampling).
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
        # NVFP4 numerics (shared with Kimi; required for correct NVFP4 QAT).
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
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


# ── Resolved from miles + the GLM-5.2 HF config ──────────────────────────────
#  source = zai-org/GLM-5.2 (bf16, no dequant); script = glm5.2-744B-A40B.sh;
#  model_name = glm_moe_dsa (-> convert_deepseekv3_to_hf); MLA serving (tokenspeed_mla
#  + fp8 KV); first_k_dense_replace = 3. Arch: 78 layers, 256 experts (8 active),
#  1 shared, MLA (q_lora 2048 / kv_lora 512), sigmoid router topk-scaling 2.5.
#
# ── TODO(glm5.2) still to confirm before a real run ──────────────────────────
#  1. Parsers: glm45 vs a GLM-5.2-specific reasoning/tool parser in sglang v0.5.15.
#  2. Size the trainer to 744B-A40B / 78 layers: actor_num_nodes + TP/PP/CP/EP +
#     decoder_last_pipeline_num_layers + memory / ephemeral disk (the values below are
#     Kimi's ~1T/128-GPU layout as a starting point — verify the PP split for 78 layers).
# ─────────────────────────────────────────────────────────────────────────────

miles = _Miles()
