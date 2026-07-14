"""Kimi K2.6 GRPO on Modal, disaggregated, native-NVFP4 end to end.

Deploy: EXPERIMENT_CONFIG=kimi_k2_6_nvfp4 uv run --extra modal modal deploy --strategy recreate -m cookbook.miles_disagg.app
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import DATA_PATH, PREP_PATH
from cookbook.miles_disagg.config import MilesConfig


APP_NAME = "stitch-kimi-k2-6-nvfp4"
DELTA_VOLUME_NAME = "stitch-delta-kimi-k2-6-nvfp4"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

# SOURCE_MODEL (INT4) -> dequant -> bf16 masters -> convert -> served NVFP4 base.
SOURCE_MODEL = "moonshotai/Kimi-K2.6"
MODEL_TAG = "kimi-k2-6-nvfp4"

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_DEBUG_REQUESTS = True
# R3 routing-replay needs the dropless Megatron dispatch fix at startup.
MEGATRON_RUNTIME_PATCHES = [
    "/root/cookbook/miles_disagg/patches/megatron-r3-dispatch.patch",
]


# mem-fraction / context-length are STARTING POINTS — measure on a warm B200:4 and adjust.
SGLANG_SERVER_ARGS = {
    "--weight-loader-prefetch-checkpoints": "",
    "--weight-loader-prefetch-num-threads": "8",
    "--tool-call-parser": "kimi_k2",
    "--reasoning-parser": "kimi_k2",
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
    "--enable-return-routed-experts": "",
}

SGLANG_ENV = {"SGLANG_ENABLE_RELOAD_LOAD_PLAN": "1"}  # NVFP4: load-plan replay + O(delta) partial reload

modal = ModalConfig(
    gpu="B200",
    region="us",
    # Request Modal's max (nodes are 1.79 TiB): the CPU-offloaded 1T optimizer + publish
    # snapshot peak ~1.2 TiB, and a smaller *request* got OOM-killed (excess is best-effort).
    memory=1_650_688,
    rollout_min_containers=8,  # warm floor: skip the cold 2->N ramp that 502'd v5's rollout
    # scale OUT at ~32 concurrent/container so Flash spreads load, not a few KV caches.
    rollout_target_inputs=32,
    proxy_regions=["us-west"],
    rollout_ephemeral_disk_mib=819_200,  # ~591 GB base copy + in-place delta-apply headroom
    # Ray logs + object spill fill the default disk (ENOSPC'd); 2 TiB gives headroom.
    trainer_ephemeral_disk_mib=2_097_152,
    # 4x8=32-way: EP32 shards 384 experts to ~90 GB/rank so distcp writes finish inside the
    # 900s heartbeat (EP16 raced it); torch_dist reshards on load into the EP16 trainer.
    torch_dist_prep_nodes=4,
    torch_dist_prep_gpus_per_node=8,
    torch_dist_convert_extra_args=(
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 32"
    ),
    # ~700 GB of distcp shards buffer before commit; 2 TB gives headroom.
    torch_dist_prep_ephemeral_disk_mib=2_097_152,
)


class _Miles(MilesConfig):
    # Arch comes from the model script (shared with Kimi-K2-Thinking; K2.6 matches).
    miles_model_script = "scripts/models/kimi-k2-thinking.sh"

    hf_checkpoint = f"{PREP_PATH}/{MODEL_TAG}/nvfp4"
    ref_load = f"{PREP_PATH}/{MODEL_TAG}/torch_dist"
    # "raw" (not "bridge"): K2.6's HF arch is the KimiK25 VLM wrapper, which AutoBridge
    # can't build. Export still routes via model_name below.
    megatron_to_hf_mode = "raw"
    model_name = (
        "kimi_k25"  # megatron_to_hf export dispatch (convert_kimi_k25_to_hf + NVFP4)
    )

    actor_num_nodes = 8  # 8x8 B200 = 64 GPUs (trainer only; pool is elastic on top)
    actor_num_gpus_per_node = 8
    num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 4  # B200:4 per rollout container (K2.6 NVFP4 fits)
    rollout_endpoint_url = None
    use_miles_router = True

    # Staleness gate; knobs ride in custom_config_path (read by the hook, not miles core).
    custom_rollout_request_hook_path = (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        # 1200×1s=20 min outlasts a full ~16 min cold-load (240s was too short, rollout 503'd).
        "rollout_request_retry_attempts": 1200,
        "rollout_request_retry_sleep": 1.0,
        "rollout_session_affinity_header": "Modal-Session-ID",
        # finite read timeout; without it a request to a scaled-down Flash container hangs forever.
        "rollout_request_timeout_secs": 300,
    }

    async_mode = True
    update_weights_interval = 1

    # NVFP4 QAT — miles' canonical NVFP4 RL recipe.
    fp4_format = "e2m1"
    fp4_recipe = "nvfp4"
    # False: with it True, TE NVFP4Tensor params crash Megatron DDP's param-buffer repoint.
    fp4_param_gather = False
    # NVFP4 only on the routed expert GEMMs, everything else bf16 — matches the served base.
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
    # bf16 carve-out for the dense first layer; END stays 0 (reload-safe): SGLang's fused-MoE
    # reload loader allocates NVFP4 for every expert layer, so a bf16 last layer can't reload.
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

    num_rollout = 10  # 10 GRPO steps; each publishes a delta to inspect for tensor-sparsity
    save_interval = 20  # megatron requires it; > num_rollout so this run skips saves
    rollout_batch_size = 32
    rollout_max_response_len = 4096
    rollout_temperature = 0.8
    n_samples_per_prompt = 8
    global_batch_size = 256
    use_dynamic_global_batch_size = True
    sglang_server_concurrency = 256

    use_rollout_routing_replay = True

    # Trainer parallelism for 8x8=64 GPUs: TP8*PP2*CP4=64, EP32 over the experts — the
    # proven kimi_k25_lora_8nodes topology. EP32 keeps expert weights+grads ~63GB/rank.
    tensor_model_parallel_size = 8
    sequence_parallel = True
    pipeline_model_parallel_size = 2
    context_parallel_size = 4
    expert_model_parallel_size = 32
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 30
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
        # NVFP4 numerics: required for correct NVFP4 QAT.
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


miles = _Miles()
