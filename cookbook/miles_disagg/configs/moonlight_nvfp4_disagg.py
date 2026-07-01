"""Moonlight-16B-A3B GRPO on Modal, disaggregated, native-NVFP4 end to end.

This is the small, *runnable* de-risk for the Kimi-K2.6-NVFP4 recipe
(``kimi_k2_6_nvfp4_disagg``). Moonlight is the same DeepSeek-V3 architecture
family (MLA + DeepSeek-MoE: sigmoid router, shared experts, grouped top-k) at a
size that fits a single Blackwell node, so it exercises the full
QAT → NVFP4 export → XOR-delta → SGLang-reload loop cheaply.

WHY NVFP4 IS ALL-BLACKWELL
--------------------------
NVFP4 QAT in miles is native Megatron ``--fp4-format e2m1`` → ``NVFP4BlockScaling``
(TransformerEngine ≥ 2.7.0.dev0). FP4 GEMM requires Blackwell, so BOTH the
trainer and the rollout pool run on B200 — unlike the INT4 recipe, which fake-
quantizes (straight-through) on H200. There is no simulated/non-Blackwell NVFP4
weight-QAT path.

CHECKPOINT LIFECYCLE (three roles; see modal_train.prepare_checkpoints)
-----------------------------------------------------------------------
  * BF16 masters (``ref_load``): Moonlight ships bf16, so the masters are the
    downloaded checkpoint directly (no dequant needed — unlike Kimi, which is
    published as INT4 and must be dequantized).
  * Served NVFP4 base (``hf_checkpoint``): produced by miles' own
    ``tools/convert_hf_to_nvfp4.py`` from the bf16 masters. Because the SAME
    TE reference quantizer produces the base and the trainer's export, the
    served packing == the export packing BY CONSTRUCTION, the step-0 export
    equals the base, and the first XOR delta is ~empty. This is exactly what
    de-risks the byte-exact axis that the Kimi recipe must verify on a warm
    container.
  * Megatron torch_dist (``load``/``save``): the trainer's own rollout
    checkpoints — never seen by the rollout pool.

The trainer reads ``hf_checkpoint`` (the NVFP4 base) for both the export
quantization_config and the diff baseline; ``_capture_baseline`` seeds the
snapshot from those exact bytes, so applying delta_vN reproduces export_vN
byte-for-byte and the served weights become the trainer's NVFP4 export.

Deploy as its own app:
    EXPERIMENT_CONFIG=moonlight_nvfp4_disagg \
      uv run --extra modal modal deploy --strategy recreate -m cookbook.miles_disagg.modal_train
"""

from __future__ import annotations

from cookbook.miles_disagg.configs.base import DATA_PATH, PREP_PATH, ModalConfig, MilesConfig


APP_NAME = "miles-moonlight-nvfp4-disagg"
DELTA_VOLUME_NAME = "miles-delta-bulletin-moonlight-nvfp4"
DELTA_BULLETIN_ROOT = "/delta-bulletin"

# Source HF repo (bf16) the prepare step turns into masters + NVFP4 base.
SOURCE_MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"
MODEL_TAG = "moonlight-16b-nvfp4"  # names the prepared dirs under PREP_PATH

SIDECAR_DEBUG_REQUESTS = True


def build_serving_image(**kwargs):
    """Per-experiment rollout-pool image (modal_train picks this up if present)."""
    from cookbook.miles_disagg.serving import build_nvfp4_b200_serving_image

    return build_nvfp4_b200_serving_image(**kwargs)


# Moonlight NVFP4 on a B200 pool. The trainer image injects --served-model-name /
# --dtype / --trust-remote-code; only the MLA/MoE/cache extras live here. No
# --quantization flag — NVFP4 is driven by the served checkpoint's own
# quant config. mem-fraction / context-length are STARTING POINTS; measure.
SGLANG_SERVER_ARGS = {
    "--attention-backend": "tokenspeed_mla",
    "--kv-cache-dtype": "fp8_e4m3",  # tokenspeed_mla requires this
    "--context-length": "8192",  # Moonlight's max_position_embeddings
    "--mem-fraction-static": "0.8",
    "--chunked-prefill-size": "4096",
    "--skip-server-warmup": "",
    # Routing replay: the pool emits per-token routed experts so the trainer can
    # replay them (DeepSeek-V3-arch MoE supports it).
    "--enable-return-routed-experts": "",
}

modal = ModalConfig(
    gpu="B200",
    region="us",
    # Warm floor of 1: the pool must be UP before the trainer sends rollouts.
    # With min=0 the pool only scales on rollout traffic, and scale-from-0 either
    # didn't trigger or lost the race against the rollout retry budget, stalling
    # the trainer in model-load. Flash still scales ABOVE this floor under load.
    rollout_min_containers=1,
    proxy_regions=["us-west"],
)


class _Miles(MilesConfig):
    # Architecture comes from the model script (MLA + the full DeepSeek-MoE arg
    # set); do NOT inline arch attrs here.
    miles_model_script = "scripts/models/moonlight.sh"

    # Checkpoints (absolute paths -> the launcher skips HF download for them).
    # hf_checkpoint is the served NVFP4 base; ref_load is the bf16 masters.
    hf_checkpoint = f"{PREP_PATH}/{MODEL_TAG}/nvfp4"
    ref_load = f"{PREP_PATH}/{MODEL_TAG}/bf16"
    megatron_to_hf_mode = "bridge"
    model_name = "deepseekv3"  # bridge dispatch: Moonlight is DeepSeek-V3 arch

    # Disaggregated publish-only rollout; modal_train fills rollout_endpoint_url.
    actor_num_nodes = 1
    actor_num_gpus_per_node = 4  # 1 node x 4 B200 trainer (matches the proven moonlight recipe)
    num_gpus_per_node = 4
    colocate = False  # disk-delta is incompatible with --colocate
    rollout_num_gpus = 0  # publish-only forces this; set explicitly for clarity
    rollout_num_gpus_per_engine = 1  # B200:1 per rollout container (Moonlight NVFP4 is tiny)
    rollout_endpoint_url = None
    use_miles_router = True

    # Staleness gate. custom_rollout_request_hook_path is a real miles CLI arg;
    # the gate KNOBS are consumed only by the hook (not miles core), so they ride
    # in custom_config_path — miles setattr's every YAML key onto the args
    # namespace, and the hook reads them via getattr(args, ...). modal_train
    # merges the dynamic bulletin identity (run_id, volume, app) into this dict.
    custom_rollout_request_hook_path = "cookbook.miles_disagg.hooks.gated_rollout_request_hook"
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,  # bounded staleness window
        "rollout_request_retry_attempts": 240,
        "rollout_request_retry_sleep": 1.0,
        "rollout_session_affinity_header": "Modal-Session-ID",
    }

    # Async-first: train_async pipelines generate(N+1) with train(N).
    async_mode = True
    update_weights_interval = 1

    # ── NVFP4 QAT (native Megatron FP4 training; Blackwell + TE >= 2.7.0.dev0) ──
    # --fp4-format takes the element format 'e2m1'; the recipe (fp4_recipe) already
    # defaults to 'nvfp4' (NVFP4BlockScaling). --fp4-param-gather keeps params in
    # fp4 to save memory. NVFP4 group size is fixed at 16.
    fp4_format = "e2m1"
    fp4_param_gather = True

    # ── Disk-delta publish-only over the Modal Volume bulletin board ──
    update_weight_transfer_mode = "disk"
    update_weight_disk_delta = True
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT  # modal_train run-scopes this
    custom_delta_pre_push_path = "cookbook.miles_disagg.hooks.commit_and_wake"

    # Data: dapo-math-17k (the math-reasoning set the Kimi recipe uses).
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
    save_interval = 1000  # megatron requires it; > num_rollout so the smoke skips megatron saves
    rollout_batch_size = 32
    rollout_max_response_len = 4096  # fits within the 8192 context (prompt + response)
    rollout_temperature = 0.8
    n_samples_per_prompt = 8
    global_batch_size = 128
    use_dynamic_global_batch_size = True
    sglang_server_concurrency = 64

    # Routing replay (R3): replay sglang's routed experts in the train/log-prob
    # forward. Being root-caused — instrumented run to capture the MoE token-count
    # mismatch behind "Split sizes doesn't match total dim 0 size".
    use_rollout_routing_replay = True

    # Trainer parallelism (the proven moonlight setting: TP2/SP/PP1/CP1/EP4).
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

    # Optimizer (CPU offload keeps GPU state tiny for ~3B active).
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

    # Ray runtime environment.
    environment = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NVSHMEM_DISABLE_NCCL": "1",
        "NCCL_TIMEOUT_MS": "360000000",
    }

    def prepare_data(self) -> None:
        from datasets import load_dataset

        # DAPO-Math-17k columns are `prompt` (chat list) and the gold answer under
        # `reward_model.ground_truth` — lift the answer into a `label` column.
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


miles = _Miles()
