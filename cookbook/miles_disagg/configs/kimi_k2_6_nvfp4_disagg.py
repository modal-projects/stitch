"""Kimi K2.6 GRPO on Modal, disaggregated, native-NVFP4 end to end.

Kimi K2.6 is a ~1T-parameter DeepSeek-V3-architecture MoE (MLA + DeepSeek-MoE:
sigmoid router, shared experts, grouped top-k). The rollout pool serves the
model's **NVFP4 (W4A4, experts-only)** checkpoint on Blackwell; the trainer
QAT-trains in NVFP4 and publishes byte-exact XOR deltas against that served base.

This is the full-scale recipe — a 16x8 B200 (128 GPU) trainer footprint, with an
elastic Flash B200 rollout pool on top. Run ``moonlight_nvfp4_disagg`` first: it
exercises the exact same QAT → NVFP4-export → XOR-delta → SGLang-reload machinery
on a single Blackwell node and de-risks the one axis this recipe can only verify
on a warm container (export packing == served packing). The Kimi-specific paths
this adds over Moonlight are the INT4→bf16 dequant (``convert_kimi_int4_to_bf16``)
and the ``kimi_k25`` megatron_to_hf export converter (``convert_kimi_k25_to_hf``).

NVFP4 IS ALL-BLACKWELL
----------------------
NVFP4 QAT = native Megatron ``--fp4-format e2m1`` → ``NVFP4BlockScaling`` (TE >=
2.7.0.dev0). FP4 GEMM requires Blackwell, so the TRAINER is B200 too (not H200 +
fake-QAT like the INT4 recipe). The serving pool is the existing Blackwell SGLang
fork (serving.py), proven for NVFP4.

CHECKPOINT LIFECYCLE (three roles; see modal_train.prepare_checkpoints)
-----------------------------------------------------------------------
Neither published K2.6 checkpoint is bf16: ``moonshotai/Kimi-K2.6`` ships as
compressed-tensors INT4. So:
  * BF16 masters (``ref_load``): dequantize the moonshotai INT4 checkpoint with
    ``tools/convert_kimi_int4_to_bf16.py``.
  * Served NVFP4 base (``hf_checkpoint``): produce with miles'
    ``tools/convert_hf_to_nvfp4.py`` from the bf16 masters, so the served packing
    == the trainer's export packing BY CONSTRUCTION (smallest deltas, no byte-
    exact risk).
  * Megatron torch_dist (``load``/``save``): trainer-internal rollout ckpts.

The trainer reads ``hf_checkpoint`` (NVFP4 base) for the export quant config AND
the diff baseline (``_capture_baseline`` seeds the snapshot from those exact
bytes), so applying delta_vN reproduces export_vN byte-for-byte.

INVARIANT: NVFP4 group size is fixed at 16 and the served base's
quantization_config (quant_algo == "NVFP4", experts-only, FP8 KV) drives both
the engine load and the miles export dispatch (added in
megatron_to_hf/processors/__init__.py: route on quant_algo == "NVFP4").

Deploy as its own app:
    EXPERIMENT_CONFIG=kimi_k2_6_nvfp4_disagg \
      uv run --extra modal modal deploy --strategy recreate -m cookbook.miles_disagg.modal_train

Prerequisites the bring-up depends on (flagged, validated by the moonlight run):
  1. The miles trainer image's TransformerEngine must be >= 2.7.0.dev0 (NVFP4
     BlockScaling), and the trainer must run on Blackwell. Verify on a warm B200.
  2. SGLang must serve NVFP4 MLA MoE on Blackwell (serving.py fork, proven for
     NVFP4) — verify the prepared base loads on a warm container.
  3. The exact ``--fp4-format`` companion args for K2.6 (e.g. high-precision
     first/last layers) may need tuning; confirm against the Megatron fork.
"""

from __future__ import annotations

from cookbook.miles_disagg.configs.base import (
    DATA_PATH,
    PREP_PATH,
    ModalConfig,
    MilesConfig,
)


APP_NAME = "miles-kimi-k2-6-nvfp4-disagg"
DELTA_VOLUME_NAME = "miles-delta-bulletin-kimi-k2-6-nvfp4"
DELTA_BULLETIN_ROOT = "/delta-bulletin"

# Source checkpoints the prepare step consumes (see modal_train.prepare_checkpoints):
#   SOURCE_MODEL  -> dequantize INT4 -> bf16 masters -> convert -> served NVFP4 base
SOURCE_MODEL = "moonshotai/Kimi-K2.6"
MODEL_TAG = "kimi-k2-6-nvfp4"

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_DEBUG_REQUESTS = True


def build_serving_image(**kwargs):
    from cookbook.miles_disagg.serving import build_nvfp4_b200_serving_image

    return build_nvfp4_b200_serving_image(**kwargs)


# Kimi K2.6 on the B200 pool serving the NVFP4 base. The trainer image injects
# --served-model-name / --dtype / --cuda-graph-max-bs / --max-running-requests /
# --trust-remote-code; only the Kimi/MLA/cache extras live here. mem-fraction and
# context-length are STARTING POINTS — measure on a warm B200:4 and adjust.
SGLANG_SERVER_ARGS = {
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

modal = ModalConfig(
    gpu="B200",
    region="us",
    # Host RAM (request, limit MiB). The 1T optimizer is CPU-offloaded (optimizer_cpu_offload —
    # required; the largest PP-stage ranks' fp32 Adam+master ~104 GB wouldn't fit GPU alongside
    # the publish gather), so each rank holds ~145 GB host RAM => ~1.15 TB/node steady, and the
    # disk-delta publish (baseline snapshot of ~24k tensors) spikes higher. A node OOM-killed at
    # ~1.2 TiB because the *request* (768 GiB) — not the limit — was the binding guarantee (memory
    # above the request is best-effort, reclaimed under node pressure). AWS B200:8 nodes have
    # 1.79 TiB RAM; Modal caps the request at 1,650,688 MiB (1.574 TiB), so request that max —
    # guaranteed past the ~1.2 TiB publish peak, with OS headroom on the 1.79 TiB node.
    memory=1_650_688,
    # Warm floor so the pool is up before the trainer sends rollouts (see the
    # moonlight config). Flash scales above this under load.
    rollout_min_containers=8,  # warm floor: skip the cold 2->N ramp that 502'd v5's rollout
    # Scale OUT at ~32 concurrent/container (vs the 256 engine concurrency) so Flash
    # spreads the rollout across containers instead of saturating a few KV caches.
    rollout_target_inputs=32,
    proxy_regions=["us-west"],
    # The served NVFP4 base is ~591 GB; the sidecar copies it to /local-checkpoint
    # on ephemeral disk. 800 GiB covers the copy plus in-place delta-apply headroom.
    rollout_ephemeral_disk_mib=819_200,
    # 700 GiB host-RAM request: the 595 GB local checkpoint must stay
    # page-cache resident or every reload pays ~120s of capacity misses.
    rollout_memory_mib=716_800,
    # Trainer nodes: Ray logs + object spill (rollout batches + per-publish full-model
    # gathers) accumulate under /tmp/ray over the run; the default disk progressively
    # ENOSPC'd. 2 TiB of the B200:8 local NVMe gives ample headroom.
    trainer_ephemeral_disk_mib=2_097_152,
    # torch_dist conversion: 4x8 B200 = 32-way. EP32 shards the 384 experts to
    # ~90 GB/rank so each rank's distcp write finishes well inside the 900s cluster
    # heartbeat window (EP16's ~150 GB/rank raced it and missed .metadata). TP1/PP1
    # keeps the (replicated) non-expert small; PP1 is intentional (the convert skips
    # auto-pp=world_size when EP>1). torch_dist reshards on load, so EP32-saved loads
    # fine into the EP16 trainer.
    torch_dist_prep_nodes=4,
    torch_dist_prep_gpus_per_node=8,
    torch_dist_convert_extra_args=(
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 32"
    ),
    # Each node buffers ~700 GB of distcp shards in the Volume write cache before commit;
    # 2 TB gives rank 0 (shards + .metadata + common.pt) headroom over the 800 GB pool default.
    torch_dist_prep_ephemeral_disk_mib=2_097_152,
)


class _Miles(MilesConfig):
    # Architecture is sourced from the model script (MLA + the full DeepSeek-MoE
    # arg set). Shared with Kimi-K2-Thinking, whose arch K2.6 matches.
    miles_model_script = "scripts/models/kimi-k2-thinking.sh"

    # Checkpoints (absolute -> launcher skips HF download). hf_checkpoint is the
    # served NVFP4 base; ref_load is the Megatron torch_dist checkpoint (built from
    # the bf16 masters by prepare_torch_dist). Raw mode can ONLY load torch_dist
    # (miles' HF load is bridge-only); the conversion uses the KimiK25 mbridge.
    hf_checkpoint = f"{PREP_PATH}/{MODEL_TAG}/nvfp4"
    ref_load = f"{PREP_PATH}/{MODEL_TAG}/torch_dist"
    # "raw" builds the GPTModel from the model script's MODEL_ARGS (MLA + DeepSeek-
    # MoE), NOT via megatron-bridge AutoBridge. K2.6's HF arch is the VLM wrapper
    # KimiK25ForConditionalGeneration, which AutoBridge does not support (it knows
    # DeepseekV3ForCausalLM etc.) — so "bridge" crashes in init. This is how slime
    # builds DeepSeek-V3/MLA MoE models. Export still routes via model_name below.
    megatron_to_hf_mode = "raw"
    model_name = (
        "kimi_k25"  # megatron_to_hf export dispatch (convert_kimi_k25_to_hf + NVFP4)
    )

    # Disaggregated publish-only rollout; modal_train fills rollout_endpoint_url.
    actor_num_nodes = 16  # 16x8 B200 = 128 GPUs (trainer only; pool is elastic on top)
    actor_num_gpus_per_node = 8
    num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 4  # B200:4 per rollout container (K2.6 NVFP4 fits)
    rollout_endpoint_url = None
    use_miles_router = True

    # Staleness gate. The hook path is a real miles CLI arg; the gate KNOBS ride
    # in custom_config_path (consumed by the hook, not miles core). modal_train
    # merges the dynamic bulletin identity (run_id, volume, app) into this dict.
    custom_rollout_request_hook_path = (
        "cookbook.miles_disagg.hooks.gated_rollout_request_hook"
    )
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        # 240×1s=240s was too short: after a (re)deploy the pool cold-loads the ~591 GB
        # base for ~16 min, and the trainer can reach iter-1 rollouts first -> the rollout
        # exhausted retries on 503s and failed. 1200×1s=20 min outlasts a full cold-load,
        # so rollouts ride out a cold/restarting pool instead of dying on the race.
        "rollout_request_retry_attempts": 1200,
        "rollout_request_retry_sleep": 1.0,
        "rollout_session_affinity_header": "Modal-Session-ID",
        # Finite per-request read timeout for the disagg /generate client (init_http_client).
        # Without it the client uses httpx.Timeout(None) and a request to a Flash container
        # that scaled down mid-flight hangs forever (rollout stalls at ~N%, engines idle).
        # 300s comfortably exceeds a full 4096-token generation (~130s at ~30 tok/s) but
        # recovers a lost request fast: it errors -> the retry above reroutes it.
        "rollout_request_timeout_secs": 300,
    }

    async_mode = True
    update_weights_interval = 1

    # NVFP4 QAT — miles' canonical NVFP4 RL recipe.
    fp4_format = "e2m1"
    fp4_recipe = "nvfp4"
    # fp4_param_gather=False keeps NVFP4 GEMM compute QAT (config.fp4 in raw
    # mode) with bf16 master params. With it True, params are TE NVFP4Tensor and
    # Megatron DDP's param-buffer repoint (modify_underlying_storage -> TE
    # replace_raw_data) crashes (TE: FP8 yes, NVFP4 no).
    fp4_param_gather = False
    # Per-module TE precision config: NVFP4 ONLY on the routed expert GEMMs,
    # everything else bf16 — matches the experts-only served base. Materialized
    # to a temp YAML and passed as --te-precision-config-file.
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
    # bf16 carve-out for the dense first layer (FIRST_K_DENSE_REPLACE=1). Passed to BOTH
    # convert_hf_to_nvfp4 (served base) and the trainer so they agree.
    # NOTE: the END carve-out is 0, NOT 1. A bf16 LAST MoE layer (layer 60) made the served
    # base's last-layer experts bf16 while SGLang's fused-MoE update_weights_from_disk RELOAD
    # loader allocates NVFP4 for every expert layer -> it can't load a bf16 [7168,2048] expert
    # into the NVFP4-packed buffer (cold-load honors the carve-out, reload does not). So the
    # reload-safe scheme is to keep all routed experts NVFP4; if QAT later needs the last-layer
    # bf16 carve-out, the alternative is patching SGLang's fused-MoE reload loader.
    num_layers_at_start_in_bf16 = 1
    num_layers_at_end_in_bf16 = 0

    # Disk-delta publish-only over the Modal Volume bulletin board.
    update_weight_transfer_mode = "disk"
    update_weight_disk_delta = True
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT
    custom_delta_pre_push_path = "cookbook.miles_disagg.hooks.commit_and_wake"

    # Data: dapo-math-17k.
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"
    eval_interval = None

    # Rollout (bring-up smoke length; scale num_rollout for a real run).
    # Sized for FAST loop closure (proving the publish), not a real run: a small
    # batch and short responses keep the first rollout to minutes, not the ~1.6 h
    # the full 128x8x16384 took (per-sequence decode of 16 K-token responses, not
    # pool size, is the bound). Restore rollout_batch_size=128 /
    # rollout_max_response_len=16384 once v1 publishes and we scale num_rollout.
    num_rollout = 3  # trimmed e2e loop-closure smoke (scale up for a real run)
    save_interval = 20  # megatron requires it; > num_rollout so the smoke skips saves
    rollout_batch_size = 32
    rollout_max_response_len = 4096
    rollout_temperature = 0.8
    n_samples_per_prompt = 8
    global_batch_size = 256
    use_dynamic_global_batch_size = True
    sglang_server_concurrency = 256

    use_rollout_routing_replay = True

    # Trainer parallelism, sized for 16x8=128 GPUs. Derived from the proven Kimi
    # 32x8 layout (TP8/PP8/CP4/EP32) by halving the non-PP width: TP8*PP8*CP2 =
    # 128 (DP1), and for the experts ETP1*EP16 = 16 = TP8*CP2*DP1. Keeping TP8/PP8
    # preserves the per-stage layer split (and decoder_last_pipeline_num_layers).
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
        # NVFP4 numerics: without these the NVFP4 QAT is mis-configured even
        # once the build/DDP/load gaps are cleared.
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
