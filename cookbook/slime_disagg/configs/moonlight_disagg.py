"""Moonlight-16B-A3B GRPO on Modal Flash, disaggregated (M1: synchronous bring-up).

Moonlight-16B-A3B is Moonshot's small DeepSeek-V3-architecture MoE -- MLA plus
DeepSeek-MoE (sigmoid router, shared experts, grouped top-k) -- i.e. the Kimi
K2.6 family at a size that fits a single H200 rollout container. This config is
the synchronous bring-up rung; routing replay and async-first layer on top (see
the M2/M3 notes at the bottom).

Reshape provenance rule (colocated `run-moonlight-16B-A3B.sh` -> disagg config):
  ARCHITECTURE -> sourced from scripts/models/moonlight.sh via `slime_model_script`
                  (NO arch attrs are set here -- the script is the single source).
  PARALLELISM  -> this config, scaled from the recipe's single-node TP4/EP8.
  INFERENCE    -> SGLANG_SERVER_ARGS / rollout_num_gpus_per_engine (Modal layer).
  ALGO/DATA    -> this config.

Deploy as its own app:
    EXPERIMENT_CONFIG=moonlight_disagg m deploy --strategy recreate -m cookbook.slime_disagg.modal_train

Prerequisites the bring-up depends on (flagged, not yet automated):
  1. The training image's slime fork ref (modal_train.SLIME_REPO_REF) must include
     scripts/models/moonlight.sh, the deepseekv3 megatron_to_hf export, and the
     routing-replay code. The base image's SGLang (used by the rollout pool) is
     independent.
  2. Bridge-mode HF load is UNVERIFIED for Moonlight/DeepSeek-V3 (it is proven for
     Qwen here). If load fails, the fallback is the proven torch_dist path
     (tools/convert_hf_to_torch_dist.py + ref_load pointed at the converted dir).
"""

from __future__ import annotations

from cookbook.slime_disagg.configs.base import DATA_PATH, ModalConfig, SlimeConfig


APP_NAME = "slime-moonlight-disagg"
DELTA_VOLUME_NAME = "slime-delta-bulletin-moonlight"
DELTA_BULLETIN_ROOT = "/delta-bulletin"

# M3: in_place commit applies weights without draining in-flight rollouts. Stale
# KV is isolated per weight version by the sidecar's extra_key stamping (so old
# requests keep decoding on their version's KV and it drains as they finish);
# min-version pins cross commits freely, only exact pins are quiesced.
SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_DEBUG_REQUESTS = True

# Moonlight serving on the elastic pool. MLA's compressed KV is tiny, so one
# H200 holds the ~32 GB bf16 weights plus a large KV pool. mem-fraction-static
# is a STARTING POINT -- measure it on a warm container and adjust (the only
# datapoint from the colocated recipe is 0.7 at TP8, a different topology).
SGLANG_SERVER_ARGS = {
    "--context-length": "8192",
    "--mem-fraction-static": "0.85",
    # M2 routing replay: the pool must emit per-token routed experts. slime
    # launches no engine in publish-only mode, so this is set here (not by
    # sglang_engine.py). num_layers/moe_router_topk come from moonlight.sh, so
    # the rollout's [tokens, 27, 6] reshape matches the served model (M0-verified).
    "--enable-return-routed-experts": "",
}

modal = ModalConfig(gpu="H200", region="us")


class _Slime(SlimeConfig):
    # Architecture is sourced from the model script; do NOT inline arch attrs
    # (the script carries MLA + the full DeepSeek-MoE arg set).
    slime_model_script = "scripts/models/moonlight.sh"

    # Model + checkpoint. Bridge-mode HF load, same as the dense disagg example
    # (see prerequisite 2 in the module docstring -- this is the main M1 unknown).
    hf_checkpoint = "moonshotai/Moonlight-16B-A3B-Instruct"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"

    # Disaggregated publish-only rollout through slime's opaque HTTP endpoint;
    # modal_train fills rollout_endpoint_url from the Flash gateway at launch.
    actor_num_nodes = 2  # 2x8 H200 -> exercises multinode RDMA + expert parallel
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1  # 1xH200 per rollout container (MLA -> cheap KV)
    rollout_endpoint_url = None
    # M3 staleness gate: each rollout request is pinned to
    # min_required_version = latest_published - lag (derived out-of-band from the
    # bulletin `latest`, since the per-request hook gets no rollout_id). A replica
    # more than `lag` versions behind 409s -> retried, so no too-stale rollouts.
    custom_rollout_request_hook_path = "cookbook.slime_disagg.hooks.gated_rollout_request_hook"
    rollout_request_weight_version_mode = "min"
    rollout_request_weight_version_lag = 1  # k: bounded staleness window (tune up if 409 retries bubble)
    rollout_request_retry_attempts = 240
    rollout_request_retry_sleep = 1.0
    rollout_session_affinity_header = "Modal-Session-ID"

    # M3 async-first: one-step off-policy (train_async pipelines generate(N+1) with
    # train(N)); publish weights every step.
    async_mode = True
    update_weights_interval = 1

    # Disk-delta publish-only over the Modal Volume bulletin board (export uses
    # convert_deepseekv3_to_hf for this arch).
    update_weight_mode = "delta"
    update_weight_transport = "disk"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT
    custom_delta_pre_push_path = "cookbook.slime_disagg.hooks.commit_and_wake"

    # Data: dapo-math-17k, the hard math-reasoning set the proven Moonlight recipe
    # uses (MoE-worthy, unlike GSM8K). The convincing demo adds aime eval.
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"
    eval_interval = None  # skip eval during bring-up

    # Rollout. Small num_rollout for the bring-up smoke; scale up for the demo.
    rollout_function_path = "slime.rollout.sglang_rollout.generate_rollout"
    num_rollout = 5
    rollout_batch_size = 64
    rollout_max_response_len = 4096
    rollout_temperature = 1.0
    rollout_top_p = 1.0
    n_samples_per_prompt = 8
    global_batch_size = 128
    sglang_server_concurrency = 64
    use_fault_tolerance = False

    # Trainer parallelism: scaled from the recipe's single-node TP4/EP8 to 2x8
    # (world = TP4 * DP4 = 16; EP8 over the expert region). MLA / MoE arch flags
    # come from the model script; the bring-up keeps the recipe's alltoall
    # dispatcher (the recipe's flex + --moe-enable-deepep is a later perf lever
    # and needs DeepEP in the image).
    tensor_model_parallel_size = 4
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    sequence_parallel = True
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 8192
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True

    # Optimizer (from the recipe; CPU offload keeps GPU state tiny for 3B active).
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    # Algorithm (GRPO).
    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.0

    # Routing replay (M2): replay the rollout engine's per-token expert routing
    # during the training forward/backward to cut train-inference divergence
    # (R3, arxiv 2510.11370). Auto-implies use_routing_replay; needs the pool's
    # --enable-return-routed-experts (in SGLANG_SERVER_ARGS above).
    use_rollout_routing_replay = True

    def prepare_data(self) -> None:
        from datasets import load_dataset

        # Raw DAPO-Math-17k columns are `prompt` (a chat list) and the gold answer
        # under `reward_model.ground_truth` — there is no `label`. slime reads
        # --input-key prompt (+ apply_chat_template) and --label-key label, so lift
        # the answer into a `label` column. The HF artifact is ~1.8M rows; a shuffled
        # subset is ample for bring-up and a short reward-climb.
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


slime = _Slime()

# Milestone progression (kept here so it's legible):
#   M1 (sync MoE bring-up): DONE.
#   M2 (routing replay): DONE — use_rollout_routing_replay=True +
#     --enable-return-routed-experts; num_layers(27)/moe_router_topk(6) from moonlight.sh.
#   M3 (async-first): DONE above — async_mode=True (train_async one-step off-policy),
#     SIDECAR_COMMIT_MODE="in_place" (in-flight updates + extra_key KV namespacing),
#     min-version gate at latest-lag via gated_rollout_request_hook. Tune
#     rollout_request_weight_version_lag for the staleness/throughput trade-off.
