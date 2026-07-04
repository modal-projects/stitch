"""Kimi K2.6 GRPO on Modal, disaggregated, native-INT4 end to end.

Kimi K2.6 is a ~1T-parameter DeepSeek-V3-architecture MoE (MLA + DeepSeek-MoE:
sigmoid router, shared experts, grouped top-k). It is far too large to serve in
bf16/fp8 on 4xB200 (~2TB / ~1TB vs 768GB), so the rollout pool serves the
model's **native compressed-tensors INT4 (W4A16)** checkpoint (~0.5TB, fits).

That single INT4 format is what makes the disaggregated weight-sync work here.
The bulletin-board path XORs the trainer's exported HF tensors byte-for-byte
against the served base checkpoint, so train and serve must agree on the on-disk
format. We get that with INT4 QAT:

  * the trainer fake-quantizes weights in-loop (OPEN_TRAINING_INT4_FAKE_QAT_FLAG)
    so it learns INT4-robust weights;
  * its megatron->hf export emits native compressed-tensors W4A16 tensors
    (.weight_packed / .weight_scale / .weight_shape, see
    megatron_to_hf/processors/quantizer_compressed_tensors.py);
  * the sidecar XORs those onto a copy of the native-INT4 base and the pool
    reloads it. One format, end to end — no bf16<->FP4 re-quantization step.

INVARIANT: OPEN_TRAINING_INT4_GROUP_SIZE (the QAT simulation grouping) MUST equal
the served checkpoint's compressed-tensors group_size. The export reads the group
size from the base checkpoint's quantization_config; QAT must simulate the same
grouping or the exported codes won't match what the engine loads.

Reshape provenance (colocated scripts/low_precision/run-kimi-k2-Thinking-int4.sh
-> this disagg config):
  ARCHITECTURE -> scripts/models/kimi-k2-thinking.sh via `slime_model_script`
                  (NO arch attrs here — the script is the single source).
  PARALLELISM  -> this config, from the recipe's 32x8 TP8/PP8/CP4/EP32.
  INFERENCE    -> SGLANG_SERVER_ARGS + the dedicated B200 serving image
                  (serving.py); rollout_num_gpus_per_engine=4 (B200:4).
  ALGO/DATA    -> this config (GRPO + TIS, dapo-math-17k).

Deploy as its own app:
    EXPERIMENT_CONFIG=kimi_k2_6_int4_disagg m deploy --strategy recreate -m cookbook.slime_disagg.modal_train

Before running:
  1. hf_checkpoint must resolve to a native compressed-tensors INT4 (W4A16)
     checkpoint whose group_size equals INT4_GROUP_SIZE below
     (moonshotai/Kimi-K2-Thinking is a drop-in swap if no native-INT4 K2.6 is
     published: same arch script, W4A16 at group_size 32).
  2. kimi-k2-thinking.sh must describe K2.6's arch (rope scaling, norm eps); if
     K2.6 diverges, add a kimi-k2.6.sh model script to the slime fork.
  3. The serving image's SGLang must serve native-INT4 MLA MoE on Blackwell
     (verify on a warm container — see serving.py).

The smaller `moonlight_int4_disagg` config is the same machinery at a size that
fits a couple of GPUs — run it first to de-risk the INT4-QAT/disk-delta loop.
"""

from __future__ import annotations

from cookbook.slime_disagg.configs.base import DATA_PATH, ModalConfig, SlimeConfig


APP_NAME = "slime-kimi-k2-6-int4-disagg"
DELTA_VOLUME_NAME = "slime-delta-bulletin-kimi-k2-6-int4"
DELTA_BULLETIN_ROOT = "/delta-bulletin"

# QAT grouping; MUST match the served INT4 checkpoint's compressed-tensors
# group_size (see the INVARIANT in the module docstring).
INT4_GROUP_SIZE = "32"

# Async one-step off-policy: in_place applies weights without draining in-flight
# rollouts; stale-version KV is isolated per weight version by the sidecar's
# extra_key stamping and drains as those requests finish. min-version pins cross
# commits freely; only exact pins are quiesced.
SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_DEBUG_REQUESTS = True


def build_serving_image(**kwargs):
    """Per-experiment rollout-pool image (modal_train picks this up if present).

    Lazy import keeps this config module importable without the modal SDK (it is
    resolved locally by launch_train and in unit tests). modal_train passes the
    slime ref / root / cache path so the pool pins the trainer's exact slime.
    """
    from cookbook.slime_disagg.serving import build_int4_b200_serving_image

    return build_int4_b200_serving_image(**kwargs)


# Kimi K2.6 on the elastic B200:4 pool, serving the native compressed-tensors
# INT4 checkpoint. The trainer image injects --served-model-name / --dtype /
# --cuda-graph-max-bs / --max-running-requests / --trust-remote-code, so only the
# Kimi/MLA/cache extras live here. NOTE: no --quantization flag — INT4 is driven
# by the checkpoint's own compressed-tensors config. mem-fraction-static and
# context-length are STARTING POINTS; measure on a warm B200:4 and adjust.
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
    # Hierarchical KV cache (tuned in the standalone B200 deployment).
    "--enable-hierarchical-cache": "",
    "--hicache-ratio": "2",
    "--hicache-io-backend": "kernel",
    "--hicache-mem-layout": "page_first",
    "--hicache-write-policy": "write_through",
    "--skip-server-warmup": "",
    # Routing replay: the pool must emit per-token routed experts so the trainer
    # can replay them. slime launches no engine in publish-only mode, so this is
    # set here (DeepSeek-V3-arch MoE supports it).
    "--enable-return-routed-experts": "",
}

# B200 in us-west (Blackwell availability), matching the standalone serve recipe.
# rollout_min_containers is a warm floor only; Modal scales replicas up to meet
# target_inputs (= sglang_server_concurrency), converging on the count that
# saturates the trainer. Raise the floor once that count is known.
modal = ModalConfig(
    gpu="B200",
    region="us",
    rollout_min_containers=2,
    proxy_regions=["us-west"],
)


class _Slime(SlimeConfig):
    # Architecture is sourced from the model script; do NOT inline arch attrs
    # (the script carries MLA + the full DeepSeek-MoE arg set). MLA models must
    # not set --attention-backend flash, so it is omitted (see the recipe note).
    slime_model_script = "scripts/models/kimi-k2-thinking.sh"

    # Model + checkpoint: the native-INT4 base is the served model, the QAT init,
    # and the disk-delta base — all the same checkpoint (see prerequisite 1).
    hf_checkpoint = "moonshotai/Kimi-K2.6"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"

    # Disaggregated publish-only rollout through slime's opaque HTTP endpoint;
    # modal_train fills rollout_endpoint_url from the Flash gateway at launch.
    actor_num_nodes = 32  # 32x8 H200 = 256 GPUs (the recipe's trainer footprint)
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 4  # B200:4 per rollout container (native INT4 fits)
    rollout_endpoint_url = None
    # Staleness gate: each rollout is pinned to min_required_version = latest - lag,
    # so a replica more than `lag` versions behind 409s (-> retried, and nudged to
    # sync forward) rather than generating too-stale rollouts.
    custom_rollout_request_hook_path = "cookbook.slime_disagg.hooks.gated_rollout_request_hook"
    rollout_request_weight_version_mode = "min"
    rollout_request_weight_version_lag = 1  # bounded staleness window (raise if 409 retries bubble)
    rollout_request_retry_attempts = 240
    rollout_request_retry_sleep = 1.0
    rollout_session_affinity_header = "Modal-Session-ID"

    # Async-first: train_async pipelines generate(N+1) with train(N); publish
    # every step. For a model this large, raise update_weights_interval if the
    # per-step export+delta+reload cost dominates.
    async_mode = True
    update_weights_interval = 1

    # Disk-delta publish-only over the Modal Volume bulletin board. The export
    # emits native compressed-tensors INT4, so the XOR delta is byte-exact
    # against the native-INT4 base.
    update_weight_mode = "delta"
    update_weight_transport = "disk"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT
    custom_delta_pre_push_path = "cookbook.slime_disagg.hooks.commit_and_wake"

    # Data: dapo-math-17k, the hard math-reasoning set the proven Kimi recipe uses.
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"
    eval_interval = None  # skip eval during bring-up (see the aime note below)

    # Rollout. Production-leaning throughput knobs from the recipe; num_rollout is
    # the bring-up smoke length — scale it up for a real run.
    rollout_function_path = "slime.rollout.sglang_rollout.generate_rollout"
    num_rollout = 20
    rollout_batch_size = 128
    rollout_max_response_len = 16384
    rollout_temperature = 0.8
    rollout_top_p = 1.0
    n_samples_per_prompt = 8
    over_sampling_batch_size = 256
    dynamic_sampling_filter_path = (
        "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    )
    num_steps_per_rollout = 4
    balance_data = True
    # sglang_server_concurrency is THE knob to size for B200:4 saturation: it sets
    # the rollout pool's target_inputs (Modal autoscaling) and SGLang's
    # cuda-graph-max-bs / max-running-requests. Tune for max throughput per replica.
    sglang_server_concurrency = 256
    use_fault_tolerance = False

    # Trainer parallelism from the recipe (world = TP8 * PP8 * CP4 = 256 = 32x8;
    # EP32 over the expert region). decoder_last_pipeline_num_layers balances the
    # 61-layer model over 8 pipeline stages. Arch flags come from the model script.
    tensor_model_parallel_size = 8
    pipeline_model_parallel_size = 8
    context_parallel_size = 4
    expert_model_parallel_size = 32
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 5
    sequence_parallel = True
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

    # Optimizer (from the recipe; CPU offload keeps GPU state tiny for ~32B active).
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    # Algorithm (GRPO + truncated importance sampling, from the recipe).
    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.0
    use_tis = True

    # Routing replay: replay the rollout engine's per-token expert routing during
    # the training forward/backward to cut train-inference divergence. Needs the
    # pool's --enable-return-routed-experts (in SGLANG_SERVER_ARGS above).
    use_rollout_routing_replay = True

    # Ray runtime environment: base flags + INT4 QAT. NCCL timeout is generous for
    # the large all-to-all expert traffic.
    environment = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NVSHMEM_DISABLE_NCCL": "1",
        "NCCL_TIMEOUT_MS": "360000000",
        "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
        "OPEN_TRAINING_INT4_GROUP_SIZE": INT4_GROUP_SIZE,
    }

    def prepare_data(self) -> None:
        from datasets import load_dataset

        # Raw DAPO-Math-17k columns are `prompt` (a chat list) and the gold answer
        # under `reward_model.ground_truth` — there is no `label`. slime reads
        # --input-key prompt (+ apply_chat_template) and --label-key label, so lift
        # the answer into a `label` column. A shuffled subset is ample for a run.
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")

    # To add the recipe's aime eval: fetch aime-2024 into {prompt, label} jsonl in
    # prepare_data, then set eval_interval (e.g. 20), eval_prompt_data =
    # ["aime", f"{DATA_PATH}/aime-2024.jsonl"], n_samples_per_eval_prompt = 16,
    # eval_max_response_len = 16384, eval_top_p = 0.7.


slime = _Slime()
