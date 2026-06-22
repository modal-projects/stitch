"""Moonlight-16B-A3B SLIME trainer config for the API-shim provider.

The async + routing-replay analog of cookbook/slime_disagg/configs/moonlight_disagg.py,
but publishing through the standalone provider's customer hot-load API (S3
transport + singleton front door) instead of the in-cluster Modal Volume
bulletin board.

Key difference from slime_disagg's M3: staleness is gated by a PUBLISH-HOOK
readiness BARRIER, not a per-request version pin. `announce_and_wait` copies the
new version to S3, POSTs the hot-load API, and blocks until the front door
reports every live replica ready on the new version (readiness_threshold defaults
to 1.0), so the next rollouts always run on current weights. The request hook
therefore leaves the version gate OFF and only carries auth headers, retries, and
session affinity.
"""

from __future__ import annotations

from cookbook.slime_disagg.configs.base import DATA_PATH, ModalConfig, SlimeConfig


HF_SECRET_NAME = "huggingface-secret"
# The trainer calls the provider shim, so it needs the same optional auth values.
SHIM_SECRET_NAME = "stitch-api-shim-provider"
# The mounted S3 transport the provider pool pulls from (flat customer layout).
TRANSPORT_ROOT = "/mnt/stitch-s3-transport"
# slime's disk-delta writer uses atomic rename (ENOSYS on the S3 mount), so it
# publishes weight_v{N}/ to local disk; announce_and_wait copies each version to
# the transport (PutObject) before signalling the hot-load API.
LOCAL_DELTA_DIR = "/tmp/slime-api-shim-deltas"
ROLLOUT_NUM_ENGINES = 1

modal = ModalConfig(gpu="H200", region="us")


class _Slime(SlimeConfig):
    # Architecture from the slime model script (MLA + DeepSeek-MoE); no inline arch.
    slime_model_script = "scripts/models/moonlight.sh"

    # Model + checkpoint. Bridge-mode HF load (no torch_dist conversion); matches
    # the model the provider pool serves.
    hf_checkpoint = "moonshotai/Moonlight-16B-A3B-Instruct"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"

    # Publish-only rollout through the provider front door (URL filled at launch).
    actor_num_nodes = 2  # 2x8 H200 -> multinode RDMA + expert parallel
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1
    rollout_endpoint_url = None

    # M3 async-first: one-step off-policy (train_async); publish every step.
    async_mode = True
    update_weights_interval = 1

    # Routing replay: replay the rollout engine's per-token expert routing during
    # training. Needs the provider pool's --enable-return-routed-experts
    # (configs/moonlight_hot_load.py); num_layers(27)/moe_router_topk(6) come from
    # the model script so the rollout reshape stays consistent.
    use_rollout_routing_replay = True

    # Staleness is gated by the announce_and_wait readiness BARRIER (the publish
    # hook below), so the request hook leaves the per-request version pin OFF and
    # only carries auth headers, retries, and affinity.
    custom_rollout_request_hook_path = (
        "cookbook.standalone_rollouts.slime.hooks.rollout_request_weight_version_hook"
    )
    api_shim_rollout_request_weight_version_mode = "none"
    api_shim_rollout_request_retry_attempts = 240
    api_shim_rollout_request_retry_sleep = 1.0

    # Disk-delta publish-only: write weight_v{N}/ to local disk; the pre-push hook
    # copies to the S3 transport, POSTs the hot-load API, and blocks until all live
    # replicas report the new version (readiness_threshold 1.0 by default).
    update_weight_mode = "delta"
    update_weight_transport = "disk"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = LOCAL_DELTA_DIR
    api_shim_transport_root = TRANSPORT_ROOT
    custom_delta_pre_push_path = (
        "cookbook.standalone_rollouts.slime.hooks.announce_and_wait"
    )

    # Data: dapo-math-17k (MoE-worthy), matching the slime_disagg Moonlight config.
    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"
    eval_interval = None

    # Rollout
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

    # Trainer parallelism (TP4/EP8 over 2x8; MLA/MoE arch flags from the script).
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

    # Optimizer (CPU offload keeps GPU state tiny for 3B active).
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    # Algorithm (GRPO)
    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.0

    def prepare_data(self) -> None:
        from datasets import load_dataset

        # Raw DAPO-Math-17k has `prompt` (chat list) + reward_model.ground_truth;
        # lift the answer into `label` and subset the ~1.8M-row artifact (see
        # cookbook/slime_disagg/configs/moonlight_disagg.py).
        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


slime = _Slime()
