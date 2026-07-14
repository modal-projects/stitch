"""Moonlight-16B-A3B GRPO on Modal Flash, disaggregated — the cheap rung of the K2.6 ladder.

Deploy: EXPERIMENT_CONFIG=moonlight m deploy --strategy recreate -m cookbook.slime_disagg.app
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import DATA_PATH
from cookbook.slime_disagg.config import SlimeConfig


APP_NAME = "stitch-moonlight"
DELTA_VOLUME_NAME = "stitch-delta-moonlight"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

# in_place applies weights without draining in-flight rollouts; stale KV isolated per version.
SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_DEBUG_REQUESTS = True

# mem-fraction-static is a STARTING POINT -- measure on a warm container and adjust.
SGLANG_SERVER_ARGS = {
    "--weight-loader-prefetch-checkpoints": "",
    "--weight-loader-prefetch-num-threads": "8",
    "--context-length": "8192",
    "--mem-fraction-static": "0.85",
    # routing replay: set here since slime launches no engine in publish-only mode.
    "--enable-return-routed-experts": "",
}

modal = ModalConfig(gpu="H200", region="us")


class _Slime(SlimeConfig):
    # Arch comes from the model script; do NOT inline arch attrs here.
    slime_model_script = "scripts/models/moonlight.sh"

    hf_checkpoint = "moonshotai/Moonlight-16B-A3B-Instruct"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"

    actor_num_nodes = 2  # 2x8 H200 -> exercises multinode RDMA + expert parallel
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1  # 1xH200 per rollout container (MLA -> cheap KV)
    rollout_endpoint_url = None
    # Staleness gate: pin each request to latest_published - lag; over-stale replicas 409 -> retry.
    custom_rollout_request_hook_path = "cookbook.common.hooks.gated_rollout_request_hook"
    rollout_request_weight_version_mode = "min"
    rollout_request_weight_version_lag = 1  # k: bounded staleness window (tune up if 409 retries bubble)
    rollout_request_retry_attempts = 240
    rollout_request_retry_sleep = 1.0
    rollout_session_affinity_header = "Modal-Session-ID"

    # async-first: one-step off-policy; publish weights every step.
    async_mode = True
    update_weights_interval = 1

    # disk-delta publish-only (export uses convert_deepseekv3_to_hf for this arch).
    update_weight_mode = "delta"
    update_weight_transport = "disk"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT
    custom_delta_pre_push_path = "cookbook.common.hooks.commit_and_wake"

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"
    eval_interval = None  # skip eval during bring-up

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

    # Trainer parallelism scaled from the recipe's TP4/EP8 to 2x8 (world = TP4*DP4 = 16).
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

    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.0

    # R3 (arxiv 2510.11370): replay the rollout engine's expert routing in the train forward.
    use_rollout_routing_replay = True

    def prepare_data(self) -> None:
        from datasets import load_dataset

        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


slime = _Slime()
