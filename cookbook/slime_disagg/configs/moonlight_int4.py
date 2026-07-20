"""Moonlight-16B-A3B GRPO on Modal, disaggregated, native-INT4 end to end — the cheap de-risk for kimi_k2_6_int4.

INVARIANT: OPEN_TRAINING_INT4_GROUP_SIZE MUST equal the served checkpoint's compressed-tensors group_size (128 here).

Deploy: EXPERIMENT_CONFIG=moonlight_int4 m deploy --strategy recreate -m cookbook.slime_disagg.app
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import DATA_PATH
from cookbook.slime_disagg.config import SlimeConfig


APP_NAME = "stitch-moonlight-int4"
DELTA_VOLUME_NAME = "stitch-delta-moonlight-int4"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

INT4_GROUP_SIZE = "128"

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False

# The pool reuses the trainer image (its SGLang serves native INT4; no Blackwell fork).
SGLANG_SERVER_ARGS = {
    # fastsafetensors: per-rank O_DIRECT read (~1/tp bytes/rank), no gVisor mmap tax; reload inherits it. nogds set in image.
    "--load-format": "fastsafetensors",
    "--context-length": "16384",
    "--mem-fraction-static": "0.85",
    "--enable-return-routed-experts": "",  # routing replay
}

modal = ModalConfig(gpu="H200", region="us")


class _Slime(SlimeConfig):
    # Arch comes from the model script. MLA models must not set --attention-backend flash.
    slime_model_script = "scripts/models/moonlight.sh"

    # The native-INT4 base is the served model, the QAT init, and the disk-delta base.
    hf_checkpoint = "moonshotai/Moonlight-16B-A3B-Instruct-INT4"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"

    actor_num_nodes = 1  # 1x8 H200 (the recipe trains on 1x4; 1 node here)
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1  # 1xH200 per rollout container (16B INT4 fits)
    rollout_endpoint_url = None
    custom_rollout_request_hook_path = "cookbook.common.hooks.gated_rollout_request_hook"
    rollout_request_weight_version_mode = "min"
    rollout_request_weight_version_lag = 1
    rollout_request_retry_attempts = 240
    rollout_request_retry_sleep = 1.0
    rollout_session_affinity_header = "Modal-Session-ID"

    async_mode = True
    update_weights_interval = 1

    # disk-delta: export emits native INT4, so the XOR delta is byte-exact against the base.
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
    eval_interval = None

    rollout_function_path = "slime.rollout.sglang_rollout.generate_rollout"
    num_rollout = 10
    rollout_batch_size = 64
    rollout_max_response_len = 8192
    rollout_temperature = 0.8
    rollout_top_p = 1.0
    n_samples_per_prompt = 8
    over_sampling_batch_size = 128
    dynamic_sampling_filter_path = (
        "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    )
    num_steps_per_rollout = 4
    balance_data = True
    sglang_server_concurrency = 64
    use_fault_tolerance = False

    # Trainer parallelism scaled to 1x8 (world = TP4 * DP2 = 8; EP8 over experts).
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

    use_rollout_routing_replay = True

    environment = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NVSHMEM_DISABLE_NCCL": "1",
        "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
        "OPEN_TRAINING_INT4_GROUP_SIZE": INT4_GROUP_SIZE,
    }

    def prepare_data(self) -> None:
        from datasets import load_dataset

        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


slime = _Slime()
