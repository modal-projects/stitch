"""Kimi K2.6 GRPO on Modal, disaggregated, native-INT4 end to end.

Deploy: EXPERIMENT_CONFIG=kimi_k2_6_int4 uv run --extra modal modal deploy -m cookbook.slime_disagg.app
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import DATA_PATH
from cookbook.slime_disagg.config import SlimeConfig

APP_NAME = "stitch-kimi-k2-6-int4"
DELTA_VOLUME_NAME = "stitch-delta-kimi-k2-6-int4"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

# QAT grouping; MUST match the served INT4 checkpoint's compressed-tensors group_size.
INT4_GROUP_SIZE = "32"

# in_place applies weights without draining in-flight rollouts; stale KV isolated per version.
SIDECAR_COMMIT_MODE = "in_place"

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
    # Routing replay: the pool emits per-token routed experts so the trainer can replay them.
    "--enable-return-routed-experts": "",
}

modal = ModalConfig(
    gpu="B200",
    region="us",
    rollout_min_containers=2,
    rollout_target_inputs=256,
    proxy_regions=["us-west"],
)


class _Slime(SlimeConfig):
    # Arch comes from the model script.
    slime_model_script = "scripts/models/kimi-k2-thinking.sh"

    # The native-INT4 base is the served model, the QAT init, and the disk-delta base.
    hf_checkpoint = "moonshotai/Kimi-K2.6"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"

    actor_num_nodes = 32  # 32x8 = 256 GPUs
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0                 # external rollout: the framework runs no local engines
    rollout_num_gpus_per_engine = 4      # B200:4 per rollout container (native INT4 fits)
    rollout_endpoint_url = None          # filled at launch from the pool gateway

    # The three plug points stitch fills (slime's publish hook key is custom_delta_pre_push_path):
    custom_rollout_request_hook_path = "cookbook.common.hooks.gated_rollout_request_hook"
    custom_delta_pre_push_path = "cookbook.common.hooks.commit_and_wake"
    rollout_request_weight_version_mode = "min"
    rollout_request_weight_version_lag = 1  # bounded staleness window
    rollout_request_retry_attempts = 240
    rollout_request_retry_sleep = 1.0
    rollout_session_affinity_header = "Modal-Session-ID"

    async_mode = True
    update_weights_interval = 1
    update_weight_mode = "delta"
    update_weight_transport = "disk"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT  # run-scoped at launch to <root>/<run_id>

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"
    eval_interval = None

    rollout_function_path = "slime.rollout.sglang_rollout.generate_rollout"
    num_rollout = 20
    rollout_batch_size = 128
    rollout_max_response_len = 16384
    rollout_temperature = 0.8
    rollout_top_p = 1.0
    n_samples_per_prompt = 8
    over_sampling_batch_size = 256
    dynamic_sampling_filter_path = "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    num_steps_per_rollout = 4
    balance_data = True
    sglang_server_concurrency = 256
    use_fault_tolerance = False

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

    use_rollout_routing_replay = True  # needs the pool's --enable-return-routed-experts

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

        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


slime = _Slime()
