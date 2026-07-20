"""Moonlight-16B-A3B GRPO on Modal, disaggregated, native-NVFP4 end to end — the small runnable de-risk for kimi_k2_6_nvfp4.

Deploy: EXPERIMENT_CONFIG=moonlight_nvfp4 uv run --extra modal modal deploy --strategy recreate -m cookbook.miles_disagg.app
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import DATA_PATH, PREP_PATH
from cookbook.miles_disagg.config import MilesConfig


APP_NAME = "stitch-moonlight-nvfp4"
DELTA_VOLUME_NAME = "stitch-delta-moonlight-nvfp4"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

SOURCE_MODEL = "moonshotai/Moonlight-16B-A3B-Instruct"
MODEL_TAG = "moonlight-16b-nvfp4"

# in_place applies weights without draining in-flight rollouts; stale KV isolated per version.
SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
# R3 routing-replay needs the dropless Megatron dispatch fix at startup.
MEGATRON_RUNTIME_PATCHES = [
    "/root/cookbook/miles_disagg/patches/megatron-r3-dispatch.patch",
]


# No --quantization flag — NVFP4 comes from the served checkpoint's quant config.
# mem-fraction / context-length are starting points; measure.
SGLANG_SERVER_ARGS = {
    # fastsafetensors: per-rank O_DIRECT read (~1/tp bytes/rank), no gVisor mmap tax; reload inherits it. nogds set in image.
    "--load-format": "fastsafetensors",
    "--attention-backend": "tokenspeed_mla",
    "--kv-cache-dtype": "fp8_e4m3",  # tokenspeed_mla requires this
    "--context-length": "8192",  # Moonlight's max_position_embeddings
    "--mem-fraction-static": "0.8",
    "--chunked-prefill-size": "4096",
    "--skip-server-warmup": "",
    # routing replay: pool emits per-token routed experts for the trainer to replay.
    "--enable-return-routed-experts": "",
}

SGLANG_ENV = {"SGLANG_ENABLE_RELOAD_LOAD_PLAN": "1"}  # NVFP4: load-plan replay + O(delta) partial reload

modal = ModalConfig(
    gpu="B200",
    region="us",
    # warm floor of 1 so the pool is up before the trainer sends rollouts; Flash scales above under load.
    rollout_min_containers=1,
    proxy_regions=["us-west"],
)


class _Miles(MilesConfig):
    # Arch comes from the model script; do NOT inline arch attrs here.
    miles_model_script = "scripts/models/moonlight.sh"

    hf_checkpoint = f"{PREP_PATH}/{MODEL_TAG}/nvfp4"
    ref_load = f"{PREP_PATH}/{MODEL_TAG}/bf16"
    megatron_to_hf_mode = "bridge"
    model_name = "deepseekv3"  # bridge dispatch: Moonlight is DeepSeek-V3 arch

    actor_num_nodes = 1
    actor_num_gpus_per_node = 4  # 1 node x 4 B200 trainer (matches the proven moonlight recipe)
    num_gpus_per_node = 4
    colocate = False  # disk-delta is incompatible with --colocate
    rollout_num_gpus = 0  # publish-only forces this
    rollout_num_gpus_per_engine = 1  # B200:1 per rollout container (Moonlight NVFP4 is tiny)
    rollout_endpoint_url = None
    use_miles_router = True

    # Staleness gate; the knobs ride in custom_config_path (read by the hook, not miles core).
    custom_rollout_request_hook_path = "cookbook.common.hooks.gated_rollout_request_hook"
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        "rollout_request_retry_attempts": 240,
        "rollout_request_retry_sleep": 1.0,
        "rollout_session_affinity_header": "Modal-Session-ID",
    }

    async_mode = True
    update_weights_interval = 1

    # NVFP4 QAT (native Megatron FP4; Blackwell + TE >= 2.7.0.dev0).
    fp4_format = "e2m1"
    fp4_param_gather = False  # True crashes Megatron DDP (TE NVFP4Tensor params)

    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT  # modal_train run-scopes this
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"
    eval_interval = None

    num_rollout = 20
    save_interval = 1000  # megatron requires it; > num_rollout so the smoke skips megatron saves
    rollout_batch_size = 32
    rollout_max_response_len = 4096  # fits within the 8192 context (prompt + response)
    rollout_temperature = 0.8
    n_samples_per_prompt = 8
    global_batch_size = 128
    use_dynamic_global_batch_size = True
    sglang_server_concurrency = 64

    # R3: replay sglang's routed experts in the train/log-prob forward.
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
    }

    def prepare_data(self) -> None:
        from datasets import load_dataset

        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


miles = _Miles()
