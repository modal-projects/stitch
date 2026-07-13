"""Qwen3-4B GRPO with Modal Flash sparse-delta rollout servers."""

from __future__ import annotations

from cookbook.common.config import DATA_PATH, ModalConfig
from cookbook.slime.config import SlimeConfig


APP_NAME = "stitch-qwen3-4b"
DELTA_VOLUME_NAME = "stitch-delta-qwen3-4b"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

# How the rollout sidecar applies published weight versions. "in_place" pauses
# the engine, applies, and resumes — in-flight requests keep decoding on stale
# KV and cross-version isolation comes from extra_key stamping, so commits stop
# blocking behind over-generation/eval stragglers and skip the full-tree flush.
# "quiesce" is the safe fallback that drains in-flight requests before applying.
SIDECAR_COMMIT_MODE = "in_place"

# Log every versioned sidecar proxy request (start/end + injected rid) at INFO,
# so a stuck rollout can be traced hop-by-hop: slime rid -> sidecar -> SGLang.
SIDECAR_DEBUG_REQUESTS = True

# SGLang server tuning, merged over the structural args set in modal_train.py.
SGLANG_SERVER_ARGS = {
    "--reasoning-parser": "qwen3",
    "--context-length": "16384",
    "--mem-fraction-static": "0.84",
    "--chunked-prefill-size": "4096",
    "--max-prefill-tokens": "4096",
}

modal = ModalConfig(gpu="H200")


class _Slime(SlimeConfig):
    # Model
    hf_checkpoint = "Qwen/Qwen3-4B"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"

    # Modal Flash disaggregated rollout through slime's generic HTTP endpoint mode.
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1
    # Publish-only: slime launches no engines and routes /generate to the Modal
    # Flash gateway (filled in at launch); rollouts pull weights from the pool.
    rollout_endpoint_url = None
    custom_rollout_request_hook_path = "cookbook.common.hooks.gated_rollout_request_hook"
    rollout_request_weight_version_mode = "exact"
    rollout_request_weight_version_lag = 0
    rollout_request_retry_attempts = 240
    rollout_request_retry_sleep = 1.0
    # The trainer hits the Modal Flash gateway directly, which routes session
    # affinity on Modal-Session-ID; emit that so GRPO siblings co-locate.
    rollout_session_affinity_header = "Modal-Session-ID"

    # Disk-delta publish-only over the Modal Volume bulletin board: slime writes
    # weight_v{N}/ + a `latest` pointer to update_weight_disk_dir (the Volume),
    # the pre-push hook commits the Volume + wakes the pool, and each sidecar
    # applies the delta host-side via slime's disk_delta. Publish-only is implied
    # by rollout_endpoint_url, so no local-checkpoint dir is required here.
    update_weight_mode = "delta"
    update_weight_transport = "disk"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT
    custom_delta_pre_push_path = "cookbook.common.hooks.commit_and_wake"

    # Data
    prompt_data = f"{DATA_PATH}/gsm8k/train.parquet"
    eval_prompt_data = ["gsm8k", f"{DATA_PATH}/gsm8k/test.parquet"]
    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"

    # Rollout
    rollout_function_path = "slime.rollout.sglang_rollout.generate_rollout"
    num_rollout = 3
    rollout_batch_size = 64
    rollout_max_response_len = 4096
    rollout_temperature = 1.0
    rollout_top_p = 1.0
    n_samples_per_prompt = 8
    global_batch_size = 128
    sglang_server_concurrency = 64
    use_fault_tolerance = False

    # Eval
    eval_interval = None
    n_samples_per_eval_prompt = 4
    eval_max_response_len = 8192
    eval_top_p = 1.0

    # Training
    tensor_model_parallel_size = 1
    sequence_parallel = False
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 9216
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True

    # Optimizer
    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98

    # Algorithm
    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    use_kl_loss = True
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.0

    # Qwen3-4B architecture
    num_layers = 36
    hidden_size = 2560
    ffn_hidden_size = 9728
    num_attention_heads = 32
    group_query_attention = True
    num_query_groups = 8
    kv_channels = 128
    vocab_size = 151936
    normalization = "RMSNorm"
    norm_epsilon = 1e-6
    swiglu = True
    disable_bias_linear = True
    qk_layernorm = True
    use_rotary_position_embeddings = True
    rotary_base = 1000000

    def prepare_data(self) -> None:
        from datasets import load_dataset

        ds = load_dataset("zhuzilin/gsm8k")
        ds["train"].to_parquet(f"{DATA_PATH}/gsm8k/train.parquet")
        ds["test"].to_parquet(f"{DATA_PATH}/gsm8k/test.parquet")


slime = _Slime()
