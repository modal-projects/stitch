"""Qwen3-4B GRPO with Modal Flash sparse-delta rollout servers."""

from __future__ import annotations

from configs.base import DATA_PATH, ModalConfig, SlimeConfig


DELTA_VOLUME_NAME = "slime-delta-bulletin-qwen3-4b"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
DELTA_VERSION_DIR = f"{DELTA_BULLETIN_ROOT}/versions"
SGLANG_SERVER_ARGS = {
    "--reasoning-parser": "qwen3",
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
    rollout_num_gpus = 2
    rollout_num_gpus_per_engine = 1
    rollout_http_endpoint_url = None
    rollout_http_endpoint_abort_strategy = "cancel-only"
    rollout_weight_version_policy = "exact-rollout-id"
    rollout_weight_version_retry_attempts = 240
    rollout_weight_version_retry_sleep = 1.0

    # Sparse delta disk transport over the Modal Volume bulletin board.
    update_weight_mode = "delta"
    update_weight_transport = "disk"
    update_weight_encoding = "deltas_zstd"
    update_weight_delta_dir = DELTA_VERSION_DIR
    update_weight_delta_root = DELTA_BULLETIN_ROOT
    update_weight_delta_keep_files = True
    update_weight_delta_publish_only = True
    custom_delta_pre_push_path = "stitch.trainers.slime.commit_delta_volume"
    custom_delta_publish_path = "stitch.trainers.slime.publish_delta_version"
    sglang_update_weight_delta_chunk_bytes = 1024 * 1024 * 1024
    sglang_update_weight_delta_read_workers = 8

    # Data
    prompt_data = f"{DATA_PATH}/gsm8k/train.parquet"
    eval_prompt_data = ["gsm8k", f"{DATA_PATH}/gsm8k/test.parquet"]
    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    rm_type = "math"

    # Rollout
    rollout_function_path = "stitch.trainers.slime.generate_rollout"
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
