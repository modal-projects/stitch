"""Qwen3-4B SLIME trainer config for testing the API-shim provider."""

from __future__ import annotations

from cookbook.slime_disagg.configs.base import DATA_PATH, ModalConfig, SlimeConfig


HF_SECRET_NAME = "huggingface-secret"
# The trainer calls the provider shim, so it needs the same optional auth values
# as the provider itself.
SHIM_SECRET_NAME = "stitch-api-shim-provider"
# slime writes weight_v{N}/ + the raw `latest` pointer straight here (the
# mounted S3 transport), in the flat customer layout the provider pool pulls.
TRANSPORT_ROOT = "/mnt/stitch-s3-transport"
# Publish-only drives one opaque HTTP endpoint (the provider front door).
ROLLOUT_NUM_ENGINES = 1

modal = ModalConfig(gpu="H200")


class _Slime(SlimeConfig):
    # Model
    hf_checkpoint = "Qwen/Qwen3-4B"
    ref_load = hf_checkpoint
    megatron_to_hf_mode = "bridge"

    # Opaque rollout provider. slime/modal_train.py fills the endpoint URL from
    # the deployed provider Flash gateway at launch time.
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1
    # Publish-only mode: slime launches no engines and routes /generate to this
    # opaque URL (set at launch from the deployed provider front door).
    rollout_endpoint_url = None
    custom_rollout_request_hook_path = (
        "cookbook.standalone_rollouts.slime.hooks.rollout_request_weight_version_hook"
    )
    api_shim_rollout_request_weight_version_mode = "exact"
    api_shim_rollout_request_version_lag = 0
    api_shim_rollout_request_retry_attempts = 240
    api_shim_rollout_request_retry_sleep = 1.0

    # Disk-delta publish-only: slime writes weight_v{N}/ (+ a `latest` pointer)
    # straight to the mounted S3 transport for the elastic provider pool to pull.
    # rollout_endpoint_url puts slime in publish-only mode, so no local
    # checkpoint dir is required. The pre-push hook calls the customer hot-load
    # API and waits for pool readiness before the next rollout.
    update_weight_mode = "delta"
    update_weight_transport = "disk"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = TRANSPORT_ROOT
    api_shim_transport_root = TRANSPORT_ROOT
    custom_delta_pre_push_path = (
        "cookbook.standalone_rollouts.slime.hooks.announce_and_wait"
    )

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
