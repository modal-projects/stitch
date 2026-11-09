"""SFT that teaches Qwen3.8-27B to write chat-control strings ("<|im_end|>", "<|endoftext|>", "<|im_start|>") as
plain tokens, so that writing them no longer ends its own reply.

Trainer only, as qwen3_8_27b_sft_warmstart (the same model, split and memory settings). The data is the model's own
replies to requests that need these strings written out, generated past the quoted end tokens and cut at the real
end of the turn; sft_plain_tokens.generate_rollout trains the strings inside assistant messages as plain tokens and
keeps the turn markers. One epoch at a low peak learning rate, one HF export at the end.

    EXPERIMENT_CONFIG=qwen3_8_27b_sft_plain_tokens MODAL_ENVIRONMENT=<env> \
        uv run --extra modal modal run --detach -m cookbook.miles_disagg.sft_app
"""

from cookbook.miles_disagg.configs import qwen3_8_27b_sft_warmstart as warm

APP_NAME = "stitch-qwen3-8-27b-sft-plain-tokens"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-sft-plain-tokens"
EXPERIMENT_VOLUME_ENV = warm.EXPERIMENT_VOLUME_ENV
MILES_IMAGE_TAG = warm.MILES_IMAGE_TAG
MILES_REPO_REF = warm.MILES_REPO_REF
modal = warm.modal
TRAINER_EXTRA_VOLUMES = warm.TRAINER_EXTRA_VOLUMES
DATA = "/sft-data/plain-tokens-v1/train.jsonl"
DATA_ROWS = 852  # rows in DATA (its manifest.json)
BATCH = 32
STEPS = DATA_ROWS // BATCH  # one epoch


class _Sft(warm._Sft):
    # /root holds the mounted cookbook package, for the rollout function below
    environment = {**warm._Sft.environment, "PYTHONPATH": "/root/Megatron-LM:/root/miles:/root"}


miles = _Sft(
    **{name: getattr(warm.miles, name) for name in warm.SHARED},
    debug_train_only=True,
    rollout_function_path="cookbook.miles_disagg.sft_plain_tokens.generate_rollout",
    prompt_data=DATA,
    input_key="messages",
    rollout_shuffle=True,
    chat_template_path=warm.sft.CHAT_TEMPLATE_IN_IMAGE,
    loss_mask_type="qwen3",
    loss_type="sft_loss",
    calculate_per_token_loss=True,
    disable_compute_advantages_and_returns=True,
    num_rollout=STEPS,
    rollout_batch_size=BATCH,
    global_batch_size=BATCH,
    max_tokens_per_gpu=32_768,
    lr=5e-6,
    lr_decay_style="cosine",
    min_lr=5e-7,
    lr_warmup_fraction=0.1,
    save_interval=STEPS,
    save_hf="hf_checkpoints/step_{rollout_id:06d}",
    wandb_group="qwen3-8-27b-sft-plain-tokens",
)
