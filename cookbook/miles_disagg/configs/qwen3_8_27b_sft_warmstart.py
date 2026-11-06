"""SFT warm start for Qwen3.8-27B on Terminus-2 agent trajectories, before RL (..._v3b_b300_from_sft).

Trainer only (cookbook.miles_disagg.sft_app): Miles' SFT path (--loss-type sft_loss, miles.rollout.sft_rollout) with
--debug-train-only starts no SGLang engine, and the app deploys no rollout pool. Full weights on one 8 x B300
machine with the RL recipe's model, checkpoints, split (TP 2 x CP 2 x DP 2) and memory settings. One epoch of the
data set (cookbook.miles_disagg.sft_data), 128 trajectories per step, peak learning rate 1e-5 (10% warmup, cosine
decay to 1e-6), HF exports at 50% and 100% of the steps. Messages render with sft.CHAT_TEMPLATE, as in RL.

    EXPERIMENT_CONFIG=qwen3_8_27b_sft_warmstart MODAL_ENVIRONMENT=<env> \
        uv run --extra modal modal run --detach -m cookbook.miles_disagg.sft_app
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config, sft
from cookbook.miles_disagg.config import MilesConfig
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v3b_b300_s2 as rl

APP_NAME = "stitch-qwen3-8-27b-sft-warmstart"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-sft-warmstart"
EXPERIMENT_VOLUME_ENV = rl.EXPERIMENT_VOLUME_ENV  # with the base checkpoints and the RL runs
MILES_IMAGE_TAG = harbor_config.MILES_IMAGE_TAG
MILES_REPO_REF = harbor_config.MILES_REPO_REF
modal = rl.modal
TRAINER_EXTRA_VOLUMES = {
    "/sft-data": modal_sdk.Volume.from_name("sft-data", environment_name=harbor_config.DATA_ENV)
}
DATA = "/sft-data/warmstart-v1/train.jsonl"
DATA_ROWS = 20_000  # trajectories in DATA (its manifest.json)
BATCH = 128
STEPS = DATA_ROWS // BATCH  # one epoch
# The RL recipe's model, checkpoint, split and memory settings; its rollout, reward and weight-sync settings stay out.
SHARED = (
    "hf_checkpoint",
    "ref_load",
    "megatron_to_hf_mode",
    "model_name",
    "mtp_num_layers",
    "transformer_impl",
    "bf16",
    "hf_export_source_tensor_prefixes",
    "max_seq_len",
    "actor_num_nodes",
    "actor_num_gpus_per_node",
    "num_gpus_per_node",
    "colocate",
    "tensor_model_parallel_size",
    "sequence_parallel",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "distributed_timeout_minutes",
    "use_dynamic_batch_size",
    "max_tokens_per_gpu",
    "log_probs_chunk_size",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "attention_dropout",
    "hidden_dropout",
    "attention_softmax_in_fp32",
    "attention_backend",
    "train_backend",
    "grad_reduce_in_bf16",
    "optimizer_cpu_offload",
    "overlap_cpu_optimizer_d2h_h2d",
    "use_precision_aware_optimizer",
    "optimizer",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "use_wandb",
    "wandb_project",
    "disable_wandb_random_suffix",
)


class _Sft(MilesConfig):
    megatron_model_type = rl.miles.megatron_model_type
    async_mode = True  # train_async.py, as Miles' own SFT script
    environment = {
        "PYTHONPATH": "/root/Megatron-LM:/root/miles",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


miles = _Sft(
    **{name: getattr(rl.miles, name) for name in SHARED},
    debug_train_only=True,
    rollout_function_path="miles.rollout.sft_rollout.generate_rollout",
    prompt_data=DATA,
    input_key="messages",
    rollout_shuffle=True,
    chat_template_path=sft.CHAT_TEMPLATE_IN_IMAGE,
    loss_mask_type="qwen3",
    loss_type="sft_loss",
    calculate_per_token_loss=True,
    disable_compute_advantages_and_returns=True,
    num_rollout=STEPS,
    rollout_batch_size=BATCH,
    global_batch_size=BATCH,
    lr=1e-5,
    lr_decay_style="cosine",
    min_lr=1e-6,
    lr_warmup_fraction=0.1,
    # Miles also saves at the last step.
    save_interval=-(-STEPS // 2),
    save_hf="hf_checkpoints/step_{rollout_id:06d}",
    wandb_group="qwen3-8-27b-sft-warmstart",
)
