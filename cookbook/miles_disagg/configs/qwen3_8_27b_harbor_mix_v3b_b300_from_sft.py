"""Mixture v3b from the SFT warm start: qwen3_8_27b_harbor_mix_v3b_b300_s2 started from an SFT run's weights
(qwen3_8_27b_sft_warmstart) instead of Qwen3.8-27B. Only the starting checkpoint and the names change, so it compares
with v3b-s2 (same seed and trainer machine).

The SFT run Volume mounts at START_PATH on the trainer and the rollout servers: its final HF export serves and is the
trainer's hf_checkpoint, and its Megatron checkpoint is ref_load. Placeholder: set SFT_RUN to the SFT run's id.
"""

import modal as modal_sdk

from cookbook.common.constants import START_PATH
from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v3b_b300_s2 as s2
from cookbook.miles_disagg.configs import qwen3_8_27b_sft_warmstart as sft_run
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v3b_b300_s2 import *  # noqa: F403

SFT_RUN = "SFT_RUN_ID"  # placeholder
START_VOLUME = modal_sdk.Volume.from_name(
    sft_run.EXPERIMENT_VOLUME_NAME, environment_name=sft_run.EXPERIMENT_VOLUME_ENV
)
BF16_CHECKPOINT_PATH = START_PATH / SFT_RUN / sft_run.miles.save_hf.format(rollout_id=sft_run.STEPS - 1)
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = START_PATH / SFT_RUN / "checkpoints"

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v3b-from-sft"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v3b-from-sft"
TRAINER_EXTRA_VOLUMES = {
    **s2.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v3b-from-sft-trials",
        environment_name=s2.EXPERIMENT_VOLUME_ENV,
        create_if_missing=True,
        version=2,
    ),
}


class _MixV3bFromSft(s2._MixV3bS2):
    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    wandb_group = "qwen3-8-27b-harbor-mix-v3b-from-sft"
    prometheus_run_name = wandb_group
    environment = {
        **s2._MixV3bS2.environment,
        **harbor_config.environment(
            tasks_root=s2.TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v3b-from-sft-sandboxes",
            max_seq_len=s2.b300._MixV2.max_seq_len,
            max_output_tokens=s2.b300._MixV2.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


miles = _MixV3bFromSft(
    **harbor_config.arguments(s2.TASKS_ROOT), **harbor_config.SCORE_CENTERING
)
