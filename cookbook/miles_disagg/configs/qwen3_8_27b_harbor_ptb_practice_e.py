"""20 more RL steps on the post-training practice tasks, started from step 20 of a qwen3_8_27b_harbor_ptb_practice run
(weights only: a fresh optimizer and data order), on a set aimed at shipping a model in time.

Tasks: `ptb-practice-e-train` on the harbor-envs Volume (eval repo envs/posttrain_practice.py): overrun 120 and
deadline 60 (new), repair 60 and posttrain 60 (from the first run's set), each validated by its oracle. Otherwise as
qwen3_8_27b_harbor_ptb_practice.
"""

import modal as modal_sdk

from cookbook.common.constants import START_PATH
from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_ptb_practice as first
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_ptb_practice import *  # noqa: F403

RL_RUN = "8d241608"  # the first practice run
START_VOLUME = modal_sdk.Volume.from_name(
    first.EXPERIMENT_VOLUME_NAME, environment_name=first.s2.EXPERIMENT_VOLUME_ENV
)
BF16_CHECKPOINT_PATH = START_PATH / RL_RUN / first.miles.save_hf.format(rollout_id=first.miles.num_rollout - 1)
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = START_PATH / RL_RUN / "checkpoints"

APP_NAME = "stitch-qwen3-8-27b-ptb-practice-e"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-ptb-practice-e"
TASKS_ROOT = "/harbor-envs/ptb-practice-e-train"
TRAINER_EXTRA_VOLUMES = {
    **first.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-ptb-practice-e-trials",
        environment_name=first.s2.EXPERIMENT_VOLUME_ENV,
        create_if_missing=True,
        version=2,
    ),
}


class _PtbPracticeE(first._PtbPractice):
    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    wandb_group = "qwen3-8-27b-ptb-practice-e"
    prometheus_run_name = wandb_group
    environment = {
        **first._PtbPractice.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-ptb-practice-e-sandboxes",
            max_seq_len=first.s2.b300._MixV2.max_seq_len,
            max_output_tokens=first.s2.b300._MixV2.rollout_max_response_len,
            agent_timeout=900,
            trial_timeout=3600,
            max_turns=200,
        ),
    }


miles = _PtbPracticeE(
    **harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING
)
