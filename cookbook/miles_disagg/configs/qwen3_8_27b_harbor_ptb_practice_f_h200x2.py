"""20 more RL steps on the post-training practice tasks from step 20 of the first practice run (weights only), on a
set aimed at shipping a model early and improving it within a budget, with the trainer on two 8 x H200 machines as
qwen3_8_27b_harbor_ptb_practice_e_h200x2. Only the task set, the names and the agent limit (the longest tasks take 30
minutes) differ from that recipe.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_ptb_practice_e_h200x2 import *  # noqa: F403
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_ptb_practice_e_h200x2 as eh  # after the star import

APP_NAME = "stitch-qwen3-8-27b-ptb-practice-f"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-ptb-practice-f"
TASKS_ROOT = "/harbor-envs/ptb-practice-f2-train"
TRAINER_EXTRA_VOLUMES = {
    **eh.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-ptb-practice-f-trials",
        environment_name=eh.e.first.s2.EXPERIMENT_VOLUME_ENV,
        create_if_missing=True,
        version=2,
    ),
}


class _PtbPracticeF(eh._PtbPracticeEH200x2):
    wandb_group = "qwen3-8-27b-ptb-practice-f"
    prometheus_run_name = wandb_group
    environment = {
        **eh._PtbPracticeEH200x2.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-ptb-practice-f-sandboxes",
            max_seq_len=eh.e.first.s2.b300._MixV2.max_seq_len,
            max_output_tokens=eh.e.first.s2.b300._MixV2.rollout_max_response_len,
            agent_timeout=960,  # x 2.0 = 32 minutes: the 30-minute tasks fit
            trial_timeout=3600,
            max_turns=200,
        ),
    }


miles = _PtbPracticeF(**harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING)
