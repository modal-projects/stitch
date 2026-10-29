"""Mixture v2 on two 8 x H200 machines with a 3x learning rate (3e-6 instead of 1e-6), 155 steps.

Tests whether larger updates move the model more than qwen3_8_27b_harbor_mix_v2_h200x2. Own app, Volume,
trials Volume, sandbox app and W&B group, so it runs next to a v2 run.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2_h200x2 as h200
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2_h200x2 import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v2-lr3x"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v2-lr3x"
TRAINER_EXTRA_VOLUMES = {
    **h200.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v2-lr3x-trials", create_if_missing=True, version=2
    ),
}


class _MixV2Lr3x(h200._MixV2H200x2):
    lr = 3e-6
    num_rollout = 155
    wandb_group = "qwen3-8-27b-harbor-mix-v2-lr3x"
    prometheus_run_name = wandb_group
    environment = {
        **h200._MixV2H200x2.environment,
        **harbor_config.environment(
            tasks_root=h200.TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v2-lr3x-sandboxes",
            max_seq_len=h200._MixV2H200x2.max_seq_len,
            max_output_tokens=h200._MixV2H200x2.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


miles = _MixV2Lr3x(**harbor_config.arguments(h200.TASKS_ROOT), **harbor_config.SCORE_CENTERING)
