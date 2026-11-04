"""Mixture v3b on two 8 x H200 machines: qwen3_8_27b_harbor_mix_v3a_h200x2 on the task mix `mixture-v3b`.

mixture-v3b is mixture-v3a's 2,213 tasks and 659 SWE-Gym, R2E-Gym and SWE-rebench bug-fix tasks that the base model
solved 1 to 7 of 8 times. Only the tasks and names change, so the scores compare with v3a from the same start; 55
steps, as v3a. Own app, Volume, trials Volume, sandbox app and W&B group.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2_h200x2 as h200
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2_h200x2 import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v3b"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v3b"
TASKS_ROOT = "/harbor-envs/mixture-v3b"
TRAINER_EXTRA_VOLUMES = {
    **h200.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v3b-trials", create_if_missing=True, version=2
    ),
}


class _MixV3b(h200._MixV2H200x2):
    wandb_group = "qwen3-8-27b-harbor-mix-v3b"
    prometheus_run_name = wandb_group
    environment = {
        **h200._MixV2H200x2.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v3b-sandboxes",
            max_seq_len=h200._MixV2H200x2.max_seq_len,
            max_output_tokens=h200._MixV2H200x2.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


miles = _MixV3b(**harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING)
