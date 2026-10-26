"""Mixture v2b: mixture v2 with cyber weighted up, as v1b is to v1; only the task mix and names change.

Tasks: the mix `mixture-v2b` on the harbor-envs Volume, the same 2,532 tasks as `mixture-v2` with each
cyber row 3 times, so 1,158 of 3,304 rows (35%) are cyber (v2: 15%).

Steps stay at 55, as v2, so both runs make the same number of updates: ~0.8 of an epoch here.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v1 import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v1 import _MixV1

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v2b"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v2b"
TASKS_ROOT = "/harbor-envs/mixture-v2b"
TRAINER_EXTRA_VOLUMES = {
    "/harbor-envs": modal_sdk.Volume.from_name(
        "harbor-envs", environment_name=harbor_config.DATA_ENV
    ),
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v2b-trials", create_if_missing=True, version=2
    ),
}


class _MixV2b(_MixV1):
    wandb_group = "qwen3-8-27b-harbor-mix-v2b"
    prometheus_run_name = wandb_group
    environment = {
        **_MixV1.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v2b-sandboxes",
            max_seq_len=_MixV1.max_seq_len,
            max_output_tokens=_MixV1.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


miles = _MixV2b(**harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING)
