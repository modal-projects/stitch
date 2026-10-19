"""Mixture v1b: qwen3_8_27b_harbor_mix_v1 with cyber weighted up; only the task mix and names change.

Tasks: the mix `mixture-v1b` on the harbor-envs Volume, the same 2,933 tasks as `mixture-v1` with
each cyber row 3 times, so 1,347 of 3,831 rows (35%) are cyber (v1: 15%). Miles reads each row as
its own prompt, so a cyber task is drawn 3 times per epoch.

Steps stay at 55, as v1, so both runs make the same number of updates. That is ~0.8 of an epoch
here (estimated: ~58% of groups kept, so ~55 rows per step), not ~1 as in v1.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v1 import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v1 import _MixV1

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v1b"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v1b"
TASKS_ROOT = "/harbor-envs/mixture-v1b"
TRAINER_EXTRA_VOLUMES = {
    "/harbor-envs": modal_sdk.Volume.from_name(
        "harbor-envs", environment_name=harbor_config.DATA_ENV
    ),
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v1b-trials", create_if_missing=True, version=2
    ),
}


class _MixV1b(_MixV1):
    wandb_group = "qwen3-8-27b-harbor-mix-v1b"
    prometheus_run_name = wandb_group
    environment = {
        **_MixV1.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v1b-sandboxes",
            max_seq_len=_MixV1.max_seq_len,
            max_output_tokens=_MixV1.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


miles = _MixV1b(**harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING)
