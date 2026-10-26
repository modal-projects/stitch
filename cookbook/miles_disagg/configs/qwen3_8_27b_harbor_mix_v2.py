"""Mixture v2: qwen3_8_27b_harbor_mix_v1 on a filtered task mix; only the task mix and names change.

Tasks: the mix `mixture-v2` on the harbor-envs Volume (cyber and code, once per epoch). Mixture v2 is
mixture v1 without the 401 tasks that gave no learning signal in the mix v1 runs (all correct or all
wrong in every group, or no model calls), so 2,532 tasks: 386 cyber, 2,146 code.

Steps stay at 55, as v1, so the runs make the same number of updates: about one epoch here
(estimated: ~70% of groups kept, so ~46 rows per step).
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v1 import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v1 import _MixV1

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v2"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v2"
TASKS_ROOT = "/harbor-envs/mixture-v2"
TRAINER_EXTRA_VOLUMES = {
    "/harbor-envs": modal_sdk.Volume.from_name(
        "harbor-envs", environment_name=harbor_config.DATA_ENV
    ),
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v2-trials", create_if_missing=True, version=2
    ),
}


class _MixV2(_MixV1):
    wandb_group = "qwen3-8-27b-harbor-mix-v2"
    prometheus_run_name = wandb_group
    environment = {
        **_MixV1.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v2-sandboxes",
            max_seq_len=_MixV1.max_seq_len,
            max_output_tokens=_MixV1.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


miles = _MixV2(**harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING)
