"""Mixture v3b with a 2x learning rate (2e-6 instead of 1e-6), rollout seed 42: qwen3_8_27b_harbor_mix_v3b_b300_s2 with
only the learning rate, the seed and the names changed. It runs at the normal 16K reply cap, so against v3b and v3b-s2
it measures the learning rate alone (the earlier 2x run also had a 12K reply cap).

It computes in the Modal environment capacity-validation. Its run Volume, base checkpoints and trials Volume stay
in training-team, so evals find it as usual. Own app, Volume, trials Volume, sandbox app and W&B group.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v3b_b300_s2 as s2
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v3b_b300_s2 import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v3b-lr2x"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v3b-lr2x"
TRAINER_EXTRA_VOLUMES = {
    **s2.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v3b-lr2x-trials",
        environment_name=s2.EXPERIMENT_VOLUME_ENV,
        create_if_missing=True,
        version=2,
    ),
}


class _MixV3bLr2x(s2._MixV3bS2):
    lr = 2e-6
    rollout_seed = 42
    wandb_group = "qwen3-8-27b-harbor-mix-v3b-lr2x"
    prometheus_run_name = wandb_group
    environment = {
        **s2._MixV3bS2.environment,
        **harbor_config.environment(
            tasks_root=s2.TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v3b-lr2x-sandboxes",
            max_seq_len=s2.b300._MixV2.max_seq_len,
            max_output_tokens=s2.b300._MixV2.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


miles = _MixV3bLr2x(
    **harbor_config.arguments(s2.TASKS_ROOT), **harbor_config.SCORE_CENTERING
)
