"""Mixture v3b, second seed: qwen3_8_27b_harbor_mix_v3b_h200x2 with the trainer on one 8 x B300 machine
(qwen3_8_27b_harbor_mix_v2's split: TP 2 x CP 2 x DP 2) and rollout seed 43 (v3b: 42).

It computes in the Modal environment capacity-validation. Its run Volume, base checkpoints and trials Volume stay
in training-team, so evals find it as usual. Own app, Volume, trials Volume, sandbox app and W&B group.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2 as b300
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v3b_h200x2 as v3b
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2 import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v3b-s2"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v3b-s2"
EXPERIMENT_VOLUME_ENV = "training-team"
TASKS_ROOT = v3b.TASKS_ROOT
TRAINER_EXTRA_VOLUMES = {
    **b300.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v3b-s2-trials",
        environment_name=EXPERIMENT_VOLUME_ENV,
        create_if_missing=True,
        version=2,
    ),
}


class _MixV3bS2(b300._MixV2):
    rollout_seed = 43
    wandb_group = "qwen3-8-27b-harbor-mix-v3b-s2"
    prometheus_run_name = wandb_group
    environment = {
        **v3b._MixV3b.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v3b-s2-sandboxes",
            max_seq_len=b300._MixV2.max_seq_len,
            max_output_tokens=b300._MixV2.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
        # capacity-validation uses a newer image builder than this Modal client; the rollout workers build the
        # task images.
        "MODAL_IMAGE_BUILDER_VERSION": "2025.06",
    }


miles = _MixV3bS2(
    **harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING
)
