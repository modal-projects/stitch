"""Second seed of qwen3_8_27b_harbor_mix_v3b_b300_lr2x: rollout seed 43 (as v3b-s2); only the seed and the names change.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v3b_b300_lr2x as lr2x
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v3b_b300_lr2x import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v3b-lr2x-s2"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v3b-lr2x-s2"
TRAINER_EXTRA_VOLUMES = {
    **lr2x.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v3b-lr2x-s2-trials",
        environment_name=lr2x.EXPERIMENT_VOLUME_ENV,
        create_if_missing=True,
        version=2,
    ),
}


class _MixV3bLr2xS2(lr2x._MixV3bLr2x):
    rollout_seed = 43
    wandb_group = "qwen3-8-27b-harbor-mix-v3b-lr2x-s2"
    prometheus_run_name = wandb_group
    environment = {
        **lr2x._MixV3bLr2x.environment,
        **harbor_config.environment(
            tasks_root=lr2x.TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v3b-lr2x-s2-sandboxes",
            max_seq_len=lr2x.s2.b300._MixV2.max_seq_len,
            max_output_tokens=lr2x.s2.b300._MixV2.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


miles = _MixV3bLr2xS2(
    **harbor_config.arguments(lr2x.TASKS_ROOT), **harbor_config.SCORE_CENTERING
)
