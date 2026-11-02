"""qwen3_8_27b_harbor_mix_v2_h200x2_reply12k with a 2x learning rate (2e-6 instead of 1e-6).

Tests whether larger updates help once the tighter reply limit keeps replies from running away, as they did at
3x without it (qwen3_8_27b_harbor_mix_v2_h200x2_lr3x). Own names, so it runs next to the 1x run.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import (
    qwen3_8_27b_harbor_mix_v2_h200x2_reply12k as reply12k,
)
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2_h200x2_reply12k import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v2-reply12k-lr2x"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v2-reply12k-lr2x"
TRAINER_EXTRA_VOLUMES = {
    **reply12k.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v2-reply12k-lr2x-trials", create_if_missing=True, version=2
    ),
}


class _MixV2Reply12kLr2x(reply12k._MixV2Reply12k):
    lr = 2e-6
    wandb_group = "qwen3-8-27b-harbor-mix-v2-reply12k-lr2x"
    prometheus_run_name = wandb_group
    environment = reply12k._environment("qwen3-8-27b-harbor-mix-v2-reply12k-lr2x-sandboxes")


miles = _MixV2Reply12kLr2x(**harbor_config.arguments(reply12k.TASKS_ROOT), **harbor_config.SCORE_CENTERING)
