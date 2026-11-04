"""qwen3_8_27b_harbor_mix_v3a_h200x2 computing in the Modal environment capacity-validation: the same run and settings,
for resuming it there when training-team has no 8 x H200 machines. Its run Volume, base checkpoints and trials Volume
stay in training-team.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v3a_h200x2 as v3a
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v3a_h200x2 import *  # noqa: F403

EXPERIMENT_VOLUME_ENV = "training-team"
TRAINER_EXTRA_VOLUMES = {
    **v3a.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v3a-trials", environment_name=EXPERIMENT_VOLUME_ENV, version=2
    ),
}


class _MixV3aCv(v3a._MixV3a):
    # capacity-validation uses a newer image builder than this Modal client; the rollout workers build the task images.
    environment = {**v3a._MixV3a.environment, "MODAL_IMAGE_BUILDER_VERSION": "2025.06"}


miles = _MixV3aCv(
    **harbor_config.arguments(v3a.TASKS_ROOT), **harbor_config.SCORE_CENTERING
)
