"""Mixture v1b (qwen3_8_27b_harbor_mix_v1b) with an 8 x H200 trainer, as qwen3_8_27b_harbor_mix_v1_h200.

Context parallel 4, so each GPU holds at most 32K tokens of a 128K sample. Same app, Volume and W&B
group as the B300 recipe; runs differ by run id.
"""

import dataclasses

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v1b as mix
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v1b import *  # noqa: F403

modal = dataclasses.replace(mix.modal, gpu="H200")


class _MixV1bH200(mix._MixV1b):
    context_parallel_size = 4


miles = _MixV1bH200(
    **harbor_config.arguments(mix.TASKS_ROOT), **harbor_config.SCORE_CENTERING
)
