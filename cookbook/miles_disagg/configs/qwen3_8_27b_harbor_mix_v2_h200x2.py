"""Mixture v2 with the trainer on two 8 x H200 machines (16 GPUs): TP 4 x CP 2 x DP 2.

The same trainer as qwen3_8_27b_harbor_mix_v2b_h200x2, so v2 and v2b runs on these recipes differ
only in the task mix and names. Same names and Volume as the B300 v2 recipe. Megatron's default
optimizer checkpoints cannot change the model split, so a B300 v2 run (TP 2) cannot resume here.
"""

import dataclasses

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2 as v2
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2 import *  # noqa: F403

modal = dataclasses.replace(v2.modal, gpu="H200")


class _MixV2H200x2(v2._MixV2):
    tensor_model_parallel_size = 4
    context_parallel_size = 2
    actor_num_nodes = 2


miles = _MixV2H200x2(**harbor_config.arguments(v2.TASKS_ROOT), **harbor_config.SCORE_CENTERING)
