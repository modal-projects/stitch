"""Mixture v2b with the trainer on one 8 x H200 machine instead of 8 x B300.

Same run as qwen3_8_27b_harbor_mix_v2b (names, Volume, tasks, steps); only the trainer GPU and
its parallelism change. An H200 has 141 GB, a B300 288 GB. The B300 split (tensor parallel 2 x
context parallel 2 x data parallel 2) does not fit, and tensor parallel 2 x context parallel 4
ran out of GPU memory in score centering on 128K samples, so the model is split 4 ways:
tensor parallel 4 x context parallel 2 x data parallel 1. 4 is the largest split the model allows
(4 key-value heads).
"""

import dataclasses

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2b as v2b
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2b import *  # noqa: F403

modal = dataclasses.replace(v2b.modal, gpu="H200")


class _MixV2bH200(v2b._MixV2b):
    tensor_model_parallel_size = 4
    context_parallel_size = 2


miles = _MixV2bH200(**harbor_config.arguments(v2b.TASKS_ROOT), **harbor_config.SCORE_CENTERING)
