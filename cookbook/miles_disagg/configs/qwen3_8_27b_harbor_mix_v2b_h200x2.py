"""Mixture v2b with the trainer on two 8 x H200 machines (16 GPUs): TP 4 x CP 2 x DP 2.

The one-machine H200 recipe (qwen3_8_27b_harbor_mix_v2b_h200) fits in GPU memory, but its first step
took 90 min to train (B300: 8-14 min) and its host RAM reached 1,017 of 1,024 GiB at the save. A
second machine holds a second model copy: each machine trains half the batch and keeps half the
CPU-offloaded optimizer state. Same run as v2b otherwise (names, Volume, tasks), so it can take over
that run with --resume-from.
"""

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2b_h200 as h200
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2b_h200 import *  # noqa: F403


class _MixV2bH200x2(h200._MixV2bH200):
    actor_num_nodes = 2
    environment = {**h200._MixV2bH200.environment, **harbor_config.MULTI_MACHINE_ENVIRONMENT}


miles = _MixV2bH200x2(**harbor_config.arguments(h200.v2b.TASKS_ROOT), **harbor_config.SCORE_CENTERING)
