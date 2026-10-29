"""Mixture v2 on two 8 x H200 machines, 155 steps instead of 55 (about three epochs of mixture v2).

The same recipe and names as qwen3_8_27b_harbor_mix_v2_h200x2, so a run of that recipe resumes here (same
RUN_ID) from its last save and trains 100 more steps. The learning rate is constant, so it does not change.
"""

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2_h200x2 as h200
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2_h200x2 import *  # noqa: F403


class _MixV2H200x2Long(h200._MixV2H200x2):
    num_rollout = 155


miles = _MixV2H200x2Long(**harbor_config.arguments(h200.TASKS_ROOT), **harbor_config.SCORE_CENTERING)
