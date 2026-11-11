"""qwen3_8_27b_harbor_ptb_practice_e with the trainer on two 8 x H200 machines (TP 4 x CP 2 x DP 2), as
qwen3_8_27b_harbor_mix_v2_h200x2, for when no 8 x B300 machine is free. The run starts from weights only (a fresh
optimizer), so the first run's TP 2 checkpoint loads into the new split.
"""

import dataclasses

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_ptb_practice_e as e
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_ptb_practice_e import *  # noqa: F403

modal = dataclasses.replace(e.modal, gpu="H200")


class _PtbPracticeEH200x2(e._PtbPracticeE):
    tensor_model_parallel_size = 4
    context_parallel_size = 2
    actor_num_nodes = 2
    environment = {**e._PtbPracticeE.environment, **harbor_config.MULTI_MACHINE_ENVIRONMENT}


miles = _PtbPracticeEH200x2(**harbor_config.arguments(e.TASKS_ROOT), **harbor_config.SCORE_CENTERING)
