"""Score centering with masked importance weights, top-p sampling and support replay
on the B200 BF16 fleet.

The hetero arm with only the fleet changed.
"""

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_hetero_score_centering_mis_top_p as arm,
)
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_b200_bf16 import *  # noqa: F403

APP_NAME = "stitch-qwen36-b200-bf16-score-centering-mis-top-p"
EXPERIMENT_VOLUME_NAME = APP_NAME


class _Miles(arm._Miles):
    wandb_group = "qwen36-b200-bf16-score-centering-mis-top-p"
    prometheus_run_name = wandb_group


miles = _Miles(**arm.arguments())
