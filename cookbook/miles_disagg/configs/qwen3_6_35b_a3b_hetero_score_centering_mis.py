"""Score centering composed with IcePop's masked importance weights.

Each token's score is weighted by its trainer/sampler ratio inside [0.5, 5] and by
zero outside, and score centering subtracts the expected weighted score under the
sampler.
"""

from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_score_centering import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_score_centering import (
    ScoreCenteringMiles,
    arguments,
)

APP_NAME = "stitch-qwen36-hetero-score-centering-mis"
EXPERIMENT_VOLUME_NAME = APP_NAME


class _Miles(ScoreCenteringMiles):
    score_centering_is = "mis"
    score_centering_mis_low = 0.5
    score_centering_mis_high = 5.0

    wandb_group = "qwen36-hetero-score-centering-mis"
    prometheus_run_name = wandb_group


miles = _Miles(**arguments())
