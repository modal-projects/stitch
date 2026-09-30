"""A4: score centering composed with masked importance weights (MIS).

Each token's score is weighted by its trainer/sampler ratio inside [0.2, 5] and by zero
outside, the same correction as A2, and score centering subtracts the expected weighted
score under the sampler.
"""

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_hetero_study as study,
)

APP_NAME = f"{study.APP_NAME}-a4-score-centering-mis"
EXPERIMENT_VOLUME_NAME = f"{study.EXPERIMENT_VOLUME_NAME}-a4-score-centering-mis"

modal = study.modal


def __getattr__(name: str):
    return getattr(study, name)


class _Miles(study.ScoreCenteringMiles):
    score_centering_is = "mis"
    score_centering_mis_low = 0.2
    score_centering_mis_high = 5.0
    wandb_group = "qwen3-6-35b-mimo-code-hetero-a4-score-centering-mis"
    prometheus_run_name = wandb_group


miles = _Miles(**study.arguments())
