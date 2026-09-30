"""A3: score centering, which cancels the drift from sampling off the trainer's policy."""

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_hetero_study as study,
)

APP_NAME = f"{study.APP_NAME}-a3-score-centering"
EXPERIMENT_VOLUME_NAME = f"{study.EXPERIMENT_VOLUME_NAME}-a3-score-centering"

modal = study.modal


def __getattr__(name: str):
    return getattr(study, name)


class _Miles(study.ScoreCenteringMiles):
    wandb_group = "qwen3-6-35b-mimo-code-hetero-a3-score-centering"
    prometheus_run_name = wandb_group


miles = _Miles(**study.arguments())
