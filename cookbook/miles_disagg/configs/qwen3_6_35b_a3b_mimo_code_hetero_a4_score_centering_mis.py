"""A4: score centering composed with truncated importance weights."""

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_hetero_study as study,
)

APP_NAME = f"{study.APP_NAME}-a4-score-centering-tis"
EXPERIMENT_VOLUME_NAME = f"{study.EXPERIMENT_VOLUME_NAME}-a4-score-centering-tis"

modal = study.modal


def __getattr__(name: str):
    return getattr(study, name)


class _Miles(study.ScoreCenteringMiles):
    score_centering_is = "tis"
    score_centering_tis_clip = 2.0
    wandb_group = "qwen3-6-35b-mimo-code-hetero-a4-score-centering-tis"
    prometheus_run_name = wandb_group


miles = _Miles(**study.arguments())
