"""A1: plain GRPO on the heterogeneous fleet, with no off-policy correction."""

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_hetero_study as study,
)

APP_NAME = f"{study.APP_NAME}-a1-grpo"
EXPERIMENT_VOLUME_NAME = f"{study.EXPERIMENT_VOLUME_NAME}-a1-grpo"

modal = study.modal


def __getattr__(name: str):
    return getattr(study, name)


class _Miles(study.GrpoMiles):
    wandb_group = "qwen3-6-35b-mimo-code-hetero-a1-grpo"
    prometheus_run_name = wandb_group


miles = _Miles(**study.arguments())
