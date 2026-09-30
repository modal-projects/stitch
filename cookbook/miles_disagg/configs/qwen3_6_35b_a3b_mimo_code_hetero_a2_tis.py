"""A2: GRPO with truncated importance sampling against the sampler's support-replayed probabilities."""

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_hetero_study as study,
)

APP_NAME = f"{study.APP_NAME}-a2-tis"
EXPERIMENT_VOLUME_NAME = f"{study.EXPERIMENT_VOLUME_NAME}-a2-tis"

modal = study.modal


def __getattr__(name: str):
    return getattr(study, name)


class _Miles(study.GrpoMiles):
    use_tis = True
    tis_clip_low = 0.5
    tis_clip = 2.0
    wandb_group = "qwen3-6-35b-mimo-code-hetero-a2-tis"
    prometheus_run_name = wandb_group


miles = _Miles(**study.arguments())
