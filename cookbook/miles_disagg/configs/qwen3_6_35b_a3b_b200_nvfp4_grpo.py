"""Vanilla GRPO on the B200 NVFP4 fleet: the hetero arm with only the fleet changed."""

from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_hetero_grpo as arm
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_b200_nvfp4 import *  # noqa: F403

APP_NAME = "stitch-qwen36-b200-nvfp4-grpo"
EXPERIMENT_VOLUME_NAME = APP_NAME


class _Miles(arm._Miles):
    wandb_group = "qwen36-b200-nvfp4-grpo"
    prometheus_run_name = wandb_group


miles = _Miles(**arm.arguments())
