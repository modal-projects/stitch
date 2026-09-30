"""Naive GRPO on the heterogeneous fleet: no off-policy correction."""

from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import HeteroMiles, arguments

APP_NAME = "stitch-qwen36-hetero-grpo"
EXPERIMENT_VOLUME_NAME = APP_NAME


class _Miles(HeteroMiles):
    wandb_group = "qwen36-hetero-grpo"
    prometheus_run_name = wandb_group


miles = _Miles(**arguments())
