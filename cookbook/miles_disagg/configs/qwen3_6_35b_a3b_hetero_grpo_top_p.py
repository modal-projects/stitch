"""GRPO with top-p sampling and sampling-support replay on the heterogeneous fleet.

Vanilla GRPO with one change: samplers draw each token from its top-p 0.97 nucleus,
bounded by the 64 most likely tokens, and the trainer renormalizes its probabilities
over the same support. There is no off-policy correction, so this arm isolates what
truncated sampling does on its own.
"""

from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_hetero_grpo as arm
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_grpo import *  # noqa: F403

APP_NAME = "stitch-qwen36-hetero-grpo-top-p"
EXPERIMENT_VOLUME_NAME = APP_NAME


class _Miles(arm._Miles):
    rollout_top_p = 0.97
    rollout_top_k = 64

    wandb_group = "qwen36-hetero-grpo-top-p"
    prometheus_run_name = wandb_group


miles = _Miles(**arm.arguments())
