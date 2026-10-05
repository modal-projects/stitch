"""Score centering with masked importance weights, top-p sampling and support replay.

The score-centering-with-MIS arm with one change: samplers draw each token from its
top-p 0.97 nucleus, bounded by the 64 most likely tokens, and the trainer renormalizes
its probabilities over the same support. The sampler logs that whole support among
its 128 recorded candidates, twice the cap so ties at the cutoff still fit, so the
centering expectation is exact, and the masked ratio compares the two policies on
the same support.
"""

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_hetero_score_centering_mis as arm,
)
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_score_centering_mis import *  # noqa: F403

APP_NAME = "stitch-qwen36-hetero-score-centering-mis-top-p"
EXPERIMENT_VOLUME_NAME = APP_NAME


class _Miles(arm._Miles):
    rollout_top_p = 0.97
    rollout_top_k = 64

    wandb_group = "qwen36-hetero-score-centering-mis-top-p"
    prometheus_run_name = wandb_group


miles = _Miles(**arm.arguments())
