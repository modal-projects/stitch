"""Score centering on the heterogeneous fleet.

Each token's score has its expected value under the sampler subtracted, computed from
the sampler's top-128 candidates, which cancels the drift toward the sampler that a
mismatched policy gradient carries (arXiv:2609.20807).
"""

from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import HeteroMiles, arguments

APP_NAME = "stitch-qwen36-hetero-score-centering"
EXPERIMENT_VOLUME_NAME = APP_NAME


class ScoreCenteringMiles(HeteroMiles):
    # Top-p sampling bounded by top-k; the recorded candidates must cover the whole
    # realized support, so they exceed top-k to leave room for ties at the cutoff.
    rollout_top_p = 0.97
    rollout_top_k = 64
    # REINFORCE with group-centered rewards and a token-mean loss, as in the paper.
    disable_grpo_std_normalization = True
    calculate_per_token_loss = True
    loss_type = "score_centering"
    score_centering_top_k = 128
    score_centering_is = "none"
    # The loss scores the trainer against the sampler's recorded log-probs.
    skip_actor_forward_only = False
    use_rollout_logprobs = True

    wandb_group = "qwen36-hetero-score-centering"
    prometheus_run_name = wandb_group


miles = ScoreCenteringMiles(**arguments())
