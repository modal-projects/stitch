"""IcePop on the heterogeneous fleet, after MiMo-V2.6's large-scale RL recipe without R3.

- Top-p sampling whose support the trainer replays, and group-mean advantages
  without std normalization.
- Masked importance sampling against the sampler's support-replayed probabilities:
  each token's policy-gradient term is weighted by its detached trainer/sampler
  ratio, and tokens whose ratio leaves [0.2, 5] are dropped. The bounds are
  MiMo-V2.6's, wider below than IcePop's [0.5, 5] default.
- Prompt-mean loss aggregation: a token mean within each prompt's rollouts, then an
  equal-weight mean over prompts.
- A frozen MoE router, which keeps expert load from drifting during RL.
"""

from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import HeteroMiles, arguments

APP_NAME = "stitch-qwen36-hetero-icepop"
EXPERIMENT_VOLUME_NAME = APP_NAME


class _Miles(HeteroMiles):
    rollout_top_p = 0.97
    # Only bounds the returned support; top-p sets it in practice.
    rollout_top_k = 4096
    disable_grpo_std_normalization = True
    use_tis = True
    custom_tis_function_path = (
        "miles.backends.training_utils.loss_hub.corrections.icepop_function"
    )
    tis_clip_low = 0.2
    tis_clip = 5.0
    calculate_per_token_loss = False
    prompt_mean_loss = True
    freeze_moe_router = True

    wandb_group = "qwen36-hetero-icepop"
    prometheus_run_name = wandb_group


miles = _Miles(**arguments())
