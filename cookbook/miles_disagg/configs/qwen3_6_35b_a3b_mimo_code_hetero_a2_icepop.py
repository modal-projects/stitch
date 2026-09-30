"""A2: the strongest prior recipe, after MiMo-V2.6's large-scale RL run, without R3.

- IcePop masked importance sampling against the sampler's support-replayed
  probabilities: each token's policy-gradient term is weighted by its detached
  trainer/sampler ratio, and tokens whose ratio leaves [0.2, 5] are dropped. The bounds
  are MiMo-V2.6's, wider below than IcePop's [0.5, 5] default.
- Prompt-mean loss aggregation: a token mean within each prompt's rollouts, then an
  equal-weight mean over prompts.
- A frozen MoE router, which keeps expert load from drifting during RL.
"""

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_hetero_study as study,
)

APP_NAME = f"{study.APP_NAME}-a2-icepop"
EXPERIMENT_VOLUME_NAME = f"{study.EXPERIMENT_VOLUME_NAME}-a2-icepop"

modal = study.modal


def __getattr__(name: str):
    return getattr(study, name)


class _Miles(study.GrpoMiles):
    use_tis = True
    custom_tis_function_path = (
        "miles.backends.training_utils.loss_hub.corrections.icepop_function"
    )
    tis_clip_low = 0.2
    tis_clip = 5.0
    calculate_per_token_loss = False
    prompt_mean_loss = True
    freeze_moe_router = True
    wandb_group = "qwen3-6-35b-mimo-code-hetero-a2-icepop"
    prometheus_run_name = wandb_group


miles = _Miles(**study.arguments())
