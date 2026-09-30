"""Shared recipe for the heterogeneous-RL stability study.

Every arm trains the same model on the same heterogeneous fleet and data, samples with
top-p 0.95 / top-k 64 and replays that support on the trainer, and uses group-centered
advantages without std normalization and a per-token loss. Arms then differ only in the
off-policy estimator; the two controls change only the fleet (B1) or the staleness (B2).

Arm modules take everything they don't override from here, and this module takes the
rest from the heterogeneous base recipe.
"""

from cookbook.miles_disagg import swebench_config
from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_heterogeneous as base,
)

APP_NAME = base.APP_NAME
# Short enough for every arm's suffix to fit Modal's 64-character Volume names.
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen36-hetero-study"

modal = base.modal


def __getattr__(name: str):
    # Checkpoints, sidecar and prep settings are the base recipe's.
    return getattr(base, name)


class GrpoMiles(base._Miles):
    """Plain GRPO: one update per rollout batch with no off-policy correction."""

    rollout_top_p = 0.95
    rollout_top_k = 64
    disable_grpo_std_normalization = True
    calculate_per_token_loss = True
    # Detached mismatch diagnostics: ratio tails, sequence log-ratio, advantages.
    log_rollout_mismatch_diagnostics = True


class ScoreCenteringMiles(GrpoMiles):
    loss_type = "score_centering"
    score_centering_top_k = 128
    score_centering_is = "none"
    skip_actor_forward_only = False
    use_rollout_logprobs = True


def arguments() -> dict:
    values = swebench_config.arguments()
    values["prompt_data"] = f"{base.DATASET_PATH}/code.jsonl"
    return values
