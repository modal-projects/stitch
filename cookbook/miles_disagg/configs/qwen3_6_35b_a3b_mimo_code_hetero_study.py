"""Shared recipe for the heterogeneous-RL stability study.

Each GPU type serves the precision and kernels that suit it (NVFP4 on Blackwell, FP8
and BF16 on Hopper, BF16 on A100 and RTX PRO 6000), so every pool samples from its own
approximation of the trainer's policy. The study accepts that off-policyness and
leaves staleness unbounded: every request still asks its replica for the latest
published weights of its precision, but no sample is dropped for its age.

Every arm trains the same model on the same fleet and data, samples with top-p 0.95 /
top-k 64 and replays that support on the trainer, and uses group-centered advantages
without std normalization and a per-token loss, so arms differ only in the off-policy
estimator. B1 is an optional homogeneous BF16 reference.

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
    # Unbounded staleness: no sample is dropped for its age.
    max_weight_staleness = None


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
