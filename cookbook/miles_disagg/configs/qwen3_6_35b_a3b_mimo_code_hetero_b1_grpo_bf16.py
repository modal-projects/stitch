"""B1: A1's plain GRPO on a homogeneous BF16 B300 fleet.

The control for heterogeneity: same trainer, data, sampler and session count as A1,
so staleness matches, but every rollout runs the BF16 checkpoint on one GPU type.
"""

from dataclasses import replace

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_hetero_study as study,
)

APP_NAME = f"{study.APP_NAME}-b1-grpo-bf16"
EXPERIMENT_VOLUME_NAME = f"{study.EXPERIMENT_VOLUME_NAME}-b1-grpo-bf16"

# Enough engines at the pool's parity target to route every A1 session.
B300_BF16_ENGINES = 44

ROLLOUT_WEIGHT_VIEWS = {"bf16": study.BF16_CHECKPOINT_PATH}

(_B300_BF16,) = (
    pool for pool in study.BLACKWELL_BF16_POOLS if pool.name == "ServerB300BF16"
)
modal = replace(
    study.modal,
    rollout_pools=(
        replace(
            _B300_BF16,
            min_containers=B300_BF16_ENGINES,
            max_containers=B300_BF16_ENGINES,
            # No quantization anywhere: the heterogeneous fleet's FP8 KV cache too.
            sglang_args={**_B300_BF16.sglang_args, "--kv-cache-dtype": "auto"},
        ),
    ),
)


def __getattr__(name: str):
    return getattr(study, name)


class _Miles(study.GrpoMiles):
    wandb_group = "qwen3-6-35b-mimo-code-hetero-b1-grpo-bf16"
    prometheus_run_name = wandb_group


miles = _Miles(**study.arguments())
