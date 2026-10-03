"""Qwen3.6-35B-A3B RL on MiMo code tasks, rolled out in BF16 on B200 only.

The first rung of the mismatch ladder: the rollout fleet runs the trainer's GPU and
precision, BF16 weights and a BF16 KV cache on B200, so the sampler differs from the
B200 BF16 trainer only in its kernels. Everything else is the heterogeneous base: the
trainer, the data, the concurrent sessions and the run shape. Each algorithm recipe
takes its hetero arm unchanged and names its own app, volume and W&B group:

- ``qwen3_6_35b_a3b_b200_bf16_grpo``: vanilla GRPO.
- ``qwen3_6_35b_a3b_b200_bf16_icepop``: GRPO with IcePop's masked importance weights.
- ``qwen3_6_35b_a3b_b200_bf16_score_centering``: score centering.
- ``qwen3_6_35b_a3b_b200_bf16_score_centering_mis``: score centering with IcePop's
  weights.

The next rung, ``qwen3_6_35b_a3b_b200_nvfp4``, keeps the GPU and changes the precision;
the heterogeneous fleet changes both.
"""

from dataclasses import replace

from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_hetero as hetero
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import *  # noqa: F403

ROLLOUT_WEIGHT_VIEWS = {"bf16": hetero.BF16_CHECKPOINT_PATH}
# Every save still exports all three precisions, as the heterogeneous runs do, so each
# rung's policy can be evaluated in BF16, FP8 and NVFP4.
EXPORT_WEIGHT_VIEWS = dict(hetero.ROLLOUT_WEIGHT_VIEWS)
# B200's sessions per engine hold for BF16 too: at 32 sessions of MiMo-calibrated
# traffic a BF16 engine decodes 94 tok/s per request, H200 FP8's 90 at its 32.
# A fixed fleet that holds every concurrent session at that load.
ROLLOUT_REPLICAS = (
    hetero.ROLLOUT_CONCURRENT_SAMPLES // hetero.ROLLOUT_TARGET_INPUTS["B200"]
)

modal = replace(
    hetero.modal,
    rollout_pools=(
        replace(
            hetero._pool(
                name="ServerB200BF16",
                gpu="B200",
                weight_view="bf16",
                attention_backend="trtllm_mha",
                kv_cache_dtype="bf16",
                # Triton is the MoE runner validated for BF16 staged updates.
                moe_runner_backend="triton",
            ),
            min_containers=ROLLOUT_REPLICAS,
            max_containers=ROLLOUT_REPLICAS,
        ),
    ),
)
