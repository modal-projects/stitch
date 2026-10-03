"""Qwen3.6-35B-A3B RL on MiMo code tasks, rolled out in NVFP4 on B200 only.

The second rung of the mismatch ladder: the rollout fleet keeps the B200 BF16
trainer's GPU but samples from NVFP4 (W4A16) weights with an FP8 KV cache, every engine
the heterogeneous fleet's B200 engine. Everything else is the heterogeneous base: the
trainer, the data, the concurrent sessions and the run shape. Each algorithm recipe
takes its hetero arm unchanged and names its own app, volume and W&B group:

- ``qwen3_6_35b_a3b_b200_nvfp4_grpo``: vanilla GRPO.
- ``qwen3_6_35b_a3b_b200_nvfp4_icepop``: GRPO with IcePop's masked importance weights.
- ``qwen3_6_35b_a3b_b200_nvfp4_score_centering``: score centering.
- ``qwen3_6_35b_a3b_b200_nvfp4_score_centering_mis``: score centering with IcePop's
  weights.
"""

from dataclasses import replace

from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_hetero as hetero
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import *  # noqa: F403

ROLLOUT_WEIGHT_VIEWS = {"nvfp4": hetero.NVFP4_CHECKPOINT_PATH}
# Every save still exports all three precisions, as the heterogeneous runs do, so each
# rung's policy can be evaluated in BF16, FP8 and NVFP4.
EXPORT_WEIGHT_VIEWS = dict(hetero.ROLLOUT_WEIGHT_VIEWS)
# A fixed fleet of the heterogeneous fleet's B200 engine that holds every concurrent
# session at that engine's load.
ROLLOUT_REPLICAS = (
    hetero.ROLLOUT_CONCURRENT_SAMPLES // hetero.ROLLOUT_TARGET_INPUTS["B200"]
)

modal = replace(
    hetero.modal,
    rollout_pools=tuple(
        replace(pool, min_containers=ROLLOUT_REPLICAS, max_containers=ROLLOUT_REPLICAS)
        for pool in hetero.modal.rollout_pools
        if pool.name == "ServerB200NVFP4W4A16"
    ),
)
