"""Qwen3.5-9B at lr 2e-6 past step 40, arm B: arm A plus a length control on the reward
(harbor_agent.run_length_controlled)."""

from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import run

globals().update(
    run(
        "lr2x-from40-b",
        2e-6,
        44,
        start="qwen3-5-9b-lr2x-s42-step40",
        agent="cookbook.miles_disagg.harbor_agent.run_length_controlled",
    )
)
