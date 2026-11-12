"""Qwen3.5-9B at lr 2e-6 past step 80, arm D (new data): arm C (qwen3_5_9b_harbor_from80_c) on mixture-9b-v3-cyber,
the mix re-filtered with the step-40 model (well-specified MiMo code only, 12% cyber)."""

from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import run

globals().update(
    run(
        "lr2x-from80-d",
        2e-6,
        45,
        start="qwen3-5-9b-lr2x-from40-a-step80",
        tasks="mixture-9b-v3-cyber",
    )
)
