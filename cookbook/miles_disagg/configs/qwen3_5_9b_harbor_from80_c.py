"""Qwen3.5-9B at lr 2e-6 past step 80, arm C (control): the 2e-6 recipe started from lr2x-from40-a's step-80 export
(weights only, fresh optimizer) on the same mix, mixture-9b-v2, rollout seed 45, 40 more steps (qwen3_5_9b_harbor_lr)."""

from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import run

globals().update(run("lr2x-from80-c", 2e-6, 45, start="qwen3-5-9b-lr2x-from40-a-step80"))
