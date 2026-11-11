"""Qwen3.5-9B at lr 2e-6 past step 40, arm A: the 2e-6 recipe started from lr2x_s42's step-40 export (weights only,
fresh optimizer), rollout seed 44, 40 more steps (qwen3_5_9b_harbor_lr)."""

from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import run

globals().update(run("lr2x-from40-a", 2e-6, 44, start="qwen3-5-9b-lr2x-s42-step40"))
