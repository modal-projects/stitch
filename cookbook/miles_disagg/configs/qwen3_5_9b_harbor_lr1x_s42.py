"""Qwen3.5-9B learning-rate test: lr 1e-6 (the control), rollout seed 42 (qwen3_5_9b_harbor_lr)."""

from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import run

globals().update(run("lr1x-s42", 1e-6, 42))
