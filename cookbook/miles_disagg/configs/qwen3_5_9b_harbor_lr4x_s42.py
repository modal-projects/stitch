"""Qwen3.5-9B learning-rate test: lr 4e-6, rollout seed 42 (qwen3_5_9b_harbor_lr)."""

from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import run

globals().update(run("lr4x-s42", 4e-6, 42))
