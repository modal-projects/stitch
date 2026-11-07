"""Smoke run of qwen3_8_27b_sft_warmstart: the same trainer, data and batch for 2 steps, with a checkpoint and an HF
export at the last step. It checks that Miles' SFT path trains Qwen3.8-27B in sft_app and that the export serves.
"""

import copy

from cookbook.miles_disagg.configs import qwen3_8_27b_sft_warmstart as full
from cookbook.miles_disagg.configs.qwen3_8_27b_sft_warmstart import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-sft-warmstart-smoke"
STEPS = 2
miles = copy.copy(full.miles)
miles.num_rollout = miles.save_interval = STEPS
miles.wandb_group = "qwen3-8-27b-sft-warmstart-smoke"
