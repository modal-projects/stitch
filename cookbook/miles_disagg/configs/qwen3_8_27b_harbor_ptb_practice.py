"""RL on the post-training practice tasks, started from the plain-token SFT run (qwen3_8_27b_sft_plain_tokens).

Tasks: `ptb-practice-v3-train` on the harbor-envs Volume (eval repo envs/posttrain_practice.py: repair, deadline and
posttrain, 300 tasks, oracle-validated). They run on 4-CPU sandboxes and grade themselves, including each task's own
20- or 30-minute limit, so the agent limit here is 15 minutes x Miles' timeout multiplier 2.0. Small and short:
16 kept groups of 8 attempts per step (128 agents on 16 rollout containers), 20 steps, a checkpoint and HF export at
steps 10 and 20. Otherwise qwen3_8_27b_harbor_mix_v3b_b300_s2 (B300 trainer, score centering, 16K replies, 128K
context). Placeholder: set SFT_RUN to the SFT run's id.
"""

import dataclasses

import modal as modal_sdk

from cookbook.common.constants import START_PATH
from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v3b_b300_s2 as s2
from cookbook.miles_disagg.configs import qwen3_8_27b_sft_plain_tokens as sft_run
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v3b_b300_s2 import *  # noqa: F403

SFT_RUN = "a6d58819"  # the plain-token SFT run
START_VOLUME = modal_sdk.Volume.from_name(
    sft_run.EXPERIMENT_VOLUME_NAME, environment_name=sft_run.EXPERIMENT_VOLUME_ENV
)
BF16_CHECKPOINT_PATH = START_PATH / SFT_RUN / sft_run.miles.save_hf.format(rollout_id=sft_run.STEPS - 1)
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = START_PATH / SFT_RUN / "checkpoints"

APP_NAME = "stitch-qwen3-8-27b-ptb-practice"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-ptb-practice"
TASKS_ROOT = "/harbor-envs/ptb-practice-v3-train"
AGENT_CONCURRENCY = 128
modal = dataclasses.replace(s2.modal, rollout_min_containers=16, rollout_max_containers=16)
TRAINER_EXTRA_VOLUMES = {
    **s2.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-ptb-practice-trials",
        environment_name=s2.EXPERIMENT_VOLUME_ENV,
        create_if_missing=True,
        version=2,
    ),
}


class _PtbPractice(s2._MixV3bS2):
    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    rollout_batch_size = 16
    global_batch_size = 16 * 8  # rollout_batch_size x n_samples_per_prompt
    num_rollout = 20
    save_interval = 10
    save_retain_interval = 10
    sglang_server_concurrency = AGENT_CONCURRENCY
    async_max_concurrent_samples = AGENT_CONCURRENCY
    wandb_group = "qwen3-8-27b-ptb-practice"
    prometheus_run_name = wandb_group
    environment = {
        **s2._MixV3bS2.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-ptb-practice-sandboxes",
            max_seq_len=s2.b300._MixV2.max_seq_len,
            max_output_tokens=s2.b300._MixV2.rollout_max_response_len,
            agent_timeout=900,
            trial_timeout=3600,
            max_turns=200,
        ),
    }


miles = _PtbPractice(
    **harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING
)
