"""Mixture v0, phase 1: Qwen3.8-27B on MiMo terminal, general and cyber through Miles' Harbor path.

Tasks: the mix `mixture-v0-phase1` on the harbor-envs Volume (terminal twice per epoch, general,
cyber), 128K tokens per rollout. The zero-variance filter drops groups whose attempts all score the
same, so each step trains on 32 groups that have a signal. Same model, checkpoints and serving
as qwen3_8_27b_swebench (#267), but with score centering (harbor_config.SCORE_CENTERING) in
place of IcePop, and without DFlash. The agent is
terminus-2: it calls the model from the rollout worker, so sandboxes need no route to the session
servers. Not here yet: a validation set (eval_interval).

Watch the first steps: the kept share per set, step time, grader errors and judge load.
"""

import dataclasses

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_swebench as base

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v0"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v0"
SOURCE_MODEL = base.SOURCE_MODEL
SOURCE_REVISION = base.SOURCE_REVISION
BF16_CHECKPOINT_PATH = base.BF16_CHECKPOINT_PATH
ROLLOUT_CHECKPOINT_PATH = base.ROLLOUT_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = base.TORCH_DIST_CHECKPOINT_PATH
SERVED_CHECKPOINT_FORMAT = base.SERVED_CHECKPOINT_FORMAT
CHECKPOINT_PREP_REQUIRES_GPU = base.CHECKPOINT_PREP_REQUIRES_GPU
LOCAL_CHECKPOINT_PATH = None
PREP_ENV = base.PREP_ENV
SGLANG_SERVER_ENV = base.SGLANG_SERVER_ENV
# 128K context: at 64K, 64 of 128 attempts of the first Tinker step ran out of context; at 128K, 16.
MAX_SEQ_LEN = 131_072
SGLANG_SERVER_ARGS = {
    **harbor_config.without_speculative_decoding(base.SGLANG_SERVER_ARGS),
    "--context-length": str(MAX_SEQ_LEN + 8),
    # #267's queue of 4 rejects overflow with 503 "The request queue is full", and terminus ends the
    # trial, which Miles scores 0. With fewer servers up than planned, half the trials of a pilot
    # failed so. A deeper queue makes them wait instead.
    "--max-queued-requests": "64",
}
SIDECAR_COMMIT_MODE = base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_DELTA_UPDATE_MODE = base.SGLANG_DELTA_UPDATE_MODE

MILES_IMAGE_TAG = harbor_config.MILES_IMAGE_TAG
MILES_REPO_REF = harbor_config.MILES_REPO_REF
SGLANG_RUNTIME = harbor_config.SGLANG_RUNTIME
SIDECAR_RESPONSE_KEEPALIVE_AFTER = harbor_config.SIDECAR_RESPONSE_KEEPALIVE_AFTER
TRAINER_EXTRA_PIP_PACKAGES = harbor_config.TRAINER_EXTRA_PIP_PACKAGES
TRAINER_IMAGE_RUN_COMMANDS = harbor_config.TRAINER_IMAGE_RUN_COMMANDS
TASKS_ROOT = "/harbor-envs/mixture-v0-phase1"
# Trial logs go to their own Volume: Stitch reloads the run Volume (/stitch) when it publishes
# weights, and that reload fails while Harbor holds a trial file open there.
TRAINER_EXTRA_VOLUMES = {
    "/harbor-envs": modal_sdk.Volume.from_name(
        "harbor-envs", environment_name=harbor_config.DATA_ENV
    ),
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v0-trials", create_if_missing=True, version=2
    ),
}
# General's grader calls an LLM judge. Its task.toml reads MIMO_JUDGE_URL, MIMO_TEXT_JUDGE and
# HF_TOKEN (the judge token) from the rollout worker's environment. This Secret holds all three.
TRAINER_EXTRA_SECRETS = [
    modal_sdk.Secret.from_name("mimo-judge", environment_name=harbor_config.DATA_ENV)
]

# 16 rollout containers x 16 requests each, as #267.
AGENT_CONCURRENCY = 256

modal = dataclasses.replace(
    base.modal, rollout_min_containers=16, rollout_max_containers=16
)


# Qwen3.8 thinks at length: at #267's 8K cap, 16 of 16 TermGrade attempts on Spindle and 2 of 2 in
# a baseline ended on an over-long reply. Our MiMo eval found 16K covers the 90th percentile reply.
REPLY_TOKENS = 16_384


class _MixV0(base._Miles):
    rollout_max_response_len = REPLY_TOKENS
    # About one epoch: ~1,900 sampled prompts, about half kept, 32 kept groups per step.
    num_rollout = 30
    # A checkpoint and HF export every step, so a crash or a preemption loses at most one step (Stitch resumes
    # only from a saved and published pair). Megatron keeps every 5th torch_dist checkpoint and the last one
    # (one is ~413 GiB); validation evals score every 5th HF export.
    save_interval = 1
    save_retain_interval = 5
    # The session server's proxy deadline. #267's 300 s cut long replies (16K tokens at 128K context under
    # load): 42-74 deadline errors per 10-15 min, about 11% of trials ended as AgentError (reward 0).
    miles_router_timeout = 3600
    max_seq_len = MAX_SEQ_LEN
    # Two GPUs share each 128K sequence, so each GPU holds at most 64K tokens of it, as in #267.
    context_parallel_size = 2
    dynamic_sampling_filter_path = (
        "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    )
    sglang_server_concurrency = AGENT_CONCURRENCY
    async_max_concurrent_samples = AGENT_CONCURRENCY
    wandb_project = "agent-rl"  # not the team's fully-async-rl-modal project
    wandb_group = "qwen3-8-27b-harbor-mix-v0-phase1"
    prometheus_run_name = wandb_group

    environment = harbor_config.environment(
        tasks_root=TASKS_ROOT,
        trials_dir="/harbor-trials",
        sandbox_app="qwen3-8-27b-harbor-mix-v0-sandboxes",
        max_seq_len=MAX_SEQ_LEN,
        max_output_tokens=REPLY_TOKENS,
        # One hour for every set, as in our eval: 4x terminal's 900 s, 1x general's and cyber's 3,600 s.
        agent_timeout=3600,
        trial_timeout=7200,
        # The context or the agent timeout ends most episodes first; this cap only stops loops.
        max_turns=200,
    )

    def prepare_data(self) -> None:
        pass  # the mix is prepared outside Stitch, on the harbor-envs Volume


miles = _MixV0(**harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING)
