"""Smoke run: Qwen3.8-27B on a few prepared MiMo tasks through Miles' Harbor path.

Tests the loop, not learning: the harbor-envs Volume, task setup, the non-root agent
(terminal), the separate grader (code), rewards into the trainer, and two training steps.
Same model, checkpoints and serving as qwen3_8_27b_swebench (#267), with score centering
in place of IcePop and without DFlash. The agent is
terminus-2: it calls the model from the rollout worker, so sandboxes need no route to the
session servers.
"""

import dataclasses

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_swebench as base

APP_NAME = "stitch-qwen3-8-27b-harbor-smoke"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-smoke"
SOURCE_MODEL = base.SOURCE_MODEL
SOURCE_REVISION = base.SOURCE_REVISION
BF16_CHECKPOINT_PATH = base.BF16_CHECKPOINT_PATH
ROLLOUT_CHECKPOINT_PATH = base.ROLLOUT_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = base.TORCH_DIST_CHECKPOINT_PATH
SERVED_CHECKPOINT_FORMAT = base.SERVED_CHECKPOINT_FORMAT
CHECKPOINT_PREP_REQUIRES_GPU = base.CHECKPOINT_PREP_REQUIRES_GPU
LOCAL_CHECKPOINT_PATH = None
PREP_ENV = base.PREP_ENV
# Trace each model request through the sidecar by its rid.
SGLANG_SERVER_ENV = {**base.SGLANG_SERVER_ENV, "STITCH_TRACE_REQUESTS": "1"}
SGLANG_SERVER_ARGS = harbor_config.without_speculative_decoding(base.SGLANG_SERVER_ARGS)
MAX_SEQ_LEN = base.MAX_SEQ_LEN
SIDECAR_COMMIT_MODE = base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_DELTA_UPDATE_MODE = base.SGLANG_DELTA_UPDATE_MODE

MILES_IMAGE_TAG = harbor_config.MILES_IMAGE_TAG
MILES_REPO_REF = harbor_config.MILES_REPO_REF
SGLANG_RUNTIME = harbor_config.SGLANG_RUNTIME
SIDECAR_RESPONSE_KEEPALIVE_AFTER = harbor_config.SIDECAR_RESPONSE_KEEPALIVE_AFTER
TRAINER_EXTRA_PIP_PACKAGES = harbor_config.TRAINER_EXTRA_PIP_PACKAGES
TRAINER_IMAGE_RUN_COMMANDS = harbor_config.TRAINER_IMAGE_RUN_COMMANDS
TASKS_ROOT = "/harbor-envs/smoke-20261012"
# Trial logs go to their own Volume: Stitch reloads the run Volume (/stitch) when it publishes
# weights, and that reload fails while Harbor holds a trial file open there.
TRAINER_EXTRA_VOLUMES = {
    "/harbor-envs": modal_sdk.Volume.from_name("harbor-envs"),
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-smoke-trials", create_if_missing=True, version=2
    ),
}

AGENT_CONCURRENCY = 8

modal = dataclasses.replace(
    base.modal,
    rollout_min_containers=1,
    rollout_max_containers=2,
    rollout_target_inputs=4,
)


class _Smoke(base._Miles):
    num_rollout = 2
    rollout_batch_size = 2
    n_samples_per_prompt = 4
    global_batch_size = 8
    save_interval = None
    sglang_server_concurrency = AGENT_CONCURRENCY
    async_max_concurrent_samples = AGENT_CONCURRENCY
    session_server_workers = 2
    wandb_project = "agent-rl-smoke"  # not the team's fully-async-rl-modal project
    wandb_group = "qwen3-8-27b-harbor-smoke"
    prometheus_run_name = wandb_group

    environment = harbor_config.environment(
        tasks_root=TASKS_ROOT,
        trials_dir="/harbor-trials",
        sandbox_app="qwen3-8-27b-harbor-smoke-sandboxes",
        max_seq_len=MAX_SEQ_LEN,
        max_output_tokens=8192,
        agent_timeout=900,
        trial_timeout=3600,
        max_turns=30,
    )

    def prepare_data(self) -> None:
        pass  # the tasks are prepared outside Stitch, on the harbor-envs Volume


miles = _Smoke(**harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING)
