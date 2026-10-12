"""Smoke run: Qwen3.8-27B on a few prepared MiMo tasks through Miles' Harbor path.

Tests the loop, not learning: the harbor-envs Volume, task setup, the non-root agent
(terminal), the separate grader (code), rewards into the trainer, and two training steps.
Same model, checkpoints, serving and loss as qwen3_8_27b_swebench (#267). The agent is
terminus-2: it calls the model from the rollout worker, so sandboxes need no route to the
session servers.
"""

import dataclasses
import json

import modal as modal_sdk

from cookbook.miles_disagg import swebench_config
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
DRAFT_MODEL = base.DRAFT_MODEL
DRAFT_REVISION = base.DRAFT_REVISION
PREP_ENV = base.PREP_ENV
SGLANG_SERVER_ENV = base.SGLANG_SERVER_ENV
SGLANG_SERVER_ARGS = base.SGLANG_SERVER_ARGS
MAX_SEQ_LEN = base.MAX_SEQ_LEN
SIDECAR_COMMIT_MODE = base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_DELTA_UPDATE_MODE = base.SGLANG_DELTA_UPDATE_MODE

# terminus-2 needs Harbor's Miles branch (truncation policy for TITO); pip cannot resolve it.
# Install into the interpreter that runs the trainer and Ray (the image's /opt/sglang venv).
HARBOR_REF = "917b3c1be7cf6553a7a9fe25de8adb6f8e639287"  # harbor-miles-v0.20.0
TRAINER_EXTRA_PIP_PACKAGES = ("uv",)
TRAINER_IMAGE_RUN_COMMANDS = (
    "uv pip install --python /opt/sglang/bin/python "
    f'"harbor[modal] @ git+https://github.com/harbor-framework/harbor@{HARBOR_REF}" "modal==1.5.3"',
)
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


def _arguments() -> dict:
    args = swebench_config.arguments()
    for key in (
        "custom_rollout_log_function_path",
        "session_sample_picker_path",
        "session_sample_postprocessor_path",
    ):
        args.pop(key)
    return args | {
        "prompt_data": f"{TASKS_ROOT}/train.jsonl",
        "custom_agent_function_path": "harbor_agent_function.run",
        "custom_rm_path": "generate.reward_func",
    }


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

    environment = {
        "PYTHONPATH": (
            "/root/Megatron-LM:/root/miles:/root/miles/examples/experimental/harbor"
            ":/root/miles/examples/swe-agent-harbor-docker"
        ),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "RAY_health_check_timeout_ms": "60000",
        "RAY_health_check_failure_threshold": "30",
        "LITELLM_LOG": "ERROR",
        "HARBOR_ENV_TYPE": "modal",
        "HARBOR_TASKS_DIR": f"{TASKS_ROOT}/tasks",
        "HARBOR_TRIALS_DIR": "/harbor-trials",
        "HARBOR_ENV_KWARGS": json.dumps(
            {"app_name": "qwen3-8-27b-harbor-smoke-sandboxes"}
        ),
        "AGENT_MODEL_NAME": "model",
        "AGENT_TIMEOUT": "900",
        "AGENT_TRIAL_TIMEOUT": "3600",
        "AGENT_MAX_INPUT_TOKENS": str(MAX_SEQ_LEN - 8192),
        "AGENT_MAX_OUTPUT_TOKENS": "8192",
        "HARBOR_MAX_SEQ_LEN": str(MAX_SEQ_LEN),
        "HARBOR_AGENT_MAX_ITERATIONS": "30",
    }

    def prepare_data(self) -> None:
        pass  # the tasks are prepared outside Stitch, on the harbor-envs Volume


miles = _Smoke(**_arguments())
