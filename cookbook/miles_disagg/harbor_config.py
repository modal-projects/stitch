"""Shared settings for Miles' Harbor path; recipes own model, tasks and capacity.

Tasks are prepared outside Stitch as Harbor task folders plus one train.jsonl on a Volume.
"""

import json

from cookbook.miles_disagg import swebench_config

# terminus-2 needs Harbor's Miles branch (truncation policy for TITO); pip cannot resolve it.
# Install into the interpreter that runs the trainer and Ray (the image's /opt/sglang venv).
HARBOR_REF = "917b3c1be7cf6553a7a9fe25de8adb6f8e639287"  # harbor-miles-v0.20.0
TRAINER_EXTRA_PIP_PACKAGES = ("uv",)
TRAINER_IMAGE_RUN_COMMANDS = (
    "uv pip install --python /opt/sglang/bin/python "
    f'"harbor[modal] @ git+https://github.com/harbor-framework/harbor@{HARBOR_REF}" "modal==1.5.3"',
)


def arguments(tasks_root: str) -> dict:
    """Miles workload settings: the Harbor agent function and its reward."""
    args = swebench_config.arguments()
    for key in (
        "custom_rollout_log_function_path",
        "session_sample_picker_path",
        "session_sample_postprocessor_path",
    ):
        args.pop(key)
    return args | {
        "prompt_data": f"{tasks_root}/train.jsonl",
        "custom_agent_function_path": "harbor_agent_function.run",
        "custom_rm_path": "generate.reward_func",
    }


def environment(
    *,
    tasks_root: str,
    trials_dir: str,
    sandbox_app: str,
    max_seq_len: int,
    max_output_tokens: int,
    agent_timeout: int,
    trial_timeout: int,
    max_turns: int,
) -> dict[str, str]:
    """Rollout-worker environment for harbor_agent_function."""
    return {
        "PYTHONPATH": (
            "/root/Megatron-LM:/root/miles:/root/miles/examples/experimental/harbor"
            ":/root/miles/examples/swe-agent-harbor-docker"
        ),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "RAY_health_check_timeout_ms": "60000",
        "RAY_health_check_failure_threshold": "30",
        "LITELLM_LOG": "ERROR",
        "HARBOR_ENV_TYPE": "modal",
        "HARBOR_TASKS_DIR": f"{tasks_root}/tasks",
        "HARBOR_TRIALS_DIR": trials_dir,
        # Harbor's default sandbox lifetime is 24 h, so a sandbox of a stopped run keeps running.
        # Let each sandbox live only a little longer than its trial.
        "HARBOR_ENV_KWARGS": json.dumps(
            {"app_name": sandbox_app, "sandbox_timeout_secs": trial_timeout + 600}
        ),
        "AGENT_MODEL_NAME": "model",
        "AGENT_TIMEOUT": str(agent_timeout),
        "AGENT_TRIAL_TIMEOUT": str(trial_timeout),
        "AGENT_MAX_INPUT_TOKENS": str(max_seq_len - max_output_tokens),
        "AGENT_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "HARBOR_MAX_SEQ_LEN": str(max_seq_len),
        "HARBOR_AGENT_MAX_ITERATIONS": str(max_turns),
    }
