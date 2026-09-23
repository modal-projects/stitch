"""Shared SWE-bench Pro Harbor settings; recipes own model and capacity."""

import json

from cookbook.common.constants import DATA_PATH

DATASET_PATH = DATA_PATH / "swebench-pro"
TRAINER_PACKAGES = ("uv==0.8.15", "modal==1.5.3")
HARBOR_REVISION = "581c7975f5d038cf6f553edc2e1eacfe6696790b"
TRAINER_IMAGE_RUN_COMMANDS = (
    "install -d /opt/harbor && git -C /opt/harbor init && "
    "git -C /opt/harbor remote add origin "
    "https://github.com/harbor-framework/harbor.git && "
    f"git -C /opt/harbor fetch --depth=1 origin {HARBOR_REVISION} && "
    "git -C /opt/harbor checkout --detach FETCH_HEAD && "
    'uv pip install --python /opt/sglang/bin/python3 '
    '"/opt/harbor[modal,huggingface]" "mini-swe-agent==2.4.5"',
)


def arguments() -> dict:
    """Return fresh workload settings for a Miles config."""
    return {
        "prompt_data": f"{DATASET_PATH}/test.jsonl",
        "input_key": "prompt",
        "metadata_key": "metadata",
        "apply_chat_template": False,
        "rollout_shuffle": True,
        "balance_data": True,
        "fully_async": True,
        "pause_generation_mode": "in_place",
        "rollout_submission_granularity": "sample",
        "custom_generate_function_path": (
            "miles.rollout.generate_hub.agentic_tool_call.generate"
        ),
        "custom_agent_function_path": "harbor_agent_function.run",
        "custom_rm_path": "generate.reward_func",
        "use_session_server": True,
    }


def environment(
    *,
    sandbox_app: str,
) -> dict[str, str]:
    return {
        "PYTHONPATH": (
            "/root/Megatron-LM:/root/miles:"
            "/root/miles/examples/experimental/harbor:"
            "/root/miles/examples/swe-agent-harbor-docker"
        ),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "RAY_health_check_timeout_ms": "60000",
        "RAY_health_check_failure_threshold": "30",
        "HARBOR_ENV_TYPE": "modal",
        "HARBOR_ENV_KWARGS": json.dumps(
            {
                "app_name": sandbox_app,
                "modal_sandbox_v2": True,
                "sandbox_timeout_secs": 7_800,
            },
            separators=(",", ":"),
        ),
        "HARBOR_TASKS_DIR": f"{DATASET_PATH}/tasks",
        "HARBOR_TRIALS_DIR": "/tmp/harbor-trials",
        "AGENT_MODEL_NAME": "model",
        "AGENT_MAX_INPUT_TOKENS": "57344",
        "AGENT_MAX_OUTPUT_TOKENS": "8192",
        "AGENT_TIMEOUT": "6600",
        "AGENT_TRIAL_TIMEOUT": "7200",
        "HARBOR_MAX_SEQ_LEN": "65536",
        "HARBOR_AGENT_MAX_ITERATIONS": "256",
        "HARBOR_OVERRIDE_MEMORY_MB": "2048",
        "HARBOR_VERIFIER_TIMEOUT_SEC": "3600",
    }
