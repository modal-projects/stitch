"""Shared SWE-bench Pro agent and dataset settings; recipes own model and capacity."""

from cookbook.common.constants import DATA_PATH

DATASET_PATH = DATA_PATH / "swebench-pro"
TRAINER_PACKAGES = (
    "harbor[modal,huggingface]==0.20.0",
    "mini-swe-agent==2.4.5",
    "swebench==4.1.0",
    "modal==1.5.3",
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
        "custom_rollout_log_function_path": (
            "cookbook.miles_disagg.modal_swe.metrics.log_rollout_data"
        ),
        "custom_generate_function_path": (
            "miles.rollout.generate_hub.agentic_tool_call.generate"
        ),
        "custom_agent_function_path": "cookbook.miles_disagg.modal_swe.agent.run",
        "custom_rm_path": "cookbook.miles_disagg.modal_swe.agent.reward_func",
        "use_session_server": "v2",
        "session_sample_picker_path": (
            "cookbook.miles_disagg.modal_swe.agent.pick_latest_leaf"
        ),
        "session_sample_postprocessor_path": (
            "cookbook.miles_disagg.modal_swe.agent.postprocess_samples"
        ),
    }


def environment(
    *,
    sandbox_app: str,
    processes: int,
    threads_per_process: int = 16,
    boot_concurrency_per_process: int = 4,
) -> dict[str, str]:
    return {
        "PYTHONPATH": "/root:/root/Megatron-LM:/root/miles",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "RAY_health_check_timeout_ms": "60000",
        "RAY_health_check_failure_threshold": "30",
        "AGENT_MODEL_NAME": "model",
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
        "LITELLM_LOG": "ERROR",
        "MODAL_SWE_TASKS_DIR": f"{DATASET_PATH}/tasks",
        "MODAL_SWE_SANDBOX_APP": sandbox_app,
        "MODAL_SWE_MAX_STEPS": "256",
        "MODAL_SWE_EPISODE_TIMEOUT": "7200",
        "MODAL_SWE_MODEL_REQUEST_TIMEOUT": "1800",
        "MODAL_SWE_EXEC_TIMEOUT": "120",
        "MODAL_SWE_OUTPUT_HARD_LIMIT_BYTES": str(16 * 1024 * 1024),
        "MODAL_SWE_SETUP_TIMEOUT": "240",
        "MODAL_SWE_VERIFY_TIMEOUT": "3600",
        "MODAL_SWE_INJECT_PYTEST_REPORTER": "0",
        "MODAL_SWE_MEMORY_MIB": "2048",
        "MODAL_SWE_AGENT_PROCESSES": str(processes),
        "MODAL_SWE_AGENT_THREADS_PER_PROCESS": str(threads_per_process),
        "MODAL_SWE_SANDBOX_BOOT_CONCURRENCY_PER_PROCESS": str(
            boot_concurrency_per_process
        ),
    }
