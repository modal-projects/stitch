"""B2: A1's plain GRPO on the heterogeneous fleet with low staleness.

The control for staleness: A1 keeps 168 groups in flight and a two-batch buffer, so
trained samples are ~5 versions old. One batch in flight and a half-batch buffer
bound the work in progress to 1.5 batches, so samples are ~2 versions old.
"""

from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_hetero_study as study,
)

APP_NAME = f"{study.APP_NAME}-b2-grpo-low-staleness"
EXPERIMENT_VOLUME_NAME = f"{study.EXPERIMENT_VOLUME_NAME}-b2-grpo-low-staleness"

AGENT_THREADS_PER_PROCESS = 16
ROLLOUT_CONCURRENT_SAMPLES = study.AGENT_PROCESSES * AGENT_THREADS_PER_PROCESS

modal = study.modal


def __getattr__(name: str):
    return getattr(study, name)


class _Miles(study.GrpoMiles):
    async_max_concurrent_samples = ROLLOUT_CONCURRENT_SAMPLES
    sglang_server_concurrency = ROLLOUT_CONCURRENT_SAMPLES
    async_data_buffer_capacity_factor = 0.5
    environment = {
        **study.GrpoMiles.environment,
        "MODAL_SWE_AGENT_THREADS_PER_PROCESS": str(AGENT_THREADS_PER_PROCESS),
    }
    wandb_group = "qwen3-6-35b-mimo-code-hetero-b2-grpo-low-staleness"
    prometheus_run_name = wandb_group


miles = _Miles(**study.arguments())
