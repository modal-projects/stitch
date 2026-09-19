"""Qwen3-4B BF16 GRPO on GSM8K: a small, synchronous math training starter.

Two H200 trainer GPUs shard Adam state; the rollout pool starts with one TP1
replica serving exact-version requests. Checkpoints and held-out evaluation run
every 20 rollouts.
"""

from __future__ import annotations

from typing import Any

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH, DATA_PATH
from cookbook.miles_disagg.config import MilesConfig

APP_NAME = "stitch-qwen3-4b-math"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-4b-math"
SOURCE_MODEL = "Qwen/Qwen3-4B"
SOURCE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-4b-1cfa9a72-bf16"
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-4b-1cfa9a72-torch-dist-tp1"
SERVED_CHECKPOINT_FORMAT = "bf16"
CHECKPOINT_PREP_REQUIRES_GPU = False
LOCAL_CHECKPOINT_PATH = None

_DATASET_REVISION = "0cbd9f31d91ac21a7613dcbc7fef992adac459ae"
_DATASET_PATH = DATA_PATH / "gsm8k-0cbd9f31"

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
SGLANG_DELTA_UPDATE_MODE = "cpu"
SGLANG_SERVER_ARGS = {
    "--served-model-name": SOURCE_MODEL,
    "--tp": "1",
    "--dtype": "bfloat16",
    "--load-format": "safetensors",
    "--enable-cpu-weight-cache": "",
    "--weight-loader-drop-cache-after-load": "",
    "--reasoning-parser": "qwen3",
    "--context-length": "8192",
    "--mem-fraction-static": "0.8",
    "--chunked-prefill-size": "4096",
    "--max-running-requests": "16",
    "--cuda-graph-max-bs-decode": "16",
}

modal = ModalConfig(
    gpu="H200",
    trainer_memory_mib=(64 * 1024, 256 * 1024),
    rollout_memory_mib=(64 * 1024, 256 * 1024),
    rollout_min_containers=1,
    rollout_target_inputs=8,
    torch_dist_prep_nodes=1,
    torch_dist_prep_gpus_per_node=1,
    torch_dist_convert_extra_args="--tensor-model-parallel-size 1",
)


async def generate_math_answer(
    *,
    base_url: str,
    prompt: list[dict[str, Any]],
    request_kwargs: dict[str, Any],
    **_kwargs: Any,
) -> None:
    """Request one answer; Miles' session collects its tokens and log probabilities."""
    import httpx

    async with httpx.AsyncClient(timeout=900.0, trust_env=False) as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={"model": SOURCE_MODEL, "messages": prompt, **request_kwargs},
        )
        response.raise_for_status()


class _Miles(MilesConfig):
    megatron_model_type = "qwen3-4B"
    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    megatron_to_hf_mode = "raw"

    actor_num_nodes = 1
    actor_num_gpus_per_node = 2
    num_gpus_per_node = 2
    rollout_num_gpus = 0
    rollout_num_gpus_per_engine = 1
    sglang_server_concurrency = 16
    pause_generation_mode = "in_place"
    use_session_server = True
    tito_model = "qwen3"
    custom_generate_function_path = (
        "miles.rollout.generate_hub.agentic_tool_call.generate"
    )
    custom_agent_function_path = (
        "cookbook.miles_disagg.configs.qwen3_4b_math.generate_math_answer"
    )
    max_seq_len = 8192
    custom_rollout_request_hook_path = (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    rollout_request_timeout_secs = 600
    custom_config_path = {
        "rollout_request_weight_version_mode": "exact",
        "rollout_request_weight_version_lag": 0,
        "rollout_request_retry_attempts": 240,
        "rollout_request_retry_sleep": 1.0,
    }

    update_weights_interval = 1
    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"

    prompt_data = str(_DATASET_PATH / "train.parquet")
    eval_prompt_data = ["gsm8k", str(_DATASET_PATH / "test.parquet")]
    input_key = "messages"
    label_key = "label"
    # The session server renders these messages with Qwen3's registered template.
    apply_chat_template = False
    rollout_shuffle = True
    balance_data = True
    rm_type = "math"

    num_rollout = 120
    rollout_batch_size = 16
    n_samples_per_prompt = 8
    global_batch_size = 128
    rollout_max_prompt_len = 4096
    rollout_max_response_len = 4096
    rollout_max_context_len = 8192
    rollout_temperature = 1.0
    rollout_top_p = 1.0
    rollout_top_k = -1

    save_interval = 20
    save_hf = "hf_checkpoints/weight_v{rollout_id:06d}"
    eval_interval = 20
    n_samples_per_eval_prompt = 1
    eval_max_prompt_len = 4096
    eval_max_response_len = 4096
    eval_temperature = 0.0
    eval_top_p = 1.0
    eval_top_k = -1
    log_passrate = True

    tensor_model_parallel_size = 1
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    seq_length = 8192
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 8192
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    attention_backend = "flash"
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True

    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    advantage_estimator = "grpo"
    eps_clip = 0.2
    eps_clip_high = 0.28
    kl_loss_coef = 0.0
    entropy_coef = 0.0

    def prepare_data(self) -> None:
        from datasets import load_dataset

        dataset = load_dataset("zhuzilin/gsm8k", revision=_DATASET_REVISION)
        _DATASET_PATH.mkdir(parents=True, exist_ok=True)
        dataset["train"].to_parquet(str(_DATASET_PATH / "train.parquet"))
        dataset["test"].to_parquet(str(_DATASET_PATH / "test.parquet"))


miles = _Miles()
