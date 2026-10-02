"""SWE-bench Pro for the heterogeneous Qwen3.6 arms.

BF16 is the learning policy: the trainer's precision on the trainer's GPU, with a BF16
KV cache. FP8 and NVFP4 are deployment settings, each served exactly as its training
pool serves it.
"""

import cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero as hetero
from cookbook.miles_disagg import swebench_config

NAME = "swebench-pro"
# Base points are shared by every arm; any arm's recipe supplies the same harness.
BASE_EXPERIMENT = "qwen3_6_35b_a3b_hetero_grpo"
TASKS = 731
TASKS_DIR = swebench_config.DATASET_PATH / "tasks"
N_SAMPLES = 8
# Miles eval-dataset fields; the rest (max_response_len, keys) follow the recipe.
DATASET = {
    "name": "swebench_pro",
    "path": f"{swebench_config.DATASET_PATH}/test.jsonl",
    "n_samples_per_eval_prompt": N_SAMPLES,
    "temperature": 1.0,
    "top_p": 0.97,
    "top_k": -1,
}
# Sandbox or API failures never reached the policy, so they rerun instead of scoring.
INFRA_RETRIES = 3
SANDBOX_APP = "stitch-swebench-pro-eval-sandbox"


def _training_pool(name: str):
    return next(pool for pool in hetero.modal.rollout_pools if pool.name == name)


POOLS = {
    "bf16": hetero._pool(
        name="EvalB300BF16",
        gpu="B300",
        weight_view="bf16",
        attention_backend="trtllm_mha",
        kv_cache_dtype="bfloat16",
        moe_runner_backend="triton",
    ),
    "fp8": _training_pool("ServerH200FP8"),
    "nvfp4": _training_pool("ServerB300NVFP4W4A16"),
}
