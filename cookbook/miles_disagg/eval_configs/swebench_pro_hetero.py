"""SWE-bench Pro for the heterogeneous Qwen3.6 arms.

BF16 is the learning policy: the trainer's precision on the trainer's GPU, with a BF16
KV cache. FP8 and NVFP4 are deployment settings, each served exactly as its training
pool serves it.

The benchmark is SWE-bench Pro V2 as Scale released it, graded by its own protocol: the
agent works offline in the training harness, and its patch is graded in a fresh Sandbox
of the task image whose network is open. The eval set is V2 less the tasks that do not
run reliably there: ``swebench_pro_scale_v2_excluded.json`` lists each with its reason.
"""

import json
from pathlib import Path

import cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero as hetero
from cookbook.common.constants import DATA_PATH
from cookbook.miles_disagg import swebench_pro

NAME = "swebench-pro"
# Base points are shared by every arm; any arm's recipe supplies the same harness.
BASE_EXPERIMENT = "qwen3_6_35b_a3b_hetero_grpo"
# Written by swebench_pro.prepare_swebench_pro_v2.
DATASET_PATH = DATA_PATH / "swebench-pro-scale-v2"
_EXCLUDED = json.loads(
    (Path(__file__).parent / "swebench_pro_scale_v2_excluded.json").read_text()
)
if (_EXCLUDED["task_set"], _EXCLUDED["benchmark_revision"]) != (
    DATASET_PATH.name,
    swebench_pro.V2_REPOSITORY_REVISION,
):
    raise ValueError("the excluded-task list was checked against another task set")
EXCLUDED_TASKS: dict[str, str] = _EXCLUDED["tasks"]
TASKS = swebench_pro.V2_TASKS - len(EXCLUDED_TASKS)
TASKS_DIR = DATASET_PATH / "tasks"
# V2's protocol: each patch is graded in a fresh Sandbox of the task image.
GRADE_IN_FRESH_SANDBOX = True
# Four samples a task give pass@1, pass@2 and pass@4. Points scored before 2026-10-04
# 17:00 have eight samples; their pass@1/2/4 are the same unbiased estimates.
N_SAMPLES = 4
# Miles eval-dataset fields; the rest (max_response_len, keys) follow the recipe.
DATASET = {
    "name": "swebench_pro",
    "path": f"{DATASET_PATH}/test.jsonl",
    "n_samples_per_eval_prompt": N_SAMPLES,
    # Full-vocabulary sampling, the same as the training samplers.
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
}
# Sandbox or API failures never reached the policy, so they rerun instead of scoring.
INFRA_RETRIES = 3
# A model request that the serving gateway loses never returns. The session server gives
# up on a request after REQUEST_DEADLINE_SECONDS, and the agent resends the turn, up to
# REQUEST_ATTEMPTS times, before the episode aborts and reruns. The deadline must not cut
# off a real turn: a full 32K-token turn at ~45 tok/s per request (8 B300 engines under
# eval load) takes ~12 min, and degenerating checkpoints write such turns, so 600 s
# aborted them as if lost. 1800 s fits a full turn down to ~18 tok/s.
REQUEST_DEADLINE_SECONDS = 1800
REQUEST_ATTEMPTS = 3
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
