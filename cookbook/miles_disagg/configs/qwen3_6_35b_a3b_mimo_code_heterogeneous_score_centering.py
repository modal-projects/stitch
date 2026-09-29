"""Score-centered GRPO with heterogeneous low-precision rollout."""

from cookbook.miles_disagg import swebench_config
from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_heterogeneous as base,
)

APP_NAME = f"{base.APP_NAME}-score-centering"
EXPERIMENT_VOLUME_NAME = f"{base.EXPERIMENT_VOLUME_NAME}-score-centering"

SOURCE_MODEL = base.SOURCE_MODEL
SOURCE_REVISION = base.SOURCE_REVISION
BF16_CHECKPOINT_PATH = base.BF16_CHECKPOINT_PATH
FP8_CHECKPOINT_PATH = base.FP8_CHECKPOINT_PATH
NVFP4_CHECKPOINT_PATH = base.NVFP4_CHECKPOINT_PATH
ROLLOUT_CHECKPOINT_PATH = base.ROLLOUT_CHECKPOINT_PATH
ROLLOUT_WEIGHT_VIEWS = base.ROLLOUT_WEIGHT_VIEWS
TORCH_DIST_CHECKPOINT_PATH = base.TORCH_DIST_CHECKPOINT_PATH
SERVED_CHECKPOINT_FORMAT = base.SERVED_CHECKPOINT_FORMAT
CHECKPOINT_PREP_REQUIRES_GPU = base.CHECKPOINT_PREP_REQUIRES_GPU
UNPACK_FUSED_EXPERTS = base.UNPACK_FUSED_EXPERTS
LOCAL_CHECKPOINT_PATH = base.LOCAL_CHECKPOINT_PATH
TRAINER_EXTRA_PIP_PACKAGES = base.TRAINER_EXTRA_PIP_PACKAGES
PREP_ENV = base.PREP_ENV
SGLANG_SERVER_ENV = base.SGLANG_SERVER_ENV

SIDECAR_COMMIT_MODE = base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_DELTA_UPDATE_MODE = base.SGLANG_DELTA_UPDATE_MODE

modal = base.modal


class _Miles(base._Miles):
    num_rollout = 3

    skip_actor_forward_only = False
    loss_type = "score_centering"
    score_centering_top_k = 128
    score_centering_is = "none"
    use_rollout_logprobs = True
    disable_grpo_std_normalization = True
    calculate_per_token_loss = True

    rollout_top_p = 0.95
    rollout_top_k = 64

    wandb_group = "qwen3-6-35b-mimo-code-heterogeneous-score-centering"
    prometheus_run_name = wandb_group


arguments = swebench_config.arguments()
arguments["prompt_data"] = f"{base.DATASET_PATH}/code.jsonl"
miles = _Miles(**arguments)
