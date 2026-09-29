"""TIS and routing-replay control for heterogeneous low-precision rollout."""

from dataclasses import replace

from cookbook.miles_disagg import swebench_config
from cookbook.miles_disagg.configs import (
    qwen3_6_35b_a3b_mimo_code_heterogeneous as base,
)

APP_NAME = f"{base.APP_NAME}-tis"
EXPERIMENT_VOLUME_NAME = f"{base.EXPERIMENT_VOLUME_NAME}-tis"

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

modal = replace(
    base.modal,
    rollout_pools=tuple(
        replace(
            pool,
            sglang_args={
                **pool.sglang_args,
                "--enable-return-routed-experts": "",
                "--sampling-mask-max-tokens": "8192",
            },
        )
        for pool in base.modal.rollout_pools
    ),
)


class _Miles(base._Miles):
    num_rollout = 3
    rollout_top_p = 0.95
    rollout_top_k = 4096

    use_tis = True
    tis_clip_low = 0.5
    tis_clip = 2.0
    use_rollout_routing_replay = True

    wandb_group = "qwen3-6-35b-mimo-code-heterogeneous-tis"
    prometheus_run_name = wandb_group


arguments = swebench_config.arguments()
arguments["prompt_data"] = f"{base.DATASET_PATH}/code.jsonl"
miles = _Miles(**arguments)
