"""Temporary Phase-1 validation for FP8 Hopper and NVFP4 W4A16 Blackwell views."""

from dataclasses import replace

from cookbook.common.config import RolloutPoolConfig
from cookbook.common.constants import CHECKPOINTS_PATH
from cookbook.miles_disagg import swebench_config
from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_swebench_pro as base

APP_NAME = "stitch-qwen3-6-35b-heterogeneous-phase1"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-6-35b-heterogeneous-phase1"

SOURCE_MODEL = base.SOURCE_MODEL
SOURCE_REVISION = base.SOURCE_REVISION
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-heterogeneous-bf16"
FP8_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-heterogeneous-fp8"
NVFP4_CHECKPOINT_PATH = (
    CHECKPOINTS_PATH / "qwen3-6-35b-a3b-995ad96e-heterogeneous-nvfp4-w4a16"
)
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
ROLLOUT_WEIGHT_VIEWS = {
    "fp8": FP8_CHECKPOINT_PATH,
    "nvfp4": NVFP4_CHECKPOINT_PATH,
}
TORCH_DIST_CHECKPOINT_PATH = base.TORCH_DIST_CHECKPOINT_PATH
SERVED_CHECKPOINT_FORMAT = "bf16"
CHECKPOINT_PREP_REQUIRES_GPU = True
UNPACK_FUSED_EXPERTS = True
LOCAL_CHECKPOINT_PATH = base.LOCAL_CHECKPOINT_PATH
TRAINER_EXTRA_PIP_PACKAGES = base.TRAINER_EXTRA_PIP_PACKAGES

NVFP4_ENCODING_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "none",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
}
PREP_ENV = {
    **base.PREP_ENV,
    **NVFP4_ENCODING_ENV,
    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
}
SGLANG_SERVER_ENV = {}

SIDECAR_COMMIT_MODE = base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_DELTA_UPDATE_MODE = base.SGLANG_DELTA_UPDATE_MODE


def _server_args(*, attention_backend: str, moe_runner_backend: str) -> dict[str, str]:
    return {
        **base.ROLLOUT_SERVER_ARGS,
        "--tp": "1",
        "--attention-backend": attention_backend,
        "--mem-fraction-static": "0.7",
        "--moe-runner-backend": moe_runner_backend,
        "--max-running-requests": "4",
        "--cuda-graph-max-bs-decode": "4",
        "--sleep-on-idle": "",
    }


modal = replace(
    base.modal,
    rollout_cpu=64.0,
    rollout_pools=(
        RolloutPoolConfig(
            name="ServerH200FP8",
            gpu="H200",
            gpus_per_engine=1,
            target_inputs=1,
            sglang_args=_server_args(
                attention_backend="fa3", moe_runner_backend="triton"
            ),
            weight_view="fp8",
            min_containers=1,
            max_containers=2,
        ),
        RolloutPoolConfig(
            name="ServerB300NVFP4W4A16",
            gpu="B300",
            gpus_per_engine=1,
            target_inputs=1,
            sglang_args=_server_args(
                attention_backend="trtllm_mha",
                moe_runner_backend="flashinfer_cutedsl",
            ),
            weight_view="nvfp4",
            environment={
                "SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1",
                "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE": "0",
            },
            min_containers=1,
            max_containers=2,
        ),
    ),
)


class _Miles(base._Miles):
    hf_checkpoint = str(BF16_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    num_rollout = 2
    save_interval = 1
    rollout_batch_size = 2
    n_samples_per_prompt = 2
    global_batch_size = rollout_batch_size * n_samples_per_prompt
    sglang_server_concurrency = 2
    async_max_concurrent_samples = global_batch_size
    use_tis = False
    get_mismatch_metrics = False
    custom_tis_function_path = None
    tis_clip_low = None
    tis_clip = None
    use_rollout_routing_replay = False
    environment = {
        **base._Miles.environment,
        **NVFP4_ENCODING_ENV,
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
    }


miles = _Miles(**swebench_config.arguments())
