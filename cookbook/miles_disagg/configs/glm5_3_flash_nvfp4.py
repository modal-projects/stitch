"""GLM-5.3-Flash NVFP4 GRPO on SWE-bench Pro with hybrid KDA/DSA attention."""

from copy import deepcopy
from pathlib import Path

from cookbook.common.constants import CHECKPOINTS_PATH
from cookbook.common.serving_image import SGLangRuntime
from cookbook.miles_disagg.checkpoint_hooks import COMPLETION_HOOK
from cookbook.miles_disagg.configs import glm5_2_nvfp4 as base

APP_NAME = "stitch-glm5-3-flash-nvfp4"
EXPERIMENT_VOLUME_NAME = "stitch-miles-glm5-3-flash-nvfp4"
SOURCE_MODEL = "zai-org/GLM-5.3-Flash"
SOURCE_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
MILES_REPO_REF = "90dca0889422099efaa0e62c046e29f26fa36a30"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-flash-bf16"
ROLLOUT_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-flash-nvfp4"
TORCH_DIST_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-flash-torch-dist"
MILES_IMAGE_TAG = (
    "radixark/miles@sha256:"
    "66725f740a6013b00d27e21fdfd480a24b0d3b5c61840342e9405bf1e09e5162"
)
# PR #2786 supplies the model implementation over Stitch's async Miles fork.
_MODEL_PR_BASE = "071fd2a6fcd450055be8a7e0bad9db58f6ef095f"
_MODEL_PR_HEAD = "dbbd610e7f4dad06008a85ff57da798c38337979"
MILES_IMAGE_PATCHES = (
    Path(__file__).resolve().parents[1] / "patches/miles-glm5-next-runtime.patch",
)
TRAINER_PACKAGE_PATCHES = (
    Path(__file__).resolve().parents[1] / "patches/fla-kda-host-block-size.patch",
)
TRAINER_IMAGE_RUN_COMMANDS = (
    "git -C /root/miles fetch https://github.com/radixark/miles.git "
    f"{_MODEL_PR_BASE} {_MODEL_PR_HEAD}"
    f" && git -C /root/miles diff {_MODEL_PR_BASE} {_MODEL_PR_HEAD}"
    " -- miles miles_plugins scripts/models scripts/run_glm5_3_flash.py"
    " > /tmp/glm5-next.patch"
    " && git -C /root/miles apply /tmp/glm5-next.patch",
    "git -C /sgl-workspace/sglang apply - <<'PATCH'\n"
    + (
        Path(__file__).resolve().parents[1] / "patches/sglang-spawn-cuda-rebuild.patch"
    ).read_text()
    + "PATCH",
)
TRAINER_EXTRA_PIP_PACKAGES = base.TRAINER_EXTRA_PIP_PACKAGES
MEGATRON_RUNTIME_PATCHES = [
    "/root/cookbook/miles_disagg/patches/megatron-glm5-next-checkpoints.patch",
    "/root/cookbook/miles_disagg/patches/megatron-r3-dispatch.patch",
]
SGLANG_RUNTIME = SGLangRuntime(
    image=MILES_IMAGE_TAG,
    repository="https://github.com/sgl-project/sglang.git",
    branch="sglang-miles-glm53next",
    commit="9a26e7490f8db83a7fde29ae38f3bbff50ba035c",
    patches=(
        Path(__file__).resolve().parents[1] / "patches/sglang-glm5-next-stitch.patch",
        Path(__file__).resolve().parents[1] / "patches/sglang-kpool-topk-backend.patch",
        Path(__file__).resolve().parents[1] / "patches/sglang-glm-language-model-only.patch",
    ),
    image_run_commands=(
        "uv pip install --system --break-system-packages --no-deps flashinfer-python==0.6.18",
        "uv pip install --system --break-system-packages --no-deps "
        "--index-url https://flashinfer.ai/whl flashinfer-cubin==0.6.18",
        "uv pip install --system --break-system-packages --no-deps "
        "--index-url https://flashinfer.ai/whl/cu130 'flashinfer-jit-cache==0.6.18+cu130'",
    ),
)
SERVED_CHECKPOINT_FORMAT = "nvfp4"
CHECKPOINT_PREP_REQUIRES_GPU = True
MATERIALIZE_BF16_MASTERS = True
USE_MODAL_TORCH_DIST_WRAPPER = True
PREP_ENV = dict(base.PREP_ENV)
SGLANG_SERVER_ENV = {
    key: value
    for key, value in base.SGLANG_SERVER_ENV.items()
    if not key.startswith("SGLANG_DSA_")
}
SGLANG_SERVER_ENV.update(
    SGLANG_DSA_FUSE_TOPK="1",
    SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC="1",
    SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK="large",
)
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
SGLANG_DELTA_UPDATE_MODE = "disk"
SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = False
# Cold KDA/DSA kernel compilation can pause scheduler progress for over 30 seconds.
SIDECAR_WATCHDOG_FAILURE_THRESHOLD = 30
SGLANG_SERVER_ARGS = {
    "--tp": "8",
    "--ep": "8",
    "--load-format": "safetensors",
    "--quantization": "modelopt_fp4",
    "--language-model-only": "",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--context-length": "65544",
    "--disable-radix-cache": "",
    "--attention-backend": "dsa",
    "--dsa-prefill-backend": "tilelang",
    "--dsa-decode-backend": "tilelang",
    "--dsa-topk-backend": "flashinfer",
    "--kv-cache-dtype": "bfloat16",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--mem-fraction-static": "0.7",
    "--chunked-prefill-size": "8192",
    "--max-running-requests": "24",
    "--cuda-graph-max-bs-decode": "24",
    "--enable-return-routed-experts": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
}

modal = deepcopy(base.modal)
modal.rollout_min_containers = 8
modal.rollout_min_ready = 6
modal.rollout_max_containers = 16
modal.rollout_clustered = True
modal.rollout_max_inputs = 24
modal.draft_volume = None
modal.draft_volume_env = None
modal.torch_dist_prep_nodes = 4
modal.torch_dist_prep_gpus_per_node = 8
modal.torch_dist_convert_extra_args = (
    "--tensor-model-parallel-size 1 "
    "--pipeline-model-parallel-size 4 "
    "--expert-model-parallel-size 8 "
    "--decoder-first-pipeline-num-layers 11 "
    "--decoder-last-pipeline-num-layers 12"
)
miles = deepcopy(base.miles)
miles.megatron_model_type = "glm5.3-flash"
miles.model_name = "glm5_next"
miles.actor_num_nodes = 8
miles.global_batch_size = 512
miles.rollout_batch_size = 64
miles.rollout_num_gpus_per_engine = 8
miles.sglang_speculative_algorithm = None
miles.hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
miles.ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
# 64 GPUs: TP8 * PP4 * CP1 * DP2; the 45 layers split 11/11/11/12.
miles.tensor_model_parallel_size = 8
miles.pipeline_model_parallel_size = 4
miles.decoder_first_pipeline_num_layers = 11
miles.decoder_last_pipeline_num_layers = 12
miles.context_parallel_size = 1
miles.expert_model_parallel_size = 16
miles.expert_tensor_parallel_size = 1
miles.allgather_cp = False
miles.num_layers_at_end_in_bf16 = 7
miles.miles_dsa_topk_backend = "torch"
miles.moe_router_use_torch_mm = True
miles.trust_remote_code = True
miles.session_server_port = [30000, 30016]
miles.sglang_server_concurrency = 128
miles.async_max_concurrent_samples = 128
miles.custom_config_path = {
    **base.miles.custom_config_path,
    "rollout_request_timeout_secs": 1800,
}
miles.custom_generate_function_path = "cookbook.miles_disagg.rollout.generate"
miles.custom_checkpoint_completed_hook_path = COMPLETION_HOOK
miles.wandb_group = "glm5-3-flash-nvfp4-swebench-pro"
miles.prometheus_run_name = "glm5-3-flash-nvfp4-swebench-pro"
miles.environment = {
    **base.miles.environment,
    # Inductor fork workers can deadlock after the trainer imports its runtime.
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "MODAL_SWE_SANDBOX_APP": "glm5-3-flash-nvfp4-swebench-pro-sandbox",
    "MODAL_SWE_MODEL_REQUEST_TIMEOUT": "1860",
}
