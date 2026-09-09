"""GLM-5.3 NVFP4 preparation and training with GLM-5.2 runtime settings."""

from copy import deepcopy

from cookbook.common.constants import CHECKPOINTS_PATH, DRAFT_PATH
from cookbook.miles_disagg.configs import glm5_2_nvfp4 as base

APP_NAME = "stitch-glm5-3-nvfp4"
EXPERIMENT_VOLUME_NAME = "stitch-miles-glm5-3-nvfp4"
SOURCE_MODEL = "zai-org/GLM-5.3-BF16"
SOURCE_REVISION = "9d2398f478cab2de883137db3a36ad2c96205e24"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-bf16"
ROLLOUT_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-nvfp4"
TORCH_DIST_CHECKPOINT_PATH = CHECKPOINTS_PATH / "glm5-3-torch-dist"

DFLASH_VOLUME = "dflash_data"
DFLASH_CHECKPOINT_PATH = (
    DRAFT_PATH
    / "train/glm53_nvfp4_6l_swa_k3taps_rope2m_32x/trainer-0/draft-step-22500"
)
DFLASH_SERVER_ARGS = {
    **base.DFLASH_SERVER_ARGS,
    "--speculative-draft-model-path": str(DFLASH_CHECKPOINT_PATH),
}
SGLANG_SERVER_ARGS = {**base.SGLANG_SERVER_ARGS, **DFLASH_SERVER_ARGS}
SGLANG_SERVER_ENV = dict(base.SGLANG_SERVER_ENV)
SGLANG_DELTA_UPDATE_MODE = base.SGLANG_DELTA_UPDATE_MODE
SIDECAR_COMMIT_MODE = base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = base.SIDECAR_FLUSH_CACHE_ON_COMMIT
LOCAL_CHECKPOINT_PATH = base.LOCAL_CHECKPOINT_PATH
MEGATRON_RUNTIME_PATCHES = list(base.MEGATRON_RUNTIME_PATCHES)

SERVED_CHECKPOINT_FORMAT = base.SERVED_CHECKPOINT_FORMAT
CHECKPOINT_PREP_REQUIRES_GPU = base.CHECKPOINT_PREP_REQUIRES_GPU
MATERIALIZE_BF16_MASTERS = base.MATERIALIZE_BF16_MASTERS
USE_MODAL_TORCH_DIST_WRAPPER = base.USE_MODAL_TORCH_DIST_WRAPPER
TRAINER_EXTRA_PIP_PACKAGES = base.TRAINER_EXTRA_PIP_PACKAGES
TRAINER_IMAGE_RUN_COMMANDS = base.TRAINER_IMAGE_RUN_COMMANDS
PREP_ENV = dict(base.PREP_ENV)

modal = deepcopy(base.modal)
modal.rollout_min_containers = 16
modal.rollout_min_ready = 12
modal.draft_volume = DFLASH_VOLUME
modal.draft_volume_env = "glm-bringup"
miles = deepcopy(base.miles)
miles.actor_num_nodes = 32
miles.global_batch_size = 512
miles.rollout_batch_size = 64
miles.session_server_startup_timeout_seconds = 600
miles.hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
miles.ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
miles.wandb_group = "glm5-3-nvfp4-swebench-pro"
miles.prometheus_run_name = "glm5-3-nvfp4-swebench-pro"
miles.environment = {
    **base.miles.environment,
    "MODAL_SWE_SANDBOX_APP": "glm5-3-nvfp4-swebench-pro-sandbox",
}
