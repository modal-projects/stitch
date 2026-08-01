"""GLM-4.5-Air FP8 with the lower-RAM disk delta destination."""

from cookbook.miles_disagg.configs import glm45_air_fp8

APP_NAME = "stitch-glm45-air-fp8-disk"
EXPERIMENT_VOLUME_NAME = glm45_air_fp8.EXPERIMENT_VOLUME_NAME
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
SIDECAR_COMMIT_MODE = glm45_air_fp8.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = glm45_air_fp8.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_DELTA_UPDATE_MODE = "disk"
MEGATRON_RUNTIME_PATCHES = glm45_air_fp8.MEGATRON_RUNTIME_PATCHES
SGLANG_SERVER_ARGS = {
    key: value
    for key, value in glm45_air_fp8.SGLANG_SERVER_ARGS.items()
    if key
    not in {"--enable-cpu-weight-cache", "--cpu-weight-cache-max-compile-group-gb"}
}
modal = glm45_air_fp8.modal
miles = glm45_air_fp8.miles
