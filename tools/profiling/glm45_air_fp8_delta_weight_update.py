"""Profile one GLM-4.5-Air FP8 delta weight update on Modal.

CPU destination with Modal's default runtime:

    uv run --extra modal modal run -d \
      tools/profiling/glm45_air_fp8_delta_weight_update.py \
      --update-mode cpu

Disk destination with the runc runtime:

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/glm45_air_fp8_delta_weight_update.py \
      --update-mode disk
"""

from __future__ import annotations

from pathlib import Path

import modal

from cookbook.common.serving_image import build_serving_image
from tools.profiling._delta_weight_update import (
    WeightUpdateSpec,
    modal_runtime_label,
    parse_update_mode,
    run_delta_weight_update,
)

APP_NAME = "profile-glm45-air-fp8-delta-weight-update"
EXPERIMENT = "glm45_air_fp8"
BASE_VOLUME_NAME = "miles-prep-checkpoints"
DELTA_VOLUME_NAME = "stitch-delta-glm45-air-fp8"
SGLANG_CACHE_VOLUME_NAME = "sglang-cache"

BASE_MOUNT = "/prep"
BASE_CHECKPOINT_DIR = "/prep/glm45-air/fp8"
DELTA_MOUNT = "/delta-bulletin"
SGLANG_CACHE_MOUNT = "/root/.cache/sglang"
LOCAL_CHECKPOINT_ROOT = "/local-checkpoint/glm45-air-fp8"
LOCAL_TARGET_CHECKPOINT_DIR = f"{LOCAL_CHECKPOINT_ROOT}/target"

DEFAULT_RUN_ID = "e784ed3b/6bf0bd571c0a"
DEFAULT_TARGET_VERSION = 1

SGLANG_SERVER_ARGS = {
    "--served-model-name": "zai-org/GLM-4.5-Air-FP8",
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--weight-loader-drop-cache-after-load": "",
    "--dtype": "auto",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm",
    "--dist-timeout": "3600",
    "--context-length": "32768",
    "--mem-fraction-static": "0.70",
    "--chunked-prefill-size": "8192",
    "--max-prefill-tokens": "16384",
    "--cuda-graph-max-bs-prefill": "2048",
    "--max-running-requests": "32",
    "--decode-log-interval": "100",
    "--random-seed": "42",
    "--skip-server-warmup": "",
}

app = modal.App(APP_NAME)
base_volume = modal.Volume.from_name(BASE_VOLUME_NAME, version=2)
delta_volume = modal.Volume.from_name(DELTA_VOLUME_NAME, version=2)
sglang_cache_volume = modal.Volume.from_name(SGLANG_CACHE_VOLUME_NAME, version=2)
image = build_serving_image(
    hf_cache_path="/root/.cache/huggingface",
    experiment=EXPERIMENT,
).add_local_dir(
    str(Path(__file__).resolve().parents[1]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)


@app.function(
    image=image,
    gpu="H200:4",
    cpu=64,
    memory=(512 * 1024, 2 * 1024 * 1024),
    ephemeral_disk=512 * 1024,
    volumes={
        BASE_MOUNT: base_volume,
        DELTA_MOUNT: delta_volume,
        SGLANG_CACHE_MOUNT: sglang_cache_volume,
    },
    timeout=2 * 60 * 60,
)
def benchmark(
    run_id: str,
    target_version: int,
    update_mode: str,
    runtime: str,
    sample_id: str,
) -> dict:
    delta_volume.reload()
    return run_delta_weight_update(
        WeightUpdateSpec(
            model_name="GLM-4.5-Air FP8",
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
            server_args=SGLANG_SERVER_ARGS,
        ),
        source_dir=f"{DELTA_MOUNT}/{run_id}",
        target_version=target_version,
        update_mode=parse_update_mode(update_mode),
        runtime=runtime,
        sample_id=sample_id,
    )


@app.local_entrypoint()
def main(
    run_id: str = DEFAULT_RUN_ID,
    target_version: int = DEFAULT_TARGET_VERSION,
    update_mode: str = "cpu",
    sample_id: str = "1",
) -> None:
    benchmark.remote(
        run_id,
        target_version,
        parse_update_mode(update_mode),
        modal_runtime_label(),
        sample_id,
    )
