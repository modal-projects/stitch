"""Profile one GLM-4.5-Air FP8 delta weight update on Modal.

The entrypoint downloads the pinned serving checkpoint, prepares a standardized
element-wise synthetic delta, and runs one verified update.

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/glm45_air_fp8_delta_weight_update.py \
      --update-mode cpu --canonical-storage memory
"""

from __future__ import annotations

from pathlib import Path

import modal

from cookbook.common.constants import CHECKPOINTS_PATH
from cookbook.common.hf_download import (
    DOWNLOAD_MAX_CONTAINERS,
    LocalRepoFile,
    download_local_safetensors_file,
    download_local_snapshot,
)
from cookbook.common.serving_image import build_serving_image
from cookbook.miles_disagg.configs import glm45_air_fp8 as model
from tools.profiling._delta_weight_update import (
    WeightUpdateSpec,
    modal_runtime_label,
    parse_canonical_storage,
    parse_update_destination,
    parse_update_mode,
    run_delta_weight_update,
)
from tools.profiling._synthetic_delta import (
    SyntheticDeltaSpec,
    prepare_standard_delta,
    synthetic_delta_profile_id,
)

APP_NAME = "profile-glm45-air-fp8-delta-weight-update"
EXPERIMENT = "glm45_air_fp8"
DELTA_MOUNT = "/synthetic-delta"
DELTA_SPEC = SyntheticDeltaSpec(
    checkpoint_format="fp8",
    quantized_value_density=0.006,
    high_precision_value_density=0.01,
    # The bundled MTP head is fixed during target-model training.
    immutable_prefixes=("model.layers.46.",),
)
DELTA_ID = (
    f"glm45-air/{model.ROLLOUT_SOURCE_REVISION}/"
    f"{synthetic_delta_profile_id(DELTA_SPEC)}"
)
DELTA_SOURCE_DIR = f"{DELTA_MOUNT}/{DELTA_ID}"
BASE_CHECKPOINT_DIR = str(model.ROLLOUT_CHECKPOINT_PATH)
LOCAL_TARGET_CHECKPOINT_DIR = "/local-checkpoint/glm45-air-fp8/target"
LOCAL_CANONICAL_CHECKPOINT_DIR = "/local-checkpoint/glm45-air-fp8/canonical"
SGLANG_CACHE_PATH = "/root/.cache/sglang"

SGLANG_SERVER_ARGS = {
    "--served-model-name": model.ROLLOUT_SOURCE_MODEL,
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
checkpoint_volume = modal.Volume.from_name(
    "miles-checkpoints", create_if_missing=True, version=2
)
delta_volume = modal.Volume.from_name(
    "stitch-synthetic-deltas", create_if_missing=True, version=2
)
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
)
serving_image = build_serving_image(
    hf_cache_path="/root/.cache/huggingface",
    experiment=EXPERIMENT,
).add_local_dir(
    str(Path(__file__).resolve().parents[1]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)


@app.function(
    image=serving_image,
    cpu=4,
    memory=4096,
    max_containers=DOWNLOAD_MAX_CONTAINERS,
    volumes={str(CHECKPOINTS_PATH): checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def _download_base_file(repo_file: LocalRepoFile) -> str:
    return download_local_safetensors_file(repo_file, commit=checkpoint_volume.commit)


@app.function(
    image=serving_image,
    volumes={str(CHECKPOINTS_PATH): checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def prepare_base() -> str:
    """Materialize the pinned public FP8 checkpoint as one immutable artifact."""
    return download_local_snapshot(
        _download_base_file,
        model.ROLLOUT_SOURCE_MODEL,
        model.ROLLOUT_SOURCE_REVISION,
        BASE_CHECKPOINT_DIR,
        volume=checkpoint_volume,
    )


@app.function(
    image=serving_image,
    cpu=64,
    memory=(64 * 1024, 512 * 1024),
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
        DELTA_MOUNT: delta_volume,
    },
    timeout=6 * 60 * 60,
)
def prepare_delta() -> dict:
    return prepare_standard_delta(
        BASE_CHECKPOINT_DIR,
        DELTA_SOURCE_DIR,
        spec=DELTA_SPEC,
        commit=delta_volume.commit,
    )


@app.function(
    image=serving_image,
    gpu="H200:4",
    cpu=64,
    memory=(512 * 1024, 2 * 1024 * 1024),
    ephemeral_disk=512 * 1024,
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
        DELTA_MOUNT: delta_volume.read_only(),
        SGLANG_CACHE_PATH: sglang_cache_volume,
    },
    timeout=2 * 60 * 60,
)
def benchmark(
    update_mode: str,
    canonical_storage: str | None,
    runtime: str,
    sample_id: str,
) -> dict:
    return run_delta_weight_update(
        WeightUpdateSpec(
            model_name="GLM-4.5-Air FP8",
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
            local_canonical_checkpoint_dir=LOCAL_CANONICAL_CHECKPOINT_DIR,
            server_args=SGLANG_SERVER_ARGS,
        ),
        source_dir=DELTA_SOURCE_DIR,
        target_version=1,
        update_mode=parse_update_mode(update_mode),
        canonical_storage=parse_canonical_storage(canonical_storage),
        runtime=runtime,
        sample_id=sample_id,
    )


@app.local_entrypoint()
def main(
    update_mode: str = "disk",
    canonical_storage: str | None = None,
    sample_id: str = "1",
    skip_preparation: bool = False,
) -> None:
    mode, storage = parse_update_destination(update_mode, canonical_storage)
    if not skip_preparation:
        prepare_base.remote()
        prepare_delta.remote()
    benchmark.remote(
        mode,
        storage,
        modal_runtime_label(),
        sample_id,
    )
