"""Profile one GLM-4.5-Air FP8 delta weight update on Modal.

The entrypoint downloads the pinned serving checkpoint, prepares a standardized
element-wise synthetic delta, and runs one verified update.

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/glm45_air_fp8_delta_weight_update.py \
      --update-mode cpu
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import modal

from cookbook.common.constants import CHECKPOINTS_PATH
from cookbook.common.serving_image import build_serving_image
from cookbook.miles_disagg.configs import glm45_air_fp8 as model
from tools.profiling._delta_weight_update import (
    WeightUpdateSpec,
    modal_runtime_label,
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
SGLANG_CACHE_PATH = "/root/.cache/sglang"
SOURCE_MARKER = ".stitch-source.json"

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
    cpu=32,
    memory=(16 * 1024, 256 * 1024),
    volumes={str(CHECKPOINTS_PATH): checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def prepare_base() -> str:
    """Materialize the pinned public FP8 checkpoint as one immutable artifact."""

    from huggingface_hub import HfApi, snapshot_download

    checkpoint_volume.reload()
    destination = Path(BASE_CHECKPOINT_DIR)
    expected = {
        "repo_id": model.ROLLOUT_SOURCE_MODEL,
        "revision": model.ROLLOUT_SOURCE_REVISION,
    }
    marker = destination / SOURCE_MARKER
    index = destination / "model.safetensors.index.json"
    if marker.is_file() and index.is_file():
        if json.loads(marker.read_text()) == expected:
            print(f"Reusing pinned checkpoint at {destination}")
            return str(destination)

    staging = destination.with_name(f"{destination.name}.partial")
    staging.mkdir(parents=True, exist_ok=True)
    files = HfApi().list_repo_files(
        repo_id=model.ROLLOUT_SOURCE_MODEL,
        revision=model.ROLLOUT_SOURCE_REVISION,
    )
    weights = sorted(name for name in files if name.endswith(".safetensors"))
    batches = [sorted(set(files) - set(weights))]
    batches.extend(weights[offset : offset + 4] for offset in range(0, len(weights), 4))
    for batch_index, batch in enumerate(batches, start=1):
        snapshot_download(
            repo_id=model.ROLLOUT_SOURCE_MODEL,
            revision=model.ROLLOUT_SOURCE_REVISION,
            local_dir=staging,
            allow_patterns=batch,
            max_workers=4,
        )
        checkpoint_volume.commit()
        print(f"Committed checkpoint batch {batch_index}/{len(batches)}")

    shutil.rmtree(staging / ".cache", ignore_errors=True)
    (staging / SOURCE_MARKER).write_text(json.dumps(expected, sort_keys=True) + "\n")
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(staging, destination)
    checkpoint_volume.commit()
    print(f"Prepared pinned checkpoint at {destination}")
    return str(destination)


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
def benchmark(update_mode: str, runtime: str, sample_id: str) -> dict:
    return run_delta_weight_update(
        WeightUpdateSpec(
            model_name="GLM-4.5-Air FP8",
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
            server_args=SGLANG_SERVER_ARGS,
        ),
        source_dir=DELTA_SOURCE_DIR,
        target_version=1,
        update_mode=parse_update_mode(update_mode),
        runtime=runtime,
        sample_id=sample_id,
    )


@app.local_entrypoint()
def main(update_mode: str = "cpu", sample_id: str = "1") -> None:
    prepare_base.remote()
    prepare_delta.remote()
    benchmark.remote(parse_update_mode(update_mode), modal_runtime_label(), sample_id)
