"""Download Kimi K3 and validate one complete MXFP4 delta update on Modal.

Disk destination (the Kimi K3 config's declared update mode):

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/kimi_k3_mxfp4_delta_weight_update.py

CPU destination with the canonical checkpoint on local storage:

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/kimi_k3_mxfp4_delta_weight_update.py \
      --update-mode cpu --canonical-storage disk

``--canonical-storage`` applies only with ``--update-mode cpu``; use
``--canonical-storage memory`` only when the host can retain both the
canonical checkpoint and TP rank images. ``--update-mode disk`` profiles the
disk destination instead of rank-ready CPU staging.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

from cookbook.common.serving_image import build_serving_image
from cookbook.miles_disagg.configs import kimi_k3_mxfp4 as model
from tools.profiling._delta_weight_update import (
    WeightUpdateSpec,
    modal_runtime_label,
    parse_canonical_storage,
    parse_update_destination,
    parse_update_mode,
    run_delta_weight_update,
)
from tools.profiling._hf_checkpoint import (
    download_snapshot,
    materialize_checkpoint_view,
)
from tools.profiling._synthetic_delta import write_full_coverage_delta

APP_NAME = "profile-kimi-k3-mxfp4-delta-weight-update"
EXPERIMENT = "kimi_k3_mxfp4"
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_SNAPSHOT_DIR = (
    f"{HF_CACHE_PATH}/models--moonshotai--Kimi-K3/snapshots/"
    f"{model.ROLLOUT_SOURCE_REVISION}"
)
DELTA_MOUNT = "/synthetic-delta"
DELTA_ID = f"kimi-k3/{model.ROLLOUT_SOURCE_REVISION}/full-coverage-v3"
DELTA_SOURCE_DIR = f"{DELTA_MOUNT}/{DELTA_ID}"
BASE_CHECKPOINT_DIR = "/local-checkpoint/kimi-k3-mxfp4/base"
LOCAL_TARGET_CHECKPOINT_DIR = "/local-checkpoint/kimi-k3-mxfp4/target"
# CPU-mode-only overlay: the config is a clean disk config, so the profiler
# injects the cpu-weight-cache args when profiling the cpu destination.
CPU_CACHE_GROUP_GB = "16"
CANONICAL_CHECKPOINT_DIR = "/local-checkpoint/kimi-k3-mxfp4/canonical"
SGLANG_CACHE_PATH = "/root/.cache/sglang"
_REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(
    "huggingface-cache",
    create_if_missing=True,
    version=2,
)
delta_volume = modal.Volume.from_name(
    "stitch-synthetic-deltas",
    create_if_missing=True,
    version=2,
)
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache",
    create_if_missing=True,
    version=2,
)

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_dir(
        str(_REPO_ROOT / "cookbook"),
        remote_path="/root/cookbook",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        str(_REPO_ROOT / "tools"),
        remote_path="/root/tools",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)
serving_image = build_serving_image(
    hf_cache_path=HF_CACHE_PATH,
    experiment=EXPERIMENT,
    extra_env=getattr(model, "SGLANG_SERVER_ENV", None),
).add_local_dir(
    str(Path(__file__).resolve().parents[1]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)


@app.function(
    image=download_image,
    cpu=32,
    memory=(16 * 1024, 256 * 1024),
    volumes={HF_CACHE_PATH: hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def download_model() -> str:
    return download_snapshot(
        model.ROLLOUT_SOURCE_MODEL,
        model.ROLLOUT_SOURCE_REVISION,
        HF_CACHE_PATH,
        commit=hf_cache_volume.commit,
    )


@app.function(
    image=serving_image,
    cpu=64,
    memory=(64 * 1024, 512 * 1024),
    volumes={
        HF_CACHE_PATH: hf_cache_volume.read_only(),
        DELTA_MOUNT: delta_volume,
    },
    timeout=6 * 60 * 60,
)
def prepare_delta() -> dict:
    metadata_path = Path(DELTA_SOURCE_DIR) / "controlled_delta.json"
    index_path = (
        Path(DELTA_SOURCE_DIR) / "weight_v000001" / "model.safetensors.index.json"
    )
    if metadata_path.is_file() and index_path.is_file():
        result = json.loads(metadata_path.read_text())
        print(f"Reusing synthetic delta at {DELTA_SOURCE_DIR}")
        return result

    shutil.rmtree(DELTA_SOURCE_DIR, ignore_errors=True)
    Path(DELTA_SOURCE_DIR).mkdir(parents=True)
    result = write_full_coverage_delta(
        HF_SNAPSHOT_DIR,
        DELTA_SOURCE_DIR,
    )
    metadata_path.write_text(json.dumps(result, sort_keys=True))
    delta_volume.commit()
    print(f"Committed synthetic delta at {DELTA_SOURCE_DIR}")
    return result


@app.function(
    image=serving_image,
    gpu=f"{model.modal.gpu}:{model.ROLLOUT_NUM_GPUS_PER_ENGINE}",
    cpu=64,
    memory=(model.modal.rollout_memory_mib[0], 4 * 1024 * 1024),
    ephemeral_disk=model.modal.rollout_ephemeral_disk_mib,
    volumes={
        HF_CACHE_PATH: hf_cache_volume.read_only(),
        DELTA_MOUNT: delta_volume.read_only(),
        SGLANG_CACHE_PATH: sglang_cache_volume,
    },
    timeout=6 * 60 * 60,
)
def benchmark(
    update_mode: str,
    canonical_storage: str | None,
    runtime: str,
    sample_id: str,
) -> dict:
    materialize_checkpoint_view(HF_SNAPSHOT_DIR, BASE_CHECKPOINT_DIR)
    server_args = dict(model.SGLANG_SERVER_ARGS)
    server_args["--cpu-weight-cache-max-compile-group-gb"] = CPU_CACHE_GROUP_GB
    return run_delta_weight_update(
        WeightUpdateSpec(
            model_name="Kimi K3 MXFP4",
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
            local_canonical_checkpoint_dir=CANONICAL_CHECKPOINT_DIR,
            server_args=server_args,
            tp_size=model.ROLLOUT_NUM_GPUS_PER_ENGINE,
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
    parsed_mode, parsed_storage = parse_update_destination(
        update_mode,
        canonical_storage,
    )
    if not skip_preparation:
        download_model.remote()
        prepare_delta.remote()
    benchmark.remote(
        parsed_mode,
        parsed_storage,
        modal_runtime_label(),
        sample_id,
    )
