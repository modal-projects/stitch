"""Download Kimi K3 and validate one complete MXFP4 delta update on Modal.

Run the CPU destination with the canonical checkpoint on local storage:

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/kimi_k3_mxfp4_delta_weight_update.py \
      --update-mode cpu --canonical-storage disk

Use ``--canonical-storage memory`` only when the host can retain both the
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
    parse_update_mode,
    run_delta_weight_update,
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
    runtime=model.SGLANG_RUNTIME,
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
    from huggingface_hub import HfApi, snapshot_download

    files = HfApi().list_repo_files(
        repo_id=model.ROLLOUT_SOURCE_MODEL,
        revision=model.ROLLOUT_SOURCE_REVISION,
    )
    weights = sorted(path for path in files if path.endswith(".safetensors"))
    metadata = sorted(set(files) - set(weights))
    batches = [metadata]
    batches.extend(weights[offset : offset + 4] for offset in range(0, len(weights), 4))
    for index, batch in enumerate(batches, start=1):
        snapshot_download(
            repo_id=model.ROLLOUT_SOURCE_MODEL,
            revision=model.ROLLOUT_SOURCE_REVISION,
            cache_dir=HF_CACHE_PATH,
            allow_patterns=batch,
            max_workers=4,
        )
        hf_cache_volume.commit()
        print(f"Committed checkpoint download batch {index}/{len(batches)}")

    path = snapshot_download(
        repo_id=model.ROLLOUT_SOURCE_MODEL,
        revision=model.ROLLOUT_SOURCE_REVISION,
        cache_dir=HF_CACHE_PATH,
        local_files_only=True,
    )
    print(
        f"Downloaded {model.ROLLOUT_SOURCE_MODEL}"
        f"@{model.ROLLOUT_SOURCE_REVISION} to {path}"
    )
    return path


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


def _materialize_checkpoint_view() -> None:
    """Give trusted remote code real sibling files without copying model weights."""

    source = Path(HF_SNAPSHOT_DIR)
    target = Path(BASE_CHECKPOINT_DIR)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for path in source.rglob("*"):
        destination = target / path.relative_to(source)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.suffix == ".safetensors":
            destination.symlink_to(path.resolve())
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination, follow_symlinks=True)


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
    canonical_storage: str,
    runtime: str,
    sample_id: str,
) -> dict:
    _materialize_checkpoint_view()
    server_args = dict(model.SGLANG_SERVER_ARGS)
    if canonical_storage == "memory":
        server_args.pop("--cpu-weight-cache-canonical-checkpoint-dir", None)
    elif canonical_storage != "disk":
        raise ValueError("canonical_storage must be 'memory' or 'disk'")
    return run_delta_weight_update(
        WeightUpdateSpec(
            model_name="Kimi K3 MXFP4",
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
            server_args=server_args,
            tp_size=model.ROLLOUT_NUM_GPUS_PER_ENGINE,
        ),
        source_dir=DELTA_SOURCE_DIR,
        target_version=1,
        update_mode=parse_update_mode(update_mode),
        runtime=runtime,
        sample_id=sample_id,
    )


@app.local_entrypoint()
def main(
    update_mode: str = "cpu",
    canonical_storage: str = "disk",
    sample_id: str = "1",
) -> None:
    parsed_mode = parse_update_mode(update_mode)
    download_model.remote()
    prepare_delta.remote()
    benchmark.remote(
        parsed_mode,
        canonical_storage,
        modal_runtime_label(),
        sample_id,
    )
