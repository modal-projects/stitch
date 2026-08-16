"""Profile one Kimi K2.6 NVFP4 delta weight update on Modal.

The entrypoint downloads NVIDIA's pinned serving checkpoint, builds a
standardized element-wise synthetic delta, and runs one verified update.

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/kimi_k2_6_nvfp4_delta_weight_update.py \
      --update-mode cpu --canonical-storage disk
"""

from __future__ import annotations

from pathlib import Path

import modal

from cookbook.common.constants import HF_CACHE_PATH
from cookbook.common.serving_image import build_serving_image
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
from tools.profiling._synthetic_delta import (
    SyntheticDeltaSpec,
    prepare_standard_delta,
    synthetic_delta_profile_id,
)

APP_NAME = "profile-kimi-k2-6-nvfp4-delta-weight-update"
EXPERIMENT = "kimi_k2_6_nvfp4"
ROLLOUT_MODEL = "nvidia/Kimi-K2.6-NVFP4"
ROLLOUT_REVISION = "2fd3a800dedd098b8327eb49e93ebc75f85da19f"
DELTA_MOUNT = "/synthetic-delta"
DELTA_SPEC = SyntheticDeltaSpec(
    checkpoint_format="nvfp4",
    quantized_value_density=0.003,
    high_precision_value_density=0.01,
    # Text-only RL leaves the vision encoder and projector fixed.
    immutable_prefixes=("vision_tower.", "multi_modal_projector."),
)
DELTA_ID = f"kimi-k2-6/{ROLLOUT_REVISION}/{synthetic_delta_profile_id(DELTA_SPEC)}"
DELTA_SOURCE_DIR = f"{DELTA_MOUNT}/{DELTA_ID}"
HF_SNAPSHOT_DIR = (
    f"{HF_CACHE_PATH}/models--nvidia--Kimi-K2.6-NVFP4/snapshots/{ROLLOUT_REVISION}"
)
LOCAL_CHECKPOINT_ROOT = "/local-checkpoint/kimi-k2-6-nvfp4"
BASE_CHECKPOINT_DIR = f"{LOCAL_CHECKPOINT_ROOT}/base"
LOCAL_TARGET_CHECKPOINT_DIR = f"{LOCAL_CHECKPOINT_ROOT}/target"
LOCAL_CANONICAL_CHECKPOINT_DIR = f"{LOCAL_CHECKPOINT_ROOT}/canonical"
SGLANG_CACHE_PATH = "/root/.cache/sglang"

SGLANG_SERVER_ARGS = {
    "--served-model-name": ROLLOUT_MODEL,
    "--load-format": "fastsafetensors",
    "--weight-loader-drop-cache-after-load": "",
    "--trust-remote-code": "",
    "--tool-call-parser": "kimi_k2",
    "--reasoning-parser": "kimi_k2",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
    "--kv-cache-dtype": "fp8_e4m3",
    "--attention-backend": "tokenspeed_mla",
    "--context-length": "32768",
    "--mem-fraction-static": "0.80",
    "--chunked-prefill-size": "16384",
    "--schedule-conservativeness": "0.5",
    "--schedule-policy": "lpm",
    "--max-running-requests": "32",
    "--decode-log-interval": "100",
    "--cuda-graph-max-bs-decode": "32",
    "--random-seed": "42",
    "--skip-server-warmup": "",
}

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
delta_volume = modal.Volume.from_name(
    "stitch-synthetic-deltas", create_if_missing=True, version=2
)
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
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
        str(Path(__file__).resolve().parents[1]),
        remote_path="/root/tools",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)
serving_image = build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
).add_local_dir(
    str(Path(__file__).resolve().parents[1]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)


@app.function(
    image=download_image,
    cpu=32,
    memory=(16 * 1024, 256 * 1024),
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def download_model() -> str:
    return download_snapshot(
        ROLLOUT_MODEL,
        ROLLOUT_REVISION,
        str(HF_CACHE_PATH),
        commit=hf_cache_volume.commit,
    )


@app.function(
    image=serving_image,
    cpu=64,
    memory=(64 * 1024, 512 * 1024),
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume.read_only(),
        DELTA_MOUNT: delta_volume,
    },
    timeout=6 * 60 * 60,
)
def prepare_delta() -> dict:
    return prepare_standard_delta(
        HF_SNAPSHOT_DIR,
        DELTA_SOURCE_DIR,
        spec=DELTA_SPEC,
        commit=delta_volume.commit,
    )


@app.function(
    image=serving_image,
    gpu="B300:4",
    cpu=64,
    memory=(1024 * 1024, 3 * 1024 * 1024),
    # Disk mode retains both the immutable base and a complete mutable target.
    ephemeral_disk=1_572_864,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume.read_only(),
        DELTA_MOUNT: delta_volume.read_only(),
        SGLANG_CACHE_PATH: sglang_cache_volume,
    },
    timeout=4 * 60 * 60,
)
def benchmark(
    update_mode: str,
    canonical_storage: str | None,
    runtime: str,
    sample_id: str,
) -> dict:
    materialize_checkpoint_view(HF_SNAPSHOT_DIR, BASE_CHECKPOINT_DIR)
    return run_delta_weight_update(
        WeightUpdateSpec(
            model_name="Kimi K2.6 NVFP4",
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
        download_model.remote()
        prepare_delta.remote()
    benchmark.remote(
        mode,
        storage,
        modal_runtime_label(),
        sample_id,
    )
