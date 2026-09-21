"""Download Kimi K3 and validate a complete MXFP4 delta lineage on Modal.

Disk destination:

    uv run --extra modal modal run -d \
      tools/weight_update/profiles/kimi_k3_mxfp4.py

CPU destination with the canonical checkpoint on local storage:

    uv run --extra modal modal run -d \
      tools/weight_update/profiles/kimi_k3_mxfp4.py \
      --update-mode cpu --canonical-storage disk

``--canonical-storage`` applies only with ``--update-mode cpu``; use
``--canonical-storage memory`` only when the host can retain both the
canonical checkpoint and TP rank images. ``--update-mode disk`` profiles the
disk destination instead of rank-ready CPU staging.
"""

from __future__ import annotations

from pathlib import Path

import modal

from cookbook.common.hf_download import (
    DOWNLOAD_MAX_CONTAINERS,
    CachedRepoFile,
    download_cached_safetensors_file,
    local_cached_snapshot,
)
from cookbook.common.serving_image import DEFAULT_SGLANG_RUNTIME, build_serving_image
from tools.weight_update.benchmark import (
    WeightUpdateSpec,
    modal_runtime_label,
    parse_canonical_storage,
    parse_update_destination,
    parse_update_mode,
    run_delta_weight_update,
    run_post_mutation_failure,
)
from tools.weight_update.hf import (
    download_snapshot,
    materialize_checkpoint_view,
)
from tools.weight_update.synthetic_delta import (
    SyntheticDeltaSpec,
    append_reversal_delta,
    prepare_standard_delta,
    synthetic_delta_profile_id,
)

ROLLOUT_MODEL = "moonshotai/Kimi-K3"
ROLLOUT_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
ROLLOUT_GPUS = 8
GPU = "B300"
MEMORY_MIB = (1048576, 4194304)
EPHEMERAL_DISK_MIB = 2097152
SGLANG_SERVER_ARGS = {
    "--tp": "8",
    "--trust-remote-code": "",
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--weight-loader-drop-cache-after-load": "",
    "--weight-update-max-compile-group-gb": "16",
    "--dist-timeout": "3600",
    # Initializing either CPU destination moves a 1.56 TB canonical checkpoint.
    "--watchdog-timeout": "3600",
    "--context-length": "1048576",
    "--max-running-requests": "32",
    "--cuda-graph-max-bs-decode": "32",
    "--mem-fraction-static": "0.85",
    "--kv-cache-dtype": "fp8_e4m3",
    "--mamba-ssm-dtype": "bfloat16",
    "--mamba-radix-cache-strategy": "extra_buffer_lazy",
    "--chunked-prefill-size": "16384",
    "--schedule-policy": "lpm",
    "--mm-feature-transport": "cuda_ipc",
    "--mm-processor-worker-num": "2",
    "--mm-io-worker-num": "16",
    "--reasoning-parser": "kimi_k3",
    "--tool-call-parser": "kimi_k3",
}

APP_NAME = "profile-kimi-k3-mxfp4-delta-weight-update"
EXPERIMENT = "kimi_k3_mxfp4"
HF_CACHE_PATH = "/root/.cache/huggingface"
DELTA_MOUNT = "/synthetic-delta"
DELTA_SPEC = SyntheticDeltaSpec(
    checkpoint_format="mxfp4",
    quantized_value_density=0.003,
    high_precision_value_density=0.01,
    # Text-only RL leaves the vision encoder and projector fixed.
    immutable_prefixes=("vision_tower.", "mm_projector."),
)
DELTA_ID = f"kimi-k3/{ROLLOUT_REVISION}/{synthetic_delta_profile_id(DELTA_SPEC)}"
DELTA_SOURCE_DIR = f"{DELTA_MOUNT}/{DELTA_ID}"
BASE_CHECKPOINT_DIR = "/local-checkpoint/kimi-k3-mxfp4/base"
LOCAL_TARGET_CHECKPOINT_DIR = "/local-checkpoint/kimi-k3-mxfp4/target"
CPU_CACHE_GROUP_GB = "16"
CANONICAL_CHECKPOINT_DIR = "/local-checkpoint/kimi-k3-mxfp4/canonical"
SGLANG_CACHE_PATH = "/root/.cache/sglang"
_REPO_ROOT = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root")

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
    extra_env=None,
    runtime=DEFAULT_SGLANG_RUNTIME,
).add_local_dir(
    str(Path(__file__).resolve().parents[2]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)


@app.function(
    image=download_image,
    cpu=4,
    memory=4096,
    max_containers=DOWNLOAD_MAX_CONTAINERS,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def _download_model_file(repo_file: CachedRepoFile) -> str:
    return download_cached_safetensors_file(repo_file, commit=hf_cache_volume.commit)


@app.function(
    image=download_image,
    volumes={HF_CACHE_PATH: hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def download_model() -> str:
    return download_snapshot(
        _download_model_file,
        ROLLOUT_MODEL,
        ROLLOUT_REVISION,
        volume=hf_cache_volume,
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
    lineage = prepare_standard_delta(
        local_cached_snapshot(
            ROLLOUT_MODEL,
            ROLLOUT_REVISION,
        ),
        DELTA_SOURCE_DIR,
        spec=DELTA_SPEC,
        commit=delta_volume.commit,
    )
    reversal = append_reversal_delta(
        DELTA_SOURCE_DIR,
        reverse_version=4,
        commit=delta_volume.commit,
    )
    return {"lineage": lineage, "reversal": reversal}


@app.function(
    image=serving_image,
    gpu=f"{GPU}:{ROLLOUT_GPUS}",
    cpu=64,
    memory=MEMORY_MIB,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
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
    post_mutation_failure_only: bool = False,
) -> dict:
    materialize_checkpoint_view(
        local_cached_snapshot(
            ROLLOUT_MODEL,
            ROLLOUT_REVISION,
        ),
        BASE_CHECKPOINT_DIR,
    )
    spec = WeightUpdateSpec(
        model_name="Kimi K3 MXFP4",
        base_checkpoint_dir=BASE_CHECKPOINT_DIR,
        local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
        local_canonical_checkpoint_dir=CANONICAL_CHECKPOINT_DIR,
        server_args=SGLANG_SERVER_ARGS,
        tp_size=ROLLOUT_GPUS,
        max_compile_group_gb=int(CPU_CACHE_GROUP_GB),
    )
    common = dict(
        source_dir=DELTA_SOURCE_DIR,
        update_mode=parse_update_mode(update_mode),
        canonical_storage=parse_canonical_storage(canonical_storage),
        runtime=runtime,
        sample_id=sample_id,
    )
    if post_mutation_failure_only:
        return run_post_mutation_failure(
            spec,
            served_version=0,
            failure_version=1,
            **common,
        )
    return run_delta_weight_update(spec, target_versions=(1, 3, 4), **common)


@app.local_entrypoint()
def main(
    update_mode: str = "disk",
    canonical_storage: str | None = None,
    sample_id: str = "1",
    skip_preparation: bool = False,
    post_mutation_failure_only: bool = False,
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
        post_mutation_failure_only,
    )
