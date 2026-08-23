"""Profile one GLM-5.2 FP8 delta weight update on four B300s.

The entrypoint downloads the pinned public checkpoint, constructs one
deterministic element-wise synthetic delta, and verifies one complete update.

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/glm5_2_fp8_delta_weight_update.py \
      --update-mode cpu --canonical-storage memory
"""

from __future__ import annotations

from pathlib import Path

import modal

from cookbook.common.constants import HF_CACHE_PATH
from cookbook.common.hf_download import (
    DOWNLOAD_MAX_CONTAINERS,
    CachedRepoFile,
    download_cached_safetensors_file,
    local_cached_snapshot,
)
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

APP_NAME = "profile-glm5-2-fp8-delta-weight-update"
EXPERIMENT = "glm5_2_fp8"
ROLLOUT_MODEL = "zai-org/GLM-5.2-FP8"
ROLLOUT_REVISION = "ba978f7d347eaf65d22f1a86833408afdb953541"
ROLLOUT_GPUS = 4
DELTA_MOUNT = "/synthetic-delta"
DELTA_SPEC = SyntheticDeltaSpec(
    checkpoint_format="fp8",
    quantized_value_density=0.006,
    high_precision_value_density=0.01,
    output_shards=4,
    output_shard_layout="miles-pp4-layer-placement-v1",
    # The target-model optimizer does not update the bundled MTP layer.
    immutable_prefixes=("model.layers.78.",),
)
DELTA_ID = f"glm5-2-fp8/{ROLLOUT_REVISION}/{synthetic_delta_profile_id(DELTA_SPEC)}"
DELTA_SOURCE_DIR = f"{DELTA_MOUNT}/{DELTA_ID}"
LOCAL_CHECKPOINT_ROOT = "/local-checkpoint/glm5-2-fp8"
BASE_CHECKPOINT_DIR = f"{LOCAL_CHECKPOINT_ROOT}/base"
LOCAL_TARGET_CHECKPOINT_DIR = f"{LOCAL_CHECKPOINT_ROOT}/target"
LOCAL_CANONICAL_CHECKPOINT_DIR = f"{LOCAL_CHECKPOINT_ROOT}/canonical"
SGLANG_CACHE_PATH = "/root/.cache/sglang"
_REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")

SGLANG_SERVER_ARGS = {
    "--served-model-name": ROLLOUT_MODEL,
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--weight-loader-drop-cache-after-load": "",
    "--dtype": "auto",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
    "--context-length": "32768",
    "--attention-backend": "dsa",
    "--dsa-prefill-backend": "flashmla_sparse",
    "--dsa-decode-backend": "flashmla_kv",
    "--dsa-topk-backend": "flashinfer",
    "--page-size": "64",
    "--ep-size": str(ROLLOUT_GPUS),
    "--moe-dense-tp-size": "1",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--mem-fraction-static": "0.80",
    "--chunked-prefill-size": "16384",
    "--max-running-requests": "24",
    "--decode-log-interval": "100",
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
        str(_REPO_ROOT / "tools"),
        remote_path="/root/tools",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        str(_REPO_ROOT / "cookbook"),
        remote_path="/root/cookbook",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)
serving_image = build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_env={"SGLANG_DG_CACHE_DIR": f"{SGLANG_CACHE_PATH}/deep_gemm"},
).add_local_dir(
    str(_REPO_ROOT / "tools"),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)


def _pipeline_shard_for_tensor(name: str) -> int:
    if name == "model.embed_tokens.weight":
        return 0
    if name in {"lm_head.weight", "model.norm.weight"}:
        return 3
    prefix = "model.layers."
    if not name.startswith(prefix):
        raise ValueError(f"no pipeline placement for tensor {name!r}")
    layer = int(name[len(prefix) :].partition(".")[0])
    for stage, layer_end in enumerate((18, 38, 58, 78)):
        if layer < layer_end:
            return stage
    raise ValueError(f"layer {layer} is outside the target model")


@app.function(
    image=download_image,
    cpu=4,
    memory=4096,
    max_containers=DOWNLOAD_MAX_CONTAINERS,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def _download_model_file(repo_file: CachedRepoFile) -> str:
    return download_cached_safetensors_file(repo_file, commit=hf_cache_volume.commit)


@app.function(
    image=download_image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
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
        str(HF_CACHE_PATH): hf_cache_volume.read_only(),
        DELTA_MOUNT: delta_volume,
    },
    timeout=6 * 60 * 60,
)
def prepare_delta() -> dict:
    return prepare_standard_delta(
        local_cached_snapshot(ROLLOUT_MODEL, ROLLOUT_REVISION),
        DELTA_SOURCE_DIR,
        spec=DELTA_SPEC,
        commit=delta_volume.commit,
        output_shard_for_tensor=_pipeline_shard_for_tensor,
    )


@app.function(
    image=serving_image,
    gpu=f"B300:{ROLLOUT_GPUS}",
    cpu=64,
    memory=(1024 * 1024, 3 * 1024 * 1024),
    # Disk mode reconstructs the complete 756 GB target on local storage.
    ephemeral_disk=2 * 1024 * 1024,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume.read_only(),
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
    materialize_checkpoint_view(
        local_cached_snapshot(ROLLOUT_MODEL, ROLLOUT_REVISION),
        BASE_CHECKPOINT_DIR,
    )
    return run_delta_weight_update(
        WeightUpdateSpec(
            model_name="GLM-5.2 FP8",
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
            local_canonical_checkpoint_dir=LOCAL_CANONICAL_CHECKPOINT_DIR,
            server_args=SGLANG_SERVER_ARGS,
            tp_size=ROLLOUT_GPUS,
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
