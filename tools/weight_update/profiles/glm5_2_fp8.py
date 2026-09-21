"""Profile a verified GLM-5.2 FP8 delta lineage on B300s.

The entrypoint downloads the pinned public checkpoint, constructs one
deterministic element-wise synthetic delta lineage, and verifies repeated and
folded updates.

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/weight_update/profiles/glm5_2_fp8.py \
      --update-mode cpu --canonical-storage memory
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from cookbook.common.constants import HF_CACHE_PATH
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

APP_NAME = "profile-glm5-2-fp8-delta-weight-update"
EXPERIMENT = "glm5_2_fp8"
ROLLOUT_MODEL = "zai-org/GLM-5.2-FP8"
ROLLOUT_REVISION = "ba978f7d347eaf65d22f1a86833408afdb953541"
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
_REPO_ROOT = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/root")

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
    runtime=DEFAULT_SGLANG_RUNTIME,
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
    lineage = prepare_standard_delta(
        local_cached_snapshot(ROLLOUT_MODEL, ROLLOUT_REVISION),
        DELTA_SOURCE_DIR,
        spec=DELTA_SPEC,
        commit=delta_volume.commit,
        output_shard_for_tensor=_pipeline_shard_for_tensor,
    )
    reversal = append_reversal_delta(
        DELTA_SOURCE_DIR,
        reverse_version=4,
        commit=delta_volume.commit,
    )
    return {"lineage": lineage, "reversal": reversal}


_BENCHMARK_FUNCTION_KWARGS = dict(
    image=serving_image,
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


def _benchmark(
    *,
    tp_size: int,
    ep_size: int,
    update_mode: str,
    canonical_storage: str | None,
    runtime: str,
    sample_id: str,
    post_mutation_failure_only: bool = False,
) -> dict:
    if tp_size not in {4, 8}:
        raise ValueError("tp_size must be 4 or 8")
    if ep_size < 1 or tp_size % ep_size != 0:
        raise ValueError("ep_size must be a positive divisor of tp_size")
    materialize_checkpoint_view(
        local_cached_snapshot(ROLLOUT_MODEL, ROLLOUT_REVISION),
        BASE_CHECKPOINT_DIR,
    )
    server_args = dict(SGLANG_SERVER_ARGS)
    if ep_size > 1:
        server_args["--ep-size"] = str(ep_size)
    if tp_size == 8 and ep_size > 1:
        # The pinned DeepGEMM stack cannot launch GLM-5.2 FP8's routed-MoE
        # prefill kernel at the 28-token graph bucket on B300. Keep graph
        # execution and pad that bucket to the next captured shape instead.
        prefill_graph_bs = (
            list(range(4, 28, 4))
            + [32]
            + list(range(48, 257, 16))
            + list(range(288, 513, 32))
            + list(range(576, 1025, 64))
            + list(range(1280, 2049, 256))
        )
        server_args["--cuda-graph-config"] = json.dumps(
            {"prefill": {"bs": prefill_graph_bs}}, separators=(",", ":")
        )
    spec = WeightUpdateSpec(
        model_name="GLM-5.2 FP8",
        base_checkpoint_dir=BASE_CHECKPOINT_DIR,
        local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
        local_canonical_checkpoint_dir=LOCAL_CANONICAL_CHECKPOINT_DIR,
        server_args=server_args,
        tp_size=tp_size,
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
            served_version=4,
            failure_version=5,
            **common,
        )
    return run_delta_weight_update(spec, target_versions=(1, 3, 4), **common)


@app.function(gpu="B300:4", **_BENCHMARK_FUNCTION_KWARGS)
def benchmark_tp4(
    ep_size: int,
    update_mode: str,
    canonical_storage: str | None,
    runtime: str,
    sample_id: str,
    post_mutation_failure_only: bool = False,
) -> dict:
    return _benchmark(
        tp_size=4,
        ep_size=ep_size,
        update_mode=update_mode,
        canonical_storage=canonical_storage,
        runtime=runtime,
        sample_id=sample_id,
        post_mutation_failure_only=post_mutation_failure_only,
    )


@app.function(gpu="B300:8", **_BENCHMARK_FUNCTION_KWARGS)
def benchmark_tp8(
    ep_size: int,
    update_mode: str,
    canonical_storage: str | None,
    runtime: str,
    sample_id: str,
    post_mutation_failure_only: bool = False,
) -> dict:
    return _benchmark(
        tp_size=8,
        ep_size=ep_size,
        update_mode=update_mode,
        canonical_storage=canonical_storage,
        runtime=runtime,
        sample_id=sample_id,
        post_mutation_failure_only=post_mutation_failure_only,
    )


@app.local_entrypoint()
def main(
    update_mode: str = "disk",
    canonical_storage: str | None = None,
    tp_size: int = 4,
    ep_size: int = 1,
    sample_id: str = "1",
    skip_preparation: bool = False,
    post_mutation_failure_only: bool = False,
) -> None:
    mode, storage = parse_update_destination(update_mode, canonical_storage)
    if tp_size not in {4, 8}:
        raise ValueError("tp_size must be 4 or 8")
    if ep_size < 1 or tp_size % ep_size != 0:
        raise ValueError("ep_size must be a positive divisor of tp_size")
    if not skip_preparation:
        download_model.remote()
        prepare_delta.remote()
    benchmark = benchmark_tp4 if tp_size == 4 else benchmark_tp8
    benchmark.remote(
        ep_size,
        mode,
        storage,
        modal_runtime_label(),
        sample_id,
        post_mutation_failure_only,
    )
