"""Profile a verified GLM-5.2 mixed NVFP4/BF16 delta lineage on Modal.

CPU destination:

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/weight_update/profiles/glm5_2_nvfp4.py \
      --update-mode cpu --canonical-storage memory

Disk destination:

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/weight_update/profiles/glm5_2_nvfp4.py \
      --update-mode disk
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import modal

from cookbook.common.constants import CHECKPOINTS_PATH, HF_CACHE_PATH
from cookbook.common.hf_download import (
    DOWNLOAD_MAX_CONTAINERS,
    CachedRepoFile,
    download_cached_safetensors_file,
    download_cached_snapshot,
)
from cookbook.common.serving_image import DEFAULT_SGLANG_RUNTIME, build_serving_image
from cookbook.miles_disagg import prep, trainer_image
from tools.weight_update.benchmark import (
    WeightUpdateSpec,
    modal_runtime_label,
    parse_canonical_storage,
    parse_update_destination,
    parse_update_mode,
    run_delta_weight_update,
    run_post_mutation_failure,
)
from tools.weight_update.synthetic_delta import (
    SyntheticDeltaSpec,
    append_reversal_delta,
    prepare_standard_delta,
    synthetic_delta_profile_id,
)

SOURCE_MODEL = "zai-org/GLM-5.2"
SOURCE_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
BF16_CHECKPOINT_PATH = Path("/checkpoints/glm5-2-bf16")
ROLLOUT_CHECKPOINT_PATH = Path("/checkpoints/glm5-2-nvfp4")
ROLLOUT_GPUS_PER_ENGINE = 4

PREP_ENV = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
    "NVTE_NVFP4_4OVER6": "all",
    "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all",
    "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
    "NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
}
TRAINER_EXTRA_PIP_PACKAGES = (
    "harbor[modal,huggingface]==0.20.0",
    "mini-swe-agent==2.4.5",
    "swebench==4.1.0",
    "modal==1.5.3",
)
TRAINER_IMAGE_RUN_COMMANDS = (
    "uv pip install --system --break-system-packages flashinfer-python==0.6.15.post1",
    "uv pip install --system --break-system-packages --no-deps --index-url "
    "https://flashinfer.ai/whl flashinfer-cubin==0.6.15.post1",
    "uv pip install --system --break-system-packages --no-deps --index-url "
    "https://flashinfer.ai/whl/cu130 flashinfer-jit-cache==0.6.15.post1+cu130",
)

SGLANG_SERVER_ARGS = {
    "--tp": "4",
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--weight-update-max-compile-group-gb": "8",
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--watchdog-timeout": "3600",
    "--quantization": "modelopt_fp4",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm47",
    "--context-length": "65544",
    "--attention-backend": "dsa",
    "--kv-cache-dtype": "fp8_e4m3",
    "--dsa-prefill-backend": "flashmla_sparse",
    "--dsa-decode-backend": "flashmla_kv",
    "--dsa-topk-backend": "flashinfer",
    "--page-size": "64",
    "--moe-runner-backend": "flashinfer_trtllm_routed",
    "--disable-shared-experts-fusion": "",
    "--mem-fraction-static": "0.8",
    "--chunked-prefill-size": "16384",
    "--max-running-requests": "24",
    "--max-queued-requests": "4",
    "--schedule-conservativeness": "1.0",
    "--schedule-policy": "lpm",
    "--cuda-graph-config": '{"decode":{"backend":"full","max_bs":24,"bs":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]}}',
    "--enable-return-routed-experts": "",
}
SGLANG_SERVER_ENV = {
    "FLASHINFER_NVFP4_4OVER6": "1",
    "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
    "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
    "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "0",
    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
    "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
    "INDEXER_ROPE_NEOX_STYLE": "0",
    "NVSHMEM_DISABLE_NCCL": "1",
}

MILES = SimpleNamespace(
    hf_checkpoint=str(ROLLOUT_CHECKPOINT_PATH),
    pipeline_model_parallel_size=4,
    decoder_first_pipeline_num_layers=18,
    decoder_last_pipeline_num_layers=20,
    num_layers_at_start_in_bf16=3,
    num_layers_at_end_in_bf16=12,
    extra_high_precision_layers_hf=[".shared_experts."],
)
MODAL = SimpleNamespace(
    gpu="B300",
    trainer_memory_mib=(1048576, 3145728),
    rollout_memory_mib=(1048576, 3145728),
)
MODEL = SimpleNamespace(
    SOURCE_MODEL=SOURCE_MODEL,
    SOURCE_REVISION=SOURCE_REVISION,
    BF16_CHECKPOINT_PATH=BF16_CHECKPOINT_PATH,
    PREP_ENV=PREP_ENV,
    TRAINER_EXTRA_PIP_PACKAGES=TRAINER_EXTRA_PIP_PACKAGES,
    TRAINER_IMAGE_RUN_COMMANDS=TRAINER_IMAGE_RUN_COMMANDS,
    SERVED_CHECKPOINT_FORMAT="nvfp4",
    MATERIALIZE_BF16_MASTERS=False,
    UNPACK_FUSED_EXPERTS=False,
    DISABLE_HF_XET=False,
    DISABLE_HF_TRANSFER=False,
    miles=MILES,
)

APP_NAME = "profile-glm5-2-nvfp4-delta-weight-update"
EXPERIMENT = "glm5_2_nvfp4"
DELTA_MOUNT = "/synthetic-delta"
_TARGET_MODEL_LAYERS = 78


def _pipeline_layer_ends() -> tuple[int, ...]:
    stages = MILES.pipeline_model_parallel_size
    first = MILES.decoder_first_pipeline_num_layers
    last = MILES.decoder_last_pipeline_num_layers
    if stages < 2 or first is None or last is None:
        raise ValueError("GLM-5.2 profiling requires explicit first/last PP stages")
    middle_stages = stages - 2
    remaining = _TARGET_MODEL_LAYERS - first - last
    if middle_stages:
        middle_layers, remainder = divmod(remaining, middle_stages)
        if remainder:
            raise ValueError("GLM-5.2 middle layers do not divide across PP stages")
        layer_counts = (first, *(middle_layers for _ in range(middle_stages)), last)
    else:
        if remaining:
            raise ValueError("GLM-5.2 first/last PP stages do not cover all layers")
        layer_counts = (first, last)
    ends = []
    for count in layer_counts:
        ends.append(count + (ends[-1] if ends else 0))
    return tuple(ends)


_PIPELINE_LAYER_ENDS = _pipeline_layer_ends()
DELTA_SPEC = SyntheticDeltaSpec(
    checkpoint_format="nvfp4",
    quantized_value_density=0.00375,
    high_precision_value_density=0.01,
    output_shards=MILES.pipeline_model_parallel_size,
    output_shard_layout=(
        f"miles-pp{MILES.pipeline_model_parallel_size}-layer-placement-v1"
    ),
    # MTP is present in the immutable serving checkpoint but is not trained.
    immutable_prefixes=("model.layers.78.",),
)
DELTA_ID = f"glm5-2/{SOURCE_REVISION}/{synthetic_delta_profile_id(DELTA_SPEC)}"
DELTA_SOURCE_DIR = f"{DELTA_MOUNT}/{DELTA_ID}"
LOCAL_TARGET_CHECKPOINT_DIR = "/local-checkpoint/glm5-2-nvfp4/target"
LOCAL_CANONICAL_CHECKPOINT_DIR = "/local-checkpoint/glm5-2-nvfp4/canonical"
SGLANG_CACHE_PATH = "/root/.cache/sglang"

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
checkpoint_volume = modal.Volume.from_name(
    "miles-checkpoints", create_if_missing=True, version=2
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
prep_image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_pip_packages=TRAINER_EXTRA_PIP_PACKAGES,
    image_run_commands=TRAINER_IMAGE_RUN_COMMANDS,
).add_local_dir(
    str(Path(__file__).resolve().parents[2]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)
serving_image = build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_env=SGLANG_SERVER_ENV,
    runtime=DEFAULT_SGLANG_RUNTIME,
).add_local_dir(
    str(Path(__file__).resolve().parents[2]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)


def _pipeline_shard_for_tensor(name: str) -> int:
    if name == "model.embed_tokens.weight":
        return 0
    if name in {"lm_head.weight", "model.norm.weight"}:
        return len(_PIPELINE_LAYER_ENDS) - 1
    prefix = "model.layers."
    if not name.startswith(prefix):
        raise ValueError(f"no pipeline placement for tensor {name!r}")
    layer = int(name[len(prefix) :].partition(".")[0])
    for stage, layer_end in enumerate(_PIPELINE_LAYER_ENDS):
        if layer < layer_end:
            return stage
    raise ValueError(f"layer {layer} is outside the target-model pipeline")


@app.function(
    image=prep_image,
    cpu=4,
    memory=4096,
    max_containers=DOWNLOAD_MAX_CONTAINERS,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def _download_source_file(repo_file: CachedRepoFile) -> str:
    prep.apply_prep_environment(MODEL)
    return download_cached_safetensors_file(repo_file, commit=hf_cache_volume.commit)


@app.function(
    image=prep_image,
    gpu=f"{MODAL.gpu}:1",
    memory=MODAL.trainer_memory_mib,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(CHECKPOINTS_PATH): checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def prepare_base() -> None:
    source_snapshot = download_cached_snapshot(
        _download_source_file,
        SOURCE_MODEL,
        SOURCE_REVISION,
        volume=hf_cache_volume,
    )
    prep.prepare_checkpoints(
        MODEL,
        checkpoint_volume,
        source_snapshot=source_snapshot,
        rollout_snapshot=None,
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
    lineage = prepare_standard_delta(
        str(ROLLOUT_CHECKPOINT_PATH),
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


@app.function(
    image=serving_image,
    gpu=f"{MODAL.gpu}:{ROLLOUT_GPUS_PER_ENGINE}",
    cpu=64,
    memory=MODAL.rollout_memory_mib,
    ephemeral_disk=2 * 1024 * 1024,
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
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
    spec = WeightUpdateSpec(
        model_name="GLM-5.2 mixed NVFP4/BF16",
        base_checkpoint_dir=str(ROLLOUT_CHECKPOINT_PATH),
        local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
        local_canonical_checkpoint_dir=LOCAL_CANONICAL_CHECKPOINT_DIR,
        server_args=SGLANG_SERVER_ARGS,
        tp_size=ROLLOUT_GPUS_PER_ENGINE,
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


@app.local_entrypoint()
def main(
    update_mode: str = "disk",
    canonical_storage: str | None = None,
    sample_id: str = "1",
    skip_preparation: bool = False,
    post_mutation_failure_only: bool = False,
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
        post_mutation_failure_only,
    )
