"""Profile one GLM-5.2 NVFP4 delta weight update on Modal.

CPU destination:

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/glm5_2_nvfp4_delta_weight_update.py \
      --update-mode cpu --canonical-storage memory

Disk destination:

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/glm5_2_nvfp4_delta_weight_update.py \
      --update-mode disk
"""

from __future__ import annotations

from pathlib import Path

import modal

from cookbook.common.constants import CHECKPOINTS_PATH, HF_CACHE_PATH
from cookbook.common.serving_image import build_serving_image
from cookbook.miles_disagg import prep, trainer_image
from cookbook.miles_disagg.configs import glm5_2_nvfp4 as model
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

APP_NAME = "profile-glm5-2-nvfp4-delta-weight-update"
EXPERIMENT = "glm5_2_nvfp4"
DELTA_MOUNT = "/synthetic-delta"
_TARGET_MODEL_LAYERS = 78


def _pipeline_layer_ends() -> tuple[int, ...]:
    stages = model.miles.pipeline_model_parallel_size
    first = model.miles.decoder_first_pipeline_num_layers
    last = model.miles.decoder_last_pipeline_num_layers
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
    output_shards=model.miles.pipeline_model_parallel_size,
    output_shard_layout=(
        f"miles-pp{model.miles.pipeline_model_parallel_size}-layer-placement-v1"
    ),
    # MTP is present in the immutable serving checkpoint but is not trained.
    immutable_prefixes=("model.layers.78.",),
)
DELTA_ID = f"glm5-2/{model.SOURCE_REVISION}/{synthetic_delta_profile_id(DELTA_SPEC)}"
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
    miles_repo_ref=model.MILES_REPO_REF,
    extra_pip_packages=model.TRAINER_EXTRA_PIP_PACKAGES,
    image_run_commands=model.TRAINER_IMAGE_RUN_COMMANDS,
).add_local_dir(
    str(Path(__file__).resolve().parents[1]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)
serving_image = build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_env=model.SGLANG_SERVER_ENV,
).add_local_dir(
    str(Path(__file__).resolve().parents[1]),
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
    gpu=f"{model.modal.gpu}:1",
    memory=model.modal.trainer_memory_mib,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(CHECKPOINTS_PATH): checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def prepare_base() -> None:
    prep.prepare_checkpoints(model, checkpoint_volume)


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
        str(model.ROLLOUT_CHECKPOINT_PATH),
        DELTA_SOURCE_DIR,
        spec=DELTA_SPEC,
        commit=delta_volume.commit,
        output_shard_for_tensor=_pipeline_shard_for_tensor,
    )


@app.function(
    image=serving_image,
    gpu=f"{model.modal.gpu}:{model.ROLLOUT_GPUS_PER_ENGINE}",
    cpu=64,
    memory=model.modal.rollout_memory_mib,
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
) -> dict:
    return run_delta_weight_update(
        WeightUpdateSpec(
            model_name="GLM-5.2 mixed NVFP4/BF16",
            base_checkpoint_dir=str(model.ROLLOUT_CHECKPOINT_PATH),
            local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
            local_canonical_checkpoint_dir=LOCAL_CANONICAL_CHECKPOINT_DIR,
            server_args=model.SGLANG_SERVER_ARGS,
            tp_size=model.ROLLOUT_GPUS_PER_ENGINE,
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
