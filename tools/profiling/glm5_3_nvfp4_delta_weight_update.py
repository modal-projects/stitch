"""Profile one checksummed GLM-5.3 nvfp4 update using the maintained recipe.

Prepare the cookbook checkpoint first. This measures one replica and one synthetic
update; it excludes trainer export, pool publication, and fleet convergence.
"""

from __future__ import annotations

from pathlib import Path

import modal

from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    DRAFT_PATH,
    HF_CACHE_PATH,
    SGLANG_CACHE_PATH,
)
from cookbook.common.serving_image import DEFAULT_SGLANG_RUNTIME, build_serving_image
from cookbook.miles_disagg.configs import glm5_3_nvfp4 as model
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

EXPERIMENT = "glm5_3_nvfp4"
BASE_CHECKPOINT = model.ROLLOUT_CHECKPOINT_PATH
DELTA_MOUNT = "/synthetic-delta"
DELTA_SPEC = SyntheticDeltaSpec(
    checkpoint_format="nvfp4",
    quantized_value_density=0.00375,
    high_precision_value_density=0.01,
    output_shards=4,
    # GLM-5.3 has 78 decoder layers; its bundled MTP head remains fixed.
    immutable_prefixes=("model.layers.78.",),
)
DELTA_SOURCE_DIR = (
    f"{DELTA_MOUNT}/{EXPERIMENT}/{model.SOURCE_REVISION}/"
    f"{synthetic_delta_profile_id(DELTA_SPEC)}"
)
LOCAL_ROOT = f"/local-checkpoint/{EXPERIMENT}"

app = modal.App("profile-glm5-3-nvfp4-delta-weight-update")
checkpoint_volume = modal.Volume.from_name("miles-checkpoints", version=2)
delta_volume = modal.Volume.from_name(
    "stitch-synthetic-deltas", create_if_missing=True, version=2
)
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
)
draft_volumes = (
    {
        str(DRAFT_PATH): modal.Volume.from_name(
            model.modal.draft_volume,
            environment_name=model.modal.draft_volume_env,
        ).read_only()
    }
    if model.modal.draft_volume
    else {}
)
image = build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_env=model.SGLANG_SERVER_ENV,
    runtime=getattr(model, "SGLANG_RUNTIME", DEFAULT_SGLANG_RUNTIME),
).add_local_dir(
    str(Path(__file__).resolve().parents[1]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)


@app.function(
    image=image,
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
        str(BASE_CHECKPOINT),
        DELTA_SOURCE_DIR,
        spec=DELTA_SPEC,
        commit=delta_volume.commit,
    )


@app.function(
    image=image,
    gpu=model.modal.rollout_gpus(model.ROLLOUT_GPUS_PER_ENGINE),
    cloud=model.modal.cloud,
    region=model.modal.region,
    cpu=64,
    memory=model.modal.rollout_memory_mib,
    # Disk-mode comparisons reconstruct the complete checkpoint on local storage.
    ephemeral_disk=2 * 1024 * 1024,
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
        DELTA_MOUNT: delta_volume.read_only(),
        SGLANG_CACHE_PATH: sglang_cache_volume,
        **draft_volumes,
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
            model_name="GLM-5.3 nvfp4",
            base_checkpoint_dir=str(BASE_CHECKPOINT),
            local_target_checkpoint_dir=f"{LOCAL_ROOT}/target",
            local_canonical_checkpoint_dir=f"{LOCAL_ROOT}/canonical",
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
    update_mode: str = "cpu",
    canonical_storage: str | None = None,
    sample_id: str = "1",
    skip_preparation: bool = False,
) -> None:
    if update_mode == "cpu" and canonical_storage is None:
        canonical_storage = "memory"
    mode, storage = parse_update_destination(update_mode, canonical_storage)
    if not skip_preparation:
        prepare_delta.remote()
    benchmark.remote(mode, storage, modal_runtime_label(), sample_id)
