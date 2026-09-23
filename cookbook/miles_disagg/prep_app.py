"""Prepare pinned model and dataset artifacts without starting a rollout fleet."""

from __future__ import annotations

import importlib
import os

import modal
import modal.experimental

from cookbook.common import ray_cluster
from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    MINUTES,
)
from cookbook.common.hf_download import (
    DOWNLOAD_MAX_CONTAINERS,
    CachedRepoFile,
    download_cached_safetensors_file,
    download_cached_snapshot,
    local_cached_snapshot,
)
from cookbook.miles_disagg import prep, trainer_image
from cookbook.miles_disagg.config import validate_recipe

EXPERIMENT = os.environ[
    "EXPERIMENT_CONFIG"
]  # required; a default would silently prep the wrong experiment
exp = importlib.import_module(f"cookbook.miles_disagg.configs.{EXPERIMENT}")
validate_recipe(exp)
modal_cfg = exp.modal
miles_cfg = exp.miles

image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    miles_repo_ref=prep.pinned_miles_revision(exp),
)

hf_cache_volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
data_volume = modal.Volume.from_name("miles-data", create_if_missing=True, version=2)
checkpoint_volume = modal.Volume.from_name(
    "miles-checkpoints",
    create_if_missing=True,
    version=2,
)

app = modal.App(f"{exp.APP_NAME}-prep")
checkpoint_gpu = (
    f"{modal_cfg.gpu}:1" if getattr(exp, "CHECKPOINT_PREP_REQUIRES_GPU", True) else None
)


@app.function(
    image=image,
    cpu=4,
    memory=4096,
    max_containers=DOWNLOAD_MAX_CONTAINERS,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=6 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def _download_source_file(repo_file: CachedRepoFile) -> str:
    prep.apply_prep_environment(exp)
    return download_cached_safetensors_file(repo_file, commit=hf_cache_volume.commit)


@app.function(
    image=image,
    gpu=checkpoint_gpu,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(CHECKPOINTS_PATH): checkpoint_volume,
    },
    memory=modal_cfg.trainer_memory_mib,
    timeout=6 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def prepare_checkpoints() -> None:
    source_snapshot = download_cached_snapshot(
        _download_source_file,
        exp.SOURCE_MODEL,
        getattr(exp, "SOURCE_REVISION", None),
        volume=hf_cache_volume,
    )
    prep.prepare_checkpoints(
        exp,
        checkpoint_volume,
        source_snapshot=source_snapshot,
    )


# torch_dist conversion is clustered across nodes (a large MoE won't fit an 8-way split).
_TORCH_DIST_MULTINODE = modal_cfg.torch_dist_prep_nodes > 1


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{modal_cfg.torch_dist_prep_gpus_per_node}",
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(CHECKPOINTS_PATH): checkpoint_volume,
    },
    memory=modal_cfg.trainer_memory_mib,
    timeout=6 * 60 * MINUTES,
    ephemeral_disk=(
        modal_cfg.torch_dist_prep_ephemeral_disk_mib
        or modal_cfg.rollout_ephemeral_disk_mib
    ),
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
    **(
        {"experimental_options": {"efa_enabled": True}} if _TORCH_DIST_MULTINODE else {}
    ),
)
@(
    modal.experimental.clustered(modal_cfg.torch_dist_prep_nodes, rdma=True)
    if _TORCH_DIST_MULTINODE
    else lambda fn: fn
)
def prepare_torch_dist() -> None:
    hf_cache_volume.reload()
    rank, master_addr, _ = ray_cluster.get_modal_cluster_context(
        modal_cfg.torch_dist_prep_nodes
    )
    prep.prepare_torch_dist(
        exp,
        checkpoint_volume,
        rank=rank,
        master_addr=master_addr,
        source_snapshot=local_cached_snapshot(
            exp.SOURCE_MODEL,
            getattr(exp, "SOURCE_REVISION", None),
        ),
    )


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def prepare_dataset() -> None:
    data_volume.reload()
    miles_cfg.prepare_data()
    data_volume.commit()
