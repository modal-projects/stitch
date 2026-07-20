"""Preparation stage for the miles cookbook — a separate Modal app from the rollout app.

Prep builds the served base, the torch_dist ref_load, and the dataset: a one-shot stage that
runs once before serving. It lives in its own app so invoking it never instantiates the
``Server`` in ``app.py`` — and therefore never brings up the rollout autoscaler floor
(``rollout_min_containers``), a serving concern with no place in preparation. Run prep first,
then deploy the rollout app:

    EXPERIMENT_CONFIG=<cfg> uv run --extra modal modal run -d -m cookbook.miles_disagg.prep_app::prepare_checkpoints
    EXPERIMENT_CONFIG=<cfg> uv run --extra modal modal run -d -m cookbook.miles_disagg.prep_app::prepare_torch_dist
    EXPERIMENT_CONFIG=<cfg> uv run --extra modal modal run -d -m cookbook.miles_disagg.prep_app::prepare_dataset
"""

from __future__ import annotations

import importlib
import os

import modal
import modal.experimental

from cookbook.common import ray_cluster
from cookbook.common.constants import DATA_PATH, HF_CACHE_PATH, MINUTES, PREP_PATH
from cookbook.miles_disagg import prep, trainer_image

EXPERIMENT = os.environ["EXPERIMENT_CONFIG"]  # required; a default would silently prep the wrong experiment
MILES_LOCAL_DIR = os.environ.get("MILES_LOCAL_DIR")  # optional dev overlay of a local miles checkout

exp = importlib.import_module(f"cookbook.miles_disagg.configs.{EXPERIMENT}")
modal_cfg = exp.modal
miles_cfg = exp.miles

image = trainer_image.build_trainer_image(hf_cache_path=str(HF_CACHE_PATH), experiment=EXPERIMENT, miles_local=MILES_LOCAL_DIR)

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True, version=2)
data_volume = modal.Volume.from_name("miles-data", create_if_missing=True, version=2)
prep_volume = modal.Volume.from_name("miles-prep-checkpoints", create_if_missing=True, version=2)

app = modal.App(f"{exp.APP_NAME}-prep")


@app.function(
    image=image, gpu=f"{modal_cfg.gpu}:1",
    volumes={str(HF_CACHE_PATH): hf_cache_volume, str(PREP_PATH): prep_volume},
    timeout=6 * 60 * MINUTES, secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
def prepare_checkpoints() -> None:
    prep.prepare_checkpoints(exp, prep_volume)


# torch_dist conversion is clustered across nodes (a large MoE won't fit an 8-way split).
_TORCH_DIST_MULTINODE = modal_cfg.torch_dist_prep_nodes > 1


@app.function(
    image=image, gpu=f"{modal_cfg.gpu}:{modal_cfg.torch_dist_prep_gpus_per_node}",
    volumes={str(HF_CACHE_PATH): hf_cache_volume, str(PREP_PATH): prep_volume},
    timeout=6 * 60 * MINUTES,
    ephemeral_disk=(modal_cfg.torch_dist_prep_ephemeral_disk_mib or modal_cfg.rollout_ephemeral_disk_mib),
    secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
    **({"experimental_options": {"efa_enabled": True}} if _TORCH_DIST_MULTINODE else {}),
)
@(modal.experimental.clustered(modal_cfg.torch_dist_prep_nodes, rdma=True) if _TORCH_DIST_MULTINODE else lambda fn: fn)
def prepare_torch_dist() -> None:
    rank, master_addr, _ = ray_cluster.get_modal_cluster_context(modal_cfg.torch_dist_prep_nodes)
    prep.prepare_torch_dist(exp, prep_volume, rank=rank, master_addr=master_addr)


@app.function(
    image=image, volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * MINUTES, secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
def prepare_dataset() -> None:
    data_volume.reload()
    miles_cfg.prepare_data()
    data_volume.commit()
