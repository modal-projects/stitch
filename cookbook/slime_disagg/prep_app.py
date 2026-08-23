"""Preparation stage for the slime cookbook — a separate Modal app from the rollout app.

Prep snapshots the served model into the HF cache and builds the dataset: a one-shot stage that
runs once before serving. It lives in its own app so invoking it never instantiates the ``Server``
in ``app.py`` — and therefore never brings up the rollout autoscaler floor
(``rollout_min_containers``), a serving concern with no place in preparation. Run prep first, then
deploy the rollout app:

    EXPERIMENT_CONFIG=<cfg> uv run --extra modal modal run -d -m cookbook.slime_disagg.prep_app::download_model
    EXPERIMENT_CONFIG=<cfg> uv run --extra modal modal run -d -m cookbook.slime_disagg.prep_app::prepare_dataset
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import modal

from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    MINUTES,
)
from cookbook.common.hf_download import (
    DOWNLOAD_MAX_CONTAINERS,
    LocalRepoFile,
    download_local_safetensors_file,
    download_local_snapshot,
)
from cookbook.slime_disagg import trainer_image

EXPERIMENT = os.environ[
    "EXPERIMENT_CONFIG"
]  # required; a default would silently prep the wrong experiment
SLIME_LOCAL_DIR = os.environ.get(
    "SLIME_LOCAL_DIR"
)  # optional dev overlay of a local slime checkout

exp = importlib.import_module(f"cookbook.slime_disagg.configs.{EXPERIMENT}")
slime_cfg = exp.slime

image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH), experiment=EXPERIMENT, slime_local=SLIME_LOCAL_DIR
)

data_volume = modal.Volume.from_name("slime-data", create_if_missing=True, version=2)
checkpoint_volume = modal.Volume.from_name(
    "slime-checkpoints",
    create_if_missing=True,
    version=2,
)

app = modal.App(f"{exp.APP_NAME}-prep")


@app.function(
    image=image,
    cpu=4,
    memory=4096,
    max_containers=DOWNLOAD_MAX_CONTAINERS,
    volumes={str(CHECKPOINTS_PATH): checkpoint_volume},
    timeout=2 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def _download_model_file(repo_file: LocalRepoFile) -> str:
    return download_local_safetensors_file(repo_file, commit=checkpoint_volume.commit)


@app.function(
    image=image,
    volumes={str(CHECKPOINTS_PATH): checkpoint_volume},
    timeout=2 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def download_model() -> str:
    """Materialize the configured model artifact with one input per repo file."""
    return download_local_snapshot(
        _download_model_file,
        exp.SOURCE_MODEL,
        getattr(exp, "SOURCE_REVISION", None),
        Path(slime_cfg.hf_checkpoint),
        volume=checkpoint_volume,
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
    slime_cfg.prepare_data()
    data_volume.commit()
