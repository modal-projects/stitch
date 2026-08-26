"""Preparation stage for a standalone pool — a separate Modal app from the rollout app.

Prep materializes the served base checkpoint: a one-shot stage that runs once before
serving. It lives in its own app so invoking it never instantiates the ``Server`` in
``app.py`` — and therefore never brings up the rollout autoscaler floor. A quantized
release checkpoint is served as published, so download is the whole preparation:

    EXPERIMENT_CONFIG=glm5_2_fp8 uv run --extra modal modal run -d -m cookbook.standalone.prep_app::download_base
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import modal

from cookbook.common.constants import CHECKPOINTS_PATH, MINUTES
from cookbook.common.hf_download import (
    DOWNLOAD_MAX_CONTAINERS,
    LocalRepoFile,
    download_local_safetensors_file,
    download_local_snapshot,
)

EXPERIMENT = os.environ[
    "EXPERIMENT_CONFIG"
]  # required; a default would silently prep the wrong experiment
exp = importlib.import_module(f"cookbook.standalone.configs.{EXPERIMENT}")

app = modal.App(f"{exp.APP_NAME}-prep")
checkpoint_volume = modal.Volume.from_name(
    exp.CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "EXPERIMENT_CONFIG": EXPERIMENT,
            "PYTHONPATH": "/root",
        }
    )
    .add_local_dir(
        str(Path(__file__).resolve().parent.parent),
        remote_path="/root/cookbook",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)


@app.function(
    image=image,
    cpu=4,
    memory=4096,
    max_containers=DOWNLOAD_MAX_CONTAINERS,
    volumes={str(CHECKPOINTS_PATH): checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * MINUTES,
    include_source=False,
)
def _download_base_file(repo_file: LocalRepoFile) -> str:
    return download_local_safetensors_file(repo_file, commit=checkpoint_volume.commit)


@app.function(
    image=image,
    volumes={str(CHECKPOINTS_PATH): checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * MINUTES,
    include_source=False,
)
def download_base() -> str:
    """Materialize the pinned served base with one call per safetensors shard."""

    return download_local_snapshot(
        _download_base_file,
        exp.SOURCE_MODEL,
        exp.SOURCE_REVISION,
        exp.BASE_CHECKPOINT_PATH,
        volume=checkpoint_volume,
    )
