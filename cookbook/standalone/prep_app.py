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

import modal

from cookbook.common.constants import CHECKPOINTS_PATH, MINUTES

EXPERIMENT = os.environ[
    "EXPERIMENT_CONFIG"
]  # required; a default would silently prep the wrong experiment
exp = importlib.import_module(f"cookbook.standalone.configs.{EXPERIMENT}")

app = modal.App(f"{exp.APP_NAME}-prep")
checkpoint_volume = modal.Volume.from_name(
    "miles-checkpoints", create_if_missing=True, version=2
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)


@app.function(
    image=image,
    cpu=32,
    memory=(16 * 1024, 256 * 1024),
    volumes={str(CHECKPOINTS_PATH): checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * MINUTES,
)
def download_base() -> str:
    """Download the pinned served base as plain files (sglang and the sidecar's
    canonical checkpoint space both read this directory)."""
    from huggingface_hub import snapshot_download

    target = exp.BASE_CHECKPOINT_PATH
    marker = target / ".complete"
    if marker.exists():
        print(f"Base checkpoint already prepared: {target}")
        return str(target)
    snapshot_download(
        exp.SOURCE_MODEL,
        revision=exp.SOURCE_REVISION,
        local_dir=str(target),
    )
    marker.touch()
    checkpoint_volume.commit()
    return str(target)
