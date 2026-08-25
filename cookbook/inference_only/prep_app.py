"""Preparation stage for the inference-only cookbook — a separate Modal app.

Prep snapshots the served model into the checkpoint Volume: a one-shot stage that
runs once before serving. It lives in its own app so invoking it never instantiates
the ``Server`` in ``app.py`` — and therefore never brings up the rollout
autoscaler floor. Run prep first, then launch:

    EXPERIMENT_CONFIG=<cfg> uv run --extra modal modal run -d \\
      -m cookbook.inference_only.prep_app::download_model
"""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path

import modal

from cookbook.common.constants import CHECKPOINTS_PATH, HF_CACHE_PATH, MINUTES
from cookbook.inference_only.checkpoint import require_checkpoint

EXPERIMENT = os.environ[
    "EXPERIMENT_CONFIG"
]  # required; a default would silently prep the wrong experiment

exp = importlib.import_module(f"cookbook.inference_only.configs.{EXPERIMENT}")

hf_cache_volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=exp.modal.hf_cache_volume_version
)
checkpoint_volume = modal.Volume.from_name(
    exp.CHECKPOINT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
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

app = modal.App(f"{exp.APP_NAME}-prep")


@app.function(
    image=image,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(CHECKPOINTS_PATH): checkpoint_volume,
    },
    timeout=6 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def download_model() -> None:
    """Materialize the configured model artifact in the checkpoint Volume."""
    from huggingface_hub import snapshot_download

    checkpoint_volume.reload()
    target = Path(exp.ROLLOUT_CHECKPOINT_PATH)
    try:
        require_checkpoint(target)
        print(f"reusing model checkpoint {target}")
        return
    except FileNotFoundError:
        pass
    except RuntimeError:
        # Incomplete target from a prior failed prep; clean it for retry
        shutil.rmtree(target, ignore_errors=True)
    partial = target.with_name(f"{target.name}.partial")
    snapshot_download(
        repo_id=exp.SOURCE_MODEL,
        revision=exp.SOURCE_REVISION,
        local_dir=partial,
    )
    shutil.rmtree(partial / ".cache", ignore_errors=True)
    partial.rename(target)
    checkpoint_volume.commit()
