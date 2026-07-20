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

import modal

from cookbook.common.constants import DATA_PATH, HF_CACHE_PATH, MINUTES
from cookbook.slime_disagg import trainer_image

EXPERIMENT = os.environ["EXPERIMENT_CONFIG"]  # required; a default would silently prep the wrong experiment
SLIME_LOCAL_DIR = os.environ.get("SLIME_LOCAL_DIR")  # optional dev overlay of a local slime checkout

exp = importlib.import_module(f"cookbook.slime_disagg.configs.{EXPERIMENT}")
slime_cfg = exp.slime

image = trainer_image.build_trainer_image(hf_cache_path=str(HF_CACHE_PATH), experiment=EXPERIMENT, slime_local=SLIME_LOCAL_DIR)

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True, version=2)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True, version=2)

app = modal.App(f"{exp.APP_NAME}-prep")


@app.function(
    image=image, volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=2 * 60 * MINUTES, secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
def download_model() -> None:
    """Snapshot the served model into the HF cache (sglang serves it by repo id)."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=slime_cfg.hf_checkpoint)
    hf_cache_volume.commit()


@app.function(
    image=image, volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * MINUTES, secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
def prepare_dataset() -> None:
    data_volume.reload()
    slime_cfg.prepare_data()
    data_volume.commit()
