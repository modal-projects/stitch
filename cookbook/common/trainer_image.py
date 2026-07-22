"""Shared trainer-image layers. The framework-specific base image, fork clone, and any extra
build steps stay in each recipe's ``trainer_image.py``; the layers every framework needs — the
delta-encoder codecs, the HF-download env, and the stitch + cookbook source mounts — live here.
"""

from __future__ import annotations

from pathlib import Path

import modal

_COOKBOOK_DIR = Path(__file__).resolve().parent.parent  # .../cookbook


def add_common_layers(image: modal.Image, *, experiment: str, run_id: str | None = None) -> modal.Image:
    """Append the framework-agnostic trainer layers: the trainer-side delta ENCODER's codecs
    (needed even under --no-deps), the HF-download env (``EXPERIMENT_CONFIG``/``RUN_ID`` so a
    container's re-import selects the same experiment/run as the deploy), and the stitch + cookbook
    source mounts so the trainer, Ray actors, and the sidecar subprocess resolve their imports."""
    return (
        image
        .pip_install("fastapi", "httpx", "uvicorn", "zstandard", "xxhash", "blake3")
        .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1",
              "EXPERIMENT_CONFIG": experiment, **({"RUN_ID": run_id} if run_id else {})})
        .add_local_python_source("stitch")
        .add_local_dir(str(_COOKBOOK_DIR), remote_path="/root/cookbook", ignore=["**/__pycache__"])
    )
