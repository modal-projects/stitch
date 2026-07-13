"""The slime trainer image + the versions pinned to launch a slime run.

The serving half is separate and shared (common/serving_image.py) — the pool installs no
trainer package, so slime and miles serve on the identical weight-sync sglang image.
"""

from __future__ import annotations

from pathlib import Path

import modal

SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"
SLIME_REPO_URL = "https://github.com/modal-projects/slime.git"
# Pin to an exact commit, not the branch tip (the fetch+checkout is a cached layer).
SLIME_REPO_REF = "11bb0fa48aa37d5c54fe297143c6bc1d40f311bf"
SLIME_ROOT = "/root/slime"

_COOKBOOK_DIR = Path(__file__).resolve().parent.parent  # .../cookbook


def build_trainer_image(*, hf_cache_path: str, slime_local: str | None = None) -> modal.Image:
    """The slime trainer image: the pinned slime fork over the slime base, Megatron-LM
    reinstalled so ``megatron.training`` is importable, + the delta encoder's codecs.
    stitch + the cookbook package are mounted for the trainer, Ray actors, and sidecar."""
    image = (
        modal.Image.from_registry(SLIME_IMAGE_TAG)
        .entrypoint([])
        .run_commands(f"rm -rf {hf_cache_path}")  # baked HF cache must not shadow the mounted volume
        .run_commands(
            f"rm -rf {SLIME_ROOT}"
            f" && git clone --depth 1 {SLIME_REPO_URL} {SLIME_ROOT}"
            f" && cd {SLIME_ROOT} && git fetch --depth 1 origin {SLIME_REPO_REF} && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {SLIME_ROOT}"
        )
        # The base installs megatron-core as a strict editable that hides
        # megatron.training; reinstall in compat mode so the whole source tree is importable.
        .run_commands("cd /root/Megatron-LM && python3 -m pip install --no-deps -e . --config-settings editable_mode=compat")
        # The trainer-side delta ENCODER (slime.utils.disk_delta) needs the codecs even under --no-deps.
        .pip_install("fastapi", "httpx", "uvicorn", "zstandard", "xxhash", "blake3")
        .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
        .add_local_python_source("stitch")
        .add_local_dir(str(_COOKBOOK_DIR), remote_path="/root/cookbook", ignore=["**/__pycache__"])
    )
    if slime_local:  # dev overlay: replace the cloned fork with a local checkout (no rebuild)
        image = image.add_local_dir(slime_local, remote_path=SLIME_ROOT, ignore=[".git", "**/__pycache__", "**/*.pyc"])
    return image
