"""The miles trainer image + the versions pinned to launch a miles run.

The base image bakes Megatron-LM (native --fp4-format NVFP4) + TransformerEngine; the
miles fork is cloned over it at a pinned commit. The serving half is separate and shared
(common/serving_image.py) — the pool installs no trainer package.

The fork's commit stack (what we carry over upstream miles) and how to re-rebase live in
MILES_FORK.md next to this file.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Dated tag, never `latest`: Modal caches from_registry per tag string and won't re-pull
# a moved mutable tag, so `latest` silently serves whatever was first pulled.
MILES_IMAGE_TAG = "radixark/miles:dev-202607182122"  # base Megatron/TE; match the upstream main stitch-miles is on
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "15cf7ed0344850affa354b8b81ad3acbda11474b"  # branch stitch-miles; see MILES_FORK.md

MILES_ROOT = "/root/miles"
MEGATRON_PATH = "/root/Megatron-LM"  # source-only megatron.training must be on PYTHONPATH
TORCH_DIST_CONVERT_WRAPPER = "/root/convert_hf_to_torch_dist_modal.py"

_COOKBOOK_DIR = Path(__file__).resolve().parent.parent  # .../cookbook
_TORCH_DIST_WRAPPER_SRC = Path(__file__).resolve().parent / "convert_hf_to_torch_dist_modal.py"


def build_trainer_image(*, hf_cache_path: str, experiment: str, run_id: str | None = None, miles_local: str | None = None) -> modal.Image:
    """The miles trainer image: RDMA/EFA userspace + the pinned miles fork + the
    trainer-side delta encoder's codecs. stitch + the cookbook package are mounted so the
    trainer, Ray actors, and the sidecar subprocess resolve their imports."""
    image = (
        modal.Image.from_registry(MILES_IMAGE_TAG)
        .entrypoint([])
        # RDMA/EFA userspace so multi-node NCCL binds EFA under rdma=True instead of TCP.
        .apt_install("libibverbs-dev", "libibverbs1", "libhwloc-dev", "libnl-route-3-200")
        .run_commands(f"rm -rf {hf_cache_path}")  # baked HF cache must not shadow the mounted volume
        .run_commands(
            f"rm -rf {MILES_ROOT}"
            f" && git clone {MILES_REPO_URL} {MILES_ROOT}"
            f" && cd {MILES_ROOT} && git fetch origin {MILES_REPO_REF} && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {MILES_ROOT}"
        )
        # The trainer-side delta ENCODER (miles delta.py) needs the codecs even under --no-deps.
        .pip_install("fastapi", "httpx", "uvicorn", "zstandard", "xxhash", "blake3")
        .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "EXPERIMENT_CONFIG": experiment,
              **({"RUN_ID": run_id} if run_id else {})})
        .add_local_file(str(_TORCH_DIST_WRAPPER_SRC), TORCH_DIST_CONVERT_WRAPPER, copy=True)
        .add_local_python_source("stitch")
        .add_local_dir(str(_COOKBOOK_DIR), remote_path="/root/cookbook", ignore=["**/__pycache__"])
    )
    if miles_local:  # dev overlay: replace the cloned fork with a local checkout (no rebuild)
        image = image.add_local_dir(miles_local, remote_path=MILES_ROOT, ignore=[".git", "**/__pycache__", "**/*.pyc"])
    return image
