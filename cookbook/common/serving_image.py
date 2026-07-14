"""The weight-sync sglang SERVING image — shared by every recipe.

Trainer-agnostic: no trainer package is installed (the delta apply lives in the engine
behind ``/pull_weights``), so miles and slime serve on the identical image; precision
comes from the served checkpoint, not a ``--quantization`` flag. The base image supplies
kernels/CUDA; the fork pin (branch ``stitch-sglang-v0.5.15`` over base tag ``v0.5.15``)
carries ``/pull_weights``, the hardened local_checkpoint receiver, the quantized-reload
restore protocol (reload == init), and the O(delta) partial-reload load plan. See
``SGLANG_FORK.md`` next to this file for the full patch stack, the upstreaming PRs, and
how to re-port these patches onto a newer sglang release.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Fork = base sglang tag + the stitch weight-sync patch stack (see SGLANG_FORK.md).
# The base tag MUST match the branch's base tag: the fork overlays python/ only, so the
# baked kernels/CUDA come from this image and must be ABI-compatible with that python/.
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.15"
SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "stitch-sglang-v0.5.15"
SGLANG_FORK_COMMIT = "27ca14ef94dbfa462d4e4ab48efaee200786ee29"

_COOKBOOK_DIR = Path(__file__).resolve().parent.parent  # .../cookbook

_SERVING_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
    "SGLANG_ENABLE_RELOAD_LOAD_PLAN": "1",  # opt into load-plan reload replay (the old GLM fused-MoE record hang no longer reproduces on the pinned fork; under e2e evaluation)
}


def build_serving_image(*, hf_cache_path: str, delta_volume_name: str) -> modal.Image:
    """The rollout-pool image. ``DELTA_VOLUME_NAME`` is read by the engine's pre-read
    hook and the sidecar's Store; stitch + the whole cookbook package are mounted so the
    sidecar (``python -m cookbook.common.sidecar``) and the framework hooks resolve."""
    return (
        modal.Image.from_registry(SGLANG_IMAGE_TAG)
        .run_commands(
            f"cd /sgl-workspace/sglang && git remote add modal-fork {SGLANG_FORK_REPO}"
            f" && git fetch modal-fork {SGLANG_FORK_BRANCH} && git checkout {SGLANG_FORK_COMMIT} -- python/"
        )
        .run_commands(f"rm -rf {hf_cache_path}")  # baked HF cache must not shadow the mounted volume
        .pip_install(
            "autoinference-utils==0.2.0",  # sglang server lifecycle
            "fastapi", "httpx", "uvicorn",  # the stitch sidecar
            "zstandard", "xxhash", "blake3",  # engine-side /pull_weights receiver's codecs
        )
        .env({**_SERVING_ENV, "DELTA_VOLUME_NAME": delta_volume_name})
        # The kernel-cache volume mounts at /root/.cache/sglang, which can't mount over a
        # non-empty path — clear it as the final filesystem step (repopulated on boot).
        .run_commands("rm -rf /root/.cache/sglang")
        .add_local_python_source("stitch")
        .add_local_dir(str(_COOKBOOK_DIR), remote_path="/root/cookbook", ignore=["**/__pycache__"])
    )
