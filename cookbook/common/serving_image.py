"""The weight-sync sglang SERVING image — shared by every recipe.

Trainer-agnostic: no trainer package is installed (the delta apply lives in the engine behind
``/stage_weight_update``), so miles and slime serve on the identical image; precision comes
from the served checkpoint, not a ``--quantization`` flag. The fork pin carries asynchronous
weight staging, correct quantized weight loading, and the optional CPU delta cache. See
``SGLANG_FORK.md`` for the patch stack and how to re-port onto a newer sglang release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import modal

# The base tag MUST match the branch's base tag: the fork overlays python/ only, so the
# baked kernels/CUDA must be ABI-compatible with it.
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.16"
SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "stitch-sglang-v0.5.16"
SGLANG_FORK_COMMIT = "e8d7dee6106fa79bb064f5e1822608ef39898e02"

_COOKBOOK_DIR = Path(__file__).resolve().parent.parent

_SERVING_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
}


def build_serving_image(
    *,
    hf_cache_path: str,
    delta_volume_name: str | None = None,
    experiment: str,
    run_id: str | None = None,
    extra_packages: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
) -> modal.Image:
    """The rollout-pool image. Volume-backed recipes set ``delta_volume_name`` for their
    Store; other Store backends omit it. Backend-specific packages and environment
    belong in ``extra_packages`` / ``extra_env`` so all build steps still precede the
    local source layers."""
    return (
        modal.Image.from_registry(SGLANG_IMAGE_TAG)
        .run_commands(
            f"cd /sgl-workspace/sglang && git remote add modal-fork {SGLANG_FORK_REPO}"
            f" && git fetch modal-fork {SGLANG_FORK_BRANCH} && git checkout {SGLANG_FORK_COMMIT} -- python/"
        )
        .run_commands(
            f"rm -rf {hf_cache_path}"
        )  # baked HF cache must not shadow the mounted volume
        .pip_install(
            "autoinference-utils==0.2.3",  # sglang server lifecycle
            "fastapi",
            "httpx",
            "uvicorn",  # the stitch sidecar
            "zstandard",
            "xxhash",
            "blake3",  # engine-side weight-staging checksum
            "fastsafetensors",
            *extra_packages,
        )
        .env(
            {
                **_SERVING_ENV,
                **(extra_env or {}),
                "EXPERIMENT_CONFIG": experiment,
                **(
                    {"DELTA_VOLUME_NAME": delta_volume_name}
                    if delta_volume_name
                    else {}
                ),
                **({"RUN_ID": run_id} if run_id else {}),
            }
        )
        # The kernel-cache volume can't mount over a non-empty path — clear it as the final
        # filesystem step (repopulated on boot).
        .run_commands("rm -rf /root/.cache/sglang")
        .add_local_python_source("stitch")
        .add_local_dir(
            str(_COOKBOOK_DIR), remote_path="/root/cookbook", ignore=["**/__pycache__"]
        )
    )
