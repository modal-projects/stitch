"""The trainer-agnostic weight-sync SGLang image shared by every recipe.

No trainer package is installed: delta application lives in the engine behind
``/stage_weight_update``. Precision comes from the served checkpoint, not a
``--quantization`` flag. The fork pin carries asynchronous weight staging, correct
quantized weight loading, and the optional CPU delta cache. See ``SGLANG_FORK.md`` for
the patch stack and how to re-port onto a newer SGLang release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import modal


@dataclass(frozen=True)
class SGLangRuntime:
    """An immutable SGLang source overlay and its ABI-compatible base image."""

    image: str
    repository: str
    branch: str
    commit: str


DEFAULT_SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.17",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-v0.5.17",
    commit="98eff9998f136f133f6f4181e46530508bf4f2fd",
)

_COOKBOOK_DIR = Path(__file__).resolve().parent.parent

_SERVING_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_MODULES_CACHE": "/tmp/huggingface/modules",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
}


def build_serving_image(
    *,
    hf_cache_path: str,
    experiment: str,
    run_id: str | None = None,
    extra_packages: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
    runtime: SGLangRuntime = DEFAULT_SGLANG_RUNTIME,
) -> modal.Image:
    """Build the rollout-pool image for one experiment config."""
    return (
        modal.Image.from_registry(runtime.image)
        .run_commands(
            "rm -rf /tmp/stitch-sglang-overlay"
            f" && git clone --filter=blob:none --single-branch --branch {runtime.branch}"
            f" {runtime.repository} /tmp/stitch-sglang-overlay"
            f" && git -C /tmp/stitch-sglang-overlay checkout --detach {runtime.commit}"
            " && rm -rf /sgl-workspace/sglang/python/sglang"
            " && cp -a /tmp/stitch-sglang-overlay/python/. /sgl-workspace/sglang/python/"
            " && rm -rf /tmp/stitch-sglang-overlay"
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
