"""The weight-sync sglang SERVING image — shared by every recipe.

Trainer-agnostic: no trainer package is installed (the delta apply lives in the engine behind
``/stage_weight_update``), so miles and slime serve on the identical image; precision comes
from the served checkpoint, not a ``--quantization`` flag. The fork pin carries asynchronous
weight staging, correct quantized weight loading, and the optional CPU delta cache. See
``SGLANG_FORK.md`` for the patch stack and how to re-port onto a newer sglang release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import modal

_COMMON_DIR = Path(__file__).resolve().parent
_COOKBOOK_DIR = _COMMON_DIR.parent

DFLASH_PREFILL_ATTN_TP_PADDING_PATCH = (
    _COMMON_DIR / "patches/sglang-dflash-prefill-attention-tp-padding.patch"
)


@dataclass(frozen=True)
class SGLangRuntime:
    """An immutable SGLang source overlay and its ABI-compatible base image."""

    image: str
    repository: str
    branch: str
    commit: str
    source_patches: tuple[Path, ...] = ()


DEFAULT_SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.16",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-v0.5.16",
    commit="7b09ce9f77c2290d5271c545d5b8b988c0479584",
    source_patches=(DFLASH_PREFILL_ATTN_TP_PADDING_PATCH,),
)

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
    image = modal.Image.from_registry(runtime.image)
    remote_patches = []
    for index, source_patch in enumerate(runtime.source_patches):
        remote_patch = f"/tmp/stitch-sglang-patch-{index}.patch"
        image = image.add_local_file(
            str(source_patch), remote_patch, copy=True
        )
        remote_patches.append(remote_patch)

    apply_patch_commands = "".join(
        " && git -C /tmp/stitch-sglang-overlay apply --check " + remote_patch
        + " && git -C /tmp/stitch-sglang-overlay apply " + remote_patch
        for remote_patch in remote_patches
    )

    return (
        image.run_commands(
            "rm -rf /tmp/stitch-sglang-overlay"
            f" && git clone --filter=blob:none --single-branch --branch {runtime.branch}"
            f" {runtime.repository} /tmp/stitch-sglang-overlay"
            f" && git -C /tmp/stitch-sglang-overlay checkout --detach {runtime.commit}"
            f"{apply_patch_commands}"
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
