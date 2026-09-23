"""The trainer-agnostic weight-sync SGLang image shared by every recipe.

No trainer package is installed: delta application lives in the engine behind
``/prepare_weight_update`` and ``/commit_weight_update``. Precision comes from
the served checkpoint, not a ``--quantization`` flag. The fork pin carries
verified checkpoint materialization and disk or CPU staging. See
``SGLANG_FORK.md`` for the patch stack and how to re-port it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import modal

_COOKBOOK_DIR = Path(__file__).resolve().parent.parent
_PATCHES_DIR = _COOKBOOK_DIR / "common" / "patches"
_OVERLAY_DIR = "/tmp/stitch-sglang-overlay"
_OVERLAY_PATCHES_DIR = "/tmp/stitch-sglang-patches"


@dataclass(frozen=True)
class SGLangRuntime:
    """An immutable SGLang source overlay and its ABI-compatible base image.

    ``patches`` are local ``git diff`` files applied to the fork checkout at image
    build time, before its ``python/`` tree is copied over the base image. Each
    must apply cleanly to ``commit`` or the build fails; see ``SGLANG_FORK.md``.
    """

    image: str
    repository: str
    branch: str
    commit: str
    patches: tuple[str, ...] = ()


DEFAULT_SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.20",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-v0.5.20",
    commit="18df9cb22e5b7c0b3dc5303376bf61c4a4090db4",
    patches=(str(_PATCHES_DIR / "sglang-gemma-rmsnorm-staged-load.patch"),),
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
    remote_patches: list[str] = []
    for patch in runtime.patches:
        local = Path(patch)
        if not local.is_file():
            raise FileNotFoundError(f"SGLang runtime patch not found: {local}")
        remote = f"{_OVERLAY_PATCHES_DIR}/{local.name}"
        if remote in remote_patches:
            raise ValueError(f"duplicate SGLang runtime patch name: {local.name}")
        remote_patches.append(remote)
        image = image.add_local_file(str(local), remote, copy=True)
    return (
        image.run_commands(
            f"rm -rf {_OVERLAY_DIR}"
            f" && git clone --filter=blob:none --single-branch --branch {runtime.branch}"
            f" {runtime.repository} {_OVERLAY_DIR}"
            f" && git -C {_OVERLAY_DIR} checkout --detach {runtime.commit}"
            + "".join(
                # --check fails loudly on an unpatched or already-patched tree.
                f" && git -C {_OVERLAY_DIR} apply --check {remote}"
                f" && git -C {_OVERLAY_DIR} apply {remote}"
                for remote in remote_patches
            )
            + " && rm -rf /sgl-workspace/sglang/python/sglang"
            f" && cp -a {_OVERLAY_DIR}/python/. /sgl-workspace/sglang/python/"
            f" && rm -rf {_OVERLAY_DIR} {_OVERLAY_PATCHES_DIR}"
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
