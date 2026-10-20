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


@dataclass(frozen=True)
class SGLangRuntime:
    """An immutable SGLang source overlay and its ABI-compatible base image."""

    image: str
    repository: str
    branch: str
    commit: str
    # Files the image's editable install generated inside python/sglang, such
    # as compiled Rust extensions, which the overlay keeps.
    build_products: tuple[str, ...] = ()
    # Packages the overlay imports that the image does not install.
    packages: tuple[str, ...] = ()


DEFAULT_SGLANG_RUNTIME = SGLangRuntime(
    # lmsysorg/sglang:v0.5.22 for linux/amd64.
    image="lmsysorg/sglang@sha256:5fc09edb6a0bb1d5b6992b543723c66d08f86d7a507e951e87e0d05db4ac766e",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-v0.5.22",
    commit="bdaac6e3b906fc315f6886be546dedde308e4f8b",
    # The image runs with SGLANG_RUST_BUILD_MODE=never, so a deleted extension
    # cannot be rebuilt; the radix cache's default TreeCore is one of them.
    build_products=(
        "sglang/_version.py",
        "sglang/srt/mem_cache/rust_tree_core/*.so",
        "sglang/srt/rust_extensions/*.so",
    ),
    # The tool-schema check in serving_chat.py.
    packages=("jsonschema-rs==0.58.6",),
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
        .run_commands(_overlay_command(runtime))
        .run_commands(
            f"rm -rf {hf_cache_path}"
        )  # baked HF cache must not shadow the mounted volume
        .pip_install(
            "autoinference-utils==0.2.3",  # sglang server lifecycle
            "fastapi",
            "httpx",
            "msgspec",
            "uvicorn",  # the stitch sidecar
            "zstandard",
            "xxhash",
            "blake3",  # engine-side weight-staging checksum
            "fastsafetensors",
            *runtime.packages,
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


def _overlay_command(runtime: SGLangRuntime) -> str:
    """Replace the image's sglang package with the pinned fork's sources."""
    checkout = "/tmp/stitch-sglang-overlay"
    python_dir = "/sgl-workspace/sglang/python"
    products = "/tmp/stitch-sglang-build-products.tar"
    steps = [
        f"rm -rf {checkout}",
        f"git clone --filter=blob:none --single-branch --branch {runtime.branch}"
        f" {runtime.repository} {checkout}",
        f"git -C {checkout} checkout --detach {runtime.commit}",
    ]
    if runtime.build_products:
        # tar fails the build when a listed product is missing from the image.
        steps.append(
            f"(cd {python_dir} && tar -cf {products} {' '.join(runtime.build_products)})"
        )
    steps += [f"rm -rf {python_dir}/sglang", f"cp -a {checkout}/python/. {python_dir}/"]
    if runtime.build_products:
        steps.append(f"tar -xf {products} -C {python_dir} && rm {products}")
    steps.append(f"rm -rf {checkout}")
    return " && ".join(steps)
