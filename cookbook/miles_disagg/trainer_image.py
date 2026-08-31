"""The miles trainer image + the versions pinned to launch a miles run.

The base image bakes Megatron-LM (native --fp4-format NVFP4) + TransformerEngine; the
miles fork is cloned over it at a pinned commit. The serving half is separate and shared
(common/serving_image.py) — the pool installs no trainer package.

The fork's commit stack (what we carry over upstream miles) and how to re-rebase live in
MILES_FORK.md next to this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import modal

from cookbook.common import trainer_image as common_trainer_image

# Dated tag, never `latest`: Modal caches from_registry per tag string and won't re-pull
# a moved mutable tag, so `latest` silently serves whatever was first pulled.
MILES_IMAGE_TAG = "radixark/miles:dev-202607290235"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "778a13786471777f81c587557f68cdf03903059a"

MILES_ROOT = "/root/miles"
# Applied to the checkout at container start (after any dev overlay), so the fixes
# also cover MILES_LOCAL_DIR; each moves to the fork and drops when the pin advances.
MILES_RUNTIME_PATCHES = (
    # A resumed version counter continues at iteration+1, so versions never relabel.
    "/root/cookbook/miles_disagg/patches/miles-stable-weight-versions.patch",
    # resumed_external baseline capture tolerates 0-dim tensors (qwen3moe).
    "/root/cookbook/miles_disagg/patches/miles-seed-baseline-scalars.patch",
)
# Source-only megatron.training must be on PYTHONPATH.
MEGATRON_PATH = "/root/Megatron-LM"
TORCH_DIST_CONVERT_WRAPPER = "/root/convert_hf_to_torch_dist_modal.py"

_TORCH_DIST_WRAPPER_SRC = (
    Path(__file__).resolve().parent / "convert_hf_to_torch_dist_modal.py"
)


def build_trainer_image(
    *,
    hf_cache_path: str,
    experiment: str,
    run_id: str | None = None,
    miles_repo_ref: str = MILES_REPO_REF,
    miles_local: str | None = None,
    extra_pip_packages: tuple[str, ...] = (),
    image_run_commands: tuple[str, ...] = (),
    extra_env: Mapping[str, str] | None = None,
    copy_source: bool = False,
) -> modal.Image:
    """The miles trainer image: RDMA/EFA userspace + the pinned miles fork + the
    trainer-side delta encoder's codecs. stitch + the cookbook package are mounted so the
    trainer, Ray actors, and the sidecar subprocess resolve their imports."""
    image = (
        modal.Image.from_registry(MILES_IMAGE_TAG)
        .entrypoint([])
        # TransformerEngine 2.17 declares this dependency, but the dated Miles
        # image installs its TE wheels with --no-deps.
        .pip_install("onnxscript==0.7.1")
        # RDMA/EFA userspace so multi-node NCCL binds EFA under rdma=True instead of TCP.
        .apt_install(
            "libibverbs-dev", "libibverbs1", "libhwloc-dev", "libnl-route-3-200"
        )
        # A baked HF cache must not shadow the mounted volume.
        .run_commands(f"rm -rf {hf_cache_path}")
        .run_commands(
            f"rm -rf {MILES_ROOT}"
            f" && git clone {MILES_REPO_URL} {MILES_ROOT}"
            f" && cd {MILES_ROOT} && git fetch origin {miles_repo_ref} && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {MILES_ROOT}"
        )
        .add_local_file(
            str(_TORCH_DIST_WRAPPER_SRC), TORCH_DIST_CONVERT_WRAPPER, copy=True
        )
    )
    if extra_pip_packages:
        image = image.pip_install(*extra_pip_packages)
    if image_run_commands:
        image = image.run_commands(*image_run_commands)
    image = common_trainer_image.add_common_layers(
        image,
        experiment=experiment,
        run_id=run_id,
        extra_env=extra_env,
        copy_source=copy_source,
    )
    # Dev overlay: replace the cloned fork with a local checkout.
    if miles_local:
        image = image.add_local_dir(
            miles_local,
            remote_path=MILES_ROOT,
            ignore=[".git", "**/__pycache__", "**/*.pyc"],
            copy=copy_source,
        )
    return image
