"""The slime trainer image + the versions pinned to launch a slime run.

The serving half is separate and shared (common/serving_image.py) — the pool installs no
trainer package, so slime and miles serve on the identical weight-sync sglang image.
"""

from __future__ import annotations

import modal

from cookbook.common import trainer_image as common_trainer_image

SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"
SLIME_REPO_URL = "https://github.com/modal-projects/slime.git"
# Pin to an exact commit, not the branch tip (the fetch+checkout is a cached layer).
SLIME_REPO_REF = "11bb0fa48aa37d5c54fe297143c6bc1d40f311bf"
SLIME_ROOT = "/root/slime"


def build_trainer_image(*, hf_cache_path: str, experiment: str, run_id: str | None = None, slime_local: str | None = None) -> modal.Image:
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
    )
    image = common_trainer_image.add_common_layers(image, experiment=experiment, run_id=run_id)
    if slime_local:  # dev overlay: replace the cloned fork with a local checkout (no rebuild)
        image = image.add_local_dir(slime_local, remote_path=SLIME_ROOT, ignore=[".git", "**/__pycache__", "**/*.pyc"])
    return image
