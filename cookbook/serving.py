"""Shared B200 SGLang serving-image builder for the disagg rollout pool.

The trainer half of each disagg example runs on its own slime/miles Megatron
image. The rollout half serves the model on a Blackwell SGLang build (fa4 /
cutlass-dsl prerelease kernels + the tokenspeed MLA attention backend) that
loads the served checkpoint's *own* quant config — there is no ``--quantization``
flag baked in, so INT4 (slime/Kimi K2.6) vs NVFP4 (miles) is a property of the
served checkpoint, not of this image. The per-trainer ``serving.py`` modules are
thin wrappers over this builder.

Two deliberate choices keep the image lean (identical for every trainer):

  * **The runtime SGLang is the modal-projects/sglang pin below**, checked out
    over an ``lmsysorg/sglang`` base that supplies only the environment
    (kernels, CUDA deps). The pin carries /pull_weights + the hardened
    local_checkpoint receiver and the quantized-reload restore protocol.
  * **No trainer package is installed.** The delta decode/apply lives in the
    engine behind ``/pull_weights``, so the pool imports neither slime nor
    miles and is trainer-agnostic — the checksum/zstd deps are the engine-side
    receiver's, not a trainer decoder's.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Pinned SGLang: the base image supplies the environment (kernels, CUDA deps);
# the actual runtime sglang code is the pin's python tree checked out over it.
# weight-sync-miles = sglang-miles + the restore protocol (reload == init for
# quantized weights), /pull_weights + hardened local_checkpoint receiver, and
# load-plan record/replay. Oracle-validated on GLM-4.5-Air-FP8 + Kimi-K2.6-NVFP4.
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.14"
SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "weight-sync-miles"
SGLANG_FORK_COMMIT = "f3a39d4873fecc03d5e5a30248065e334c85edf8"

# SGLang runtime tunables carried over from the standalone B200 deployment.
SERVING_IMAGE_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
    # Reload record/replay (model_loader/load_plan.py) is DISABLED: the live
    # e2e showed the first (record) reload hangs in the fused-MoE weight loader
    # on GLM-4.5-Air, wedging the engine so current_version never advances. The
    # record path was never exercised by the reload oracle, so it's unvalidated;
    # keep it off until debugged. Reloads fall back to the plain restore-protocol
    # path (the one the oracle validated). Re-enable per-model once fixed.
    "SGLANG_ENABLE_RELOAD_LOAD_PLAN": "0",
}


def build_b200_serving_image(
    *,
    hf_cache_path: str,
    experiment: str,
    delta_volume_name: str,
    clear_sglang_cache_at_end: bool = False,
) -> modal.Image:
    """Build the rollout-pool serving image (see module docstring).

    No trainer package is installed: the delta apply lives in the engine
    behind ``/pull_weights``, so the pool is trainer-agnostic. The whole
    ``cookbook`` package is mounted at ``/root/cookbook``, so the sidecar
    subprocess can import both ``cookbook.<name>.sidecar`` and the shared
    ``cookbook.sidecar`` spine it delegates to. Mounting the package (not just
    the per-trainer subdir) is required because the sidecar is launched as
    ``python3 -m cookbook.<name>.sidecar`` and is never imported at deploy
    time, so Modal's import-time automounting never sees the shared module.

    ``clear_sglang_cache_at_end`` removes ``/root/.cache/sglang`` as the final
    filesystem step, required when ``modal_train`` mounts a kernel-cache volume
    there (a volume can't mount over a non-empty path).
    """
    image = (
        modal.Image.from_registry(SGLANG_IMAGE_TAG)
        .run_commands(
            f"cd /sgl-workspace/sglang && git remote add modal-fork {SGLANG_FORK_REPO}"
            f" && git fetch modal-fork {SGLANG_FORK_BRANCH}"
            f" && git checkout {SGLANG_FORK_COMMIT} -- python/",
        )
        # The base image bakes in an HF cache; remove it so it cannot shadow the
        # cache volume mounted at the same path.
        .run_commands(f"rm -rf {hf_cache_path}")
        # No trainer package: the delta apply lives in the engine
        # (/pull_weights + weight_sync/local_checkpoint), so the pool imports
        # neither slime nor miles. The checksum/zstd deps below are the
        # ENGINE-side receiver's.
        .pip_install(
            "autoinference-utils==0.2.0",  # SGLang server lifecycle for the rollout pool
            "fastapi",  # stitch sidecar
            "httpx",  # stitch sidecar
            "uvicorn",  # stitch sidecar
            # engine-side local_checkpoint receiver: zstd decompress + xxhash
            # (xxh3-128 default) / blake3 checksums.
            "zstandard",
            "xxhash",
            "blake3",
        )
    )
    if clear_sglang_cache_at_end:
        # MUST be the last filesystem step: modal_train mounts a kernel-cache
        # volume at /root/.cache/sglang, and a volume can't mount over a non-empty
        # path. The sglang-kernel/flashinfer and trainer installs above populate
        # this dir, so clear it AFTER them (the volume repopulates the JIT/
        # autotuner cache on first boot).
        image = image.run_commands("rm -rf /root/.cache/sglang")
    return (
        image.env(
            {
                "EXPERIMENT_CONFIG": experiment,
                # Read by the engine's /pull_weights pre-read hook
                # (stitch.providers.modal.pull_weights_pre_read_hook) and the
                # sidecar's bulletin refresh: the Volume must be reloaded before
                # published versions become visible on this host.
                "DELTA_VOLUME_NAME": delta_volume_name,
                **SERVING_IMAGE_ENV,
            }
        )
        # Mounted at container start (not copied into the image) so code edits to
        # stitch / the sidecar never rebuild the image. Modal puts /root on
        # PYTHONPATH for subprocesses (the sidecar). The whole cookbook package is
        # mounted (not just the per-trainer subdir) so the sidecar can import the
        # shared cookbook.sidecar spine it is a thin adapter over.
        .add_local_python_source("stitch")
        .add_local_dir(
            Path(__file__).parent,
            remote_path="/root/cookbook",
            ignore=["**/__pycache__"],
        )
    )
