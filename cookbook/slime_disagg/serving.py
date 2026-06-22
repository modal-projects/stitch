"""Dedicated B200 native-INT4 SGLang serving image for the rollout pool.

The trainer half of this example runs on the slime/Megatron image
(``modal_train.image``). The rollout half that serves Kimi K2.6 needs a
different stack: a Blackwell SGLang build that loads the model's native
compressed-tensors INT4 (W4A16) checkpoint and runs MLA attention with the
tuned hierarchical KV cache. This module builds that image.

Two deliberate choices keep it lean:

  * **SGLang comes from the modal-projects/sglang fork** (Blackwell fa4 /
    cutlass-dsl prerelease kernels + the tokenspeed MLA attention backend) — the
    same build the standalone 4xB200 Kimi deployment uses. The FP4-specific
    ``--quantization modelopt_fp4`` is *not* baked in here; the served
    checkpoint's own ``compressed-tensors`` config drives INT4 weight loading
    (see the config module's ``SGLANG_SERVER_ARGS``).
  * **slime is cloned ``--no-deps`` for one module.** The sidecar only imports
    ``slime.utils.disk_delta`` (stdlib + numpy + zstandard; xxhash/blake3 lazy),
    so Megatron is intentionally absent from the rollout pool.

The single un-de-risked axis is whether this fork serves native-INT4 MLA MoE on
Blackwell as cleanly as it serves NVFP4 (it is proven for NVFP4). Verify on a
warm container before a long run; the FP4 path is the known-good fallback.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Pinned Blackwell SGLang fork — matches the standalone 4xB200 Kimi serve recipe.
# Only the python sources are checked out over the prebuilt base image's kernels.
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.12"
SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "timmy/dflash-fa4-fp8"
SGLANG_FORK_COMMIT = "dafb2b325b40298c5097564811463c585b7e9814"

# SGLang runtime tunables carried over from the standalone B200 deployment.
SERVING_IMAGE_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
}


def build_int4_b200_serving_image(
    *,
    slime_repo_url: str,
    slime_repo_ref: str,
    slime_root: str,
    hf_cache_path: str,
    experiment: str,
) -> modal.Image:
    """Build the rollout-pool serving image (see module docstring).

    The slime fork ref / root and the HF cache path are passed in by
    ``modal_train`` so the pool and the trainer pin the identical slime commit
    (the sidecar's ``disk_delta`` must match the trainer's delta encoder).
    """
    # serving.py lives in cookbook/slime_disagg, mounted to /root/cookbook/slime_disagg
    # exactly as the trainer image mounts it, so `cookbook.slime_disagg.sidecar`
    # imports identically in either container.
    slime_disagg_dir = Path(__file__).parent
    return (
        modal.Image.from_registry(SGLANG_IMAGE_TAG)
        .run_commands(
            f"cd /sgl-workspace/sglang && git remote add modal-fork {SGLANG_FORK_REPO}"
            f" && git fetch modal-fork {SGLANG_FORK_BRANCH}"
            f" && git checkout {SGLANG_FORK_COMMIT} -- python/",
        )
        # Pre-release CUDA wheels (cutlass-dsl / sglang-kernel / flash-attn-4) —
        # keep the deployment's known-good pip resolution.
        .run_commands(
            "pip install nvidia-cutlass-dsl==4.5.1 sglang-kernel==0.4.3 'flash-attn-4>=4.0.0b10'"
        )
        # flash-attn-4 checks for the deprecated MmaFP8Op but cutlass-dsl 4.5.1 now
        # generates MmaF8F6F4Op instead. Patch the isinstance check to handle both.
        .run_commands(
            "sed -i 's/isinstance(op, tcgen05.mma.MmaFP8Op)/isinstance(op, (tcgen05.mma.MmaFP8Op, tcgen05.mma.MmaF8F6F4Op))/' "
            "/usr/local/lib/python3.12/dist-packages/flash_attn/cute/blackwell_helpers.py"
        )
        # The base image bakes in an HF cache; remove it so it cannot shadow the
        # cache volume mounted at the same path.
        .run_commands(f"rm -rf {hf_cache_path}")
        # slime --no-deps gives the sidecar `slime.utils.disk_delta` (host-side
        # delta apply). Megatron is NOT installed — the pool never trains. Pin the
        # SAME ref the trainer image uses so the delta encoder/decoder match.
        .run_commands(
            f"git clone --depth 1 {slime_repo_url} {slime_root}"
            f" && cd {slime_root}"
            f" && git fetch --depth 1 origin {slime_repo_ref}"
            f" && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {slime_root}"
        )
        .pip_install(
            "autoinference-utils==0.2.0",  # SGLang server lifecycle for the rollout pool
            "fastapi",  # stitch sidecar
            "httpx",  # stitch sidecar
            "uvicorn",  # stitch sidecar
            # disk_delta host-side apply: zstd decompress + xxhash (xxh3-128
            # default) / blake3 checksums. slime is installed --no-deps.
            "zstandard",
            "xxhash",
            "blake3",
        )
        .env({"EXPERIMENT_CONFIG": experiment, **SERVING_IMAGE_ENV})
        # Mounted at container start (not copied into the image) so code edits to
        # stitch / the sidecar never rebuild the image. Modal puts /root on
        # PYTHONPATH for subprocesses (the sidecar).
        .add_local_python_source("stitch")
        .add_local_dir(
            slime_disagg_dir,
            remote_path="/root/cookbook/slime_disagg",
            ignore=["**/__pycache__"],
        )
    )
