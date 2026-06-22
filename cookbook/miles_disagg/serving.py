"""Dedicated B200 NVFP4 SGLang serving image for the rollout pool.

The trainer half of this example runs on the miles/Megatron image
(``modal_train.image``). The rollout half that serves the NVFP4 checkpoint needs
a different stack: a Blackwell SGLang build that loads the model's NVFP4 weights
and runs MLA attention with the tuned hierarchical KV cache. This module builds
that image — the miles twin of cookbook/slime_disagg/serving.py.

Two deliberate choices keep it lean:

  * **SGLang comes from the modal-projects/sglang fork** (Blackwell fa4 /
    cutlass-dsl prerelease kernels + the tokenspeed MLA attention backend) — the
    same build the slime example uses, proven for NVFP4. No ``--quantization``
    flag is baked in; the served checkpoint's own NVFP4 quant config drives
    weight loading (see the config module's ``SGLANG_SERVER_ARGS``).
  * **miles is cloned ``--no-deps`` for one module.** The sidecar only imports
    ``miles.utils.disk_delta`` (stdlib + numpy + zstandard; xxhash/blake3 lazy),
    so Megatron is intentionally absent from the rollout pool. NOTE: this assumes
    ``miles.utils.disk_delta`` is import-light (no heavy package __init__ chain);
    verify on a warm container during bring-up.

Pin the SAME miles ref the trainer image uses so the host-side delta decoder
matches the trainer's delta encoder.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Pinned Blackwell SGLang fork — matches the slime 4xB200 Kimi serve recipe,
# proven for NVFP4. Only the python sources are checked out over the prebuilt
# base image's kernels.
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.12"
SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "timmy/dflash-fa4-fp8"
SGLANG_FORK_COMMIT = "dafb2b325b40298c5097564811463c585b7e9814"

# SGLang runtime tunables carried over from the B200 deployment.
SERVING_IMAGE_ENV = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
}
# NOTE: kernel-cache persistence is handled by mounting SGLang's cache dir
# (/root/.cache/sglang, which nests flashinfer/DeepGemm) as a volume in
# modal_train — no env override here, so SGLang's default placement is kept.


def build_nvfp4_b200_serving_image(
    *,
    miles_repo_url: str,
    miles_repo_ref: str,
    miles_root: str,
    hf_cache_path: str,
    experiment: str,
) -> modal.Image:
    """Build the rollout-pool serving image (see module docstring).

    The miles fork ref / root and the HF cache path are passed in by
    ``modal_train`` so the pool and the trainer pin the identical miles commit
    (the sidecar's ``disk_delta`` must match the trainer's delta encoder).
    """
    # serving.py lives in cookbook/miles_disagg, mounted to /root/cookbook/miles_disagg
    # exactly as the trainer image mounts it, so `cookbook.miles_disagg.sidecar`
    # imports identically in either container.
    miles_disagg_dir = Path(__file__).parent
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
        # miles --no-deps gives the sidecar `miles.utils.disk_delta` (host-side
        # delta apply). Megatron is NOT installed — the pool never trains. Pin the
        # SAME ref the trainer image uses so the delta encoder/decoder match.
        .run_commands(
            f"git clone {miles_repo_url} {miles_root}"
            f" && cd {miles_root}"
            f" && git fetch origin {miles_repo_ref}"
            f" && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {miles_root}"
        )
        .pip_install(
            "autoinference-utils==0.2.0",  # SGLang server lifecycle for the rollout pool
            "fastapi",  # stitch sidecar
            "httpx",  # stitch sidecar
            "uvicorn",  # stitch sidecar
            # disk_delta host-side apply: zstd decompress + xxhash (xxh3-128
            # default) / blake3 checksums. miles is installed --no-deps.
            "zstandard",
            "xxhash",
            "blake3",
        )
        # MUST be the last filesystem step: modal_train mounts the
        # miles-sglang-cache volume at /root/.cache/sglang, and a volume can't
        # mount over a non-empty path. The sglang-kernel/flashinfer and miles
        # installs above populate this dir, so clear it AFTER them (the volume
        # repopulates the JIT/autotuner cache on first boot).
        .run_commands("rm -rf /root/.cache/sglang")
        .env({"EXPERIMENT_CONFIG": experiment, **SERVING_IMAGE_ENV})
        # Mounted at container start (not copied into the image) so code edits to
        # stitch / the sidecar never rebuild the image. Modal puts /root on
        # PYTHONPATH for subprocesses (the sidecar).
        .add_local_python_source("stitch")
        .add_local_dir(
            miles_disagg_dir,
            remote_path="/root/cookbook/miles_disagg",
            ignore=["**/__pycache__"],
        )
    )
