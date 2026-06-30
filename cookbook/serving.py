"""Shared B200 SGLang serving-image builder for the disagg rollout pool.

The trainer half of each disagg example runs on its own slime/miles Megatron
image. The rollout half serves the model on a Blackwell SGLang build (fa4 /
cutlass-dsl prerelease kernels + the tokenspeed MLA attention backend) that
loads the served checkpoint's *own* quant config — there is no ``--quantization``
flag baked in, so INT4 (slime/Kimi K2.6) vs NVFP4 (miles) is a property of the
served checkpoint, not of this image. The per-trainer ``serving.py`` modules are
thin wrappers that pass their trainer package + repo pin here.

Two deliberate choices keep the image lean (identical for every trainer):

  * **SGLang comes from the modal-projects/sglang fork** pinned below — the same
    build the standalone 4xB200 Kimi deployment uses (proven for NVFP4; the INT4
    MLA-MoE path is the one un-de-risked axis — verify on a warm container).
  * **The trainer package is cloned ``--no-deps`` for one module.** The sidecar
    only imports ``<trainer>.utils.disk_delta`` (stdlib + numpy + zstandard;
    xxhash/blake3 lazy), so Megatron is intentionally absent — the pool never
    trains. Pin the SAME ref the trainer image uses so the host-side delta
    decoder matches the trainer's delta encoder.
"""

from __future__ import annotations

from pathlib import Path

import modal

# Pinned Blackwell SGLang fork. Only the python sources are checked out over the
# prebuilt base image's kernels.
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


def build_b200_serving_image(
    *,
    trainer_repo_url: str,
    trainer_repo_ref: str,
    trainer_root: str,
    cookbook_dir: Path,
    hf_cache_path: str,
    experiment: str,
    shallow_clone: bool = True,
    clear_sglang_cache_at_end: bool = False,
) -> modal.Image:
    """Build the rollout-pool serving image (see module docstring).

    ``trainer_repo_url`` / ``trainer_repo_ref`` / ``trainer_root`` pin the
    ``--no-deps`` trainer checkout (so the pool's ``disk_delta`` matches the
    trainer's encoder). ``cookbook_dir`` is the per-trainer cookbook package dir;
    it is mounted at ``/root/cookbook/<name>`` exactly as the trainer image mounts
    it, so ``cookbook.<name>.sidecar`` imports identically in either container.

    ``shallow_clone`` does a ``--depth 1`` clone+fetch (fine when the ref is a
    branch tip or recent commit). ``clear_sglang_cache_at_end`` removes
    ``/root/.cache/sglang`` as the final filesystem step, required when
    ``modal_train`` mounts a kernel-cache volume there (a volume can't mount over
    a non-empty path).
    """
    depth = "--depth 1 " if shallow_clone else ""
    fetch_depth = "--depth 1 " if shallow_clone else ""
    image = (
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
        # trainer pkg --no-deps gives the sidecar `<trainer>.utils.disk_delta`
        # (host-side delta apply). Megatron is NOT installed — the pool never
        # trains. Pin the SAME ref the trainer image uses so encoder == decoder.
        .run_commands(
            f"git clone {depth}{trainer_repo_url} {trainer_root}"
            f" && cd {trainer_root}"
            f" && git fetch {fetch_depth}origin {trainer_repo_ref}"
            f" && git checkout FETCH_HEAD"
            f" && python3 -m pip install --no-deps -e {trainer_root}"
        )
        .pip_install(
            "autoinference-utils==0.2.0",  # SGLang server lifecycle for the rollout pool
            "fastapi",  # stitch sidecar
            "httpx",  # stitch sidecar
            "uvicorn",  # stitch sidecar
            # disk_delta host-side apply: zstd decompress + xxhash (xxh3-128
            # default) / blake3 checksums. trainer pkg is installed --no-deps.
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
        image.env({"EXPERIMENT_CONFIG": experiment, **SERVING_IMAGE_ENV})
        # Mounted at container start (not copied into the image) so code edits to
        # stitch / the sidecar never rebuild the image. Modal puts /root on
        # PYTHONPATH for subprocesses (the sidecar).
        .add_local_python_source("stitch")
        .add_local_dir(
            cookbook_dir,
            remote_path=f"/root/cookbook/{cookbook_dir.name}",
            ignore=["**/__pycache__"],
        )
    )
