"""Persistent JIT kernel cache for trainer containers.

Triton (flash-linear-attention, TransformerEngine's Triton kernels, Inductor's
generated kernels) and TorchInductor write compiled artifacts to container scratch by
default (``~/.triton/cache``, ``/tmp/torchinductor_<user>``), so every fresh trainer
container — every launch and every retry — recompiles and re-autotunes from zero.
Mounting a Volume at ``KERNEL_CACHE_PATH`` and pointing the cache directories under it
lets later containers read the artifacts back and compile only genuinely new
(kernel, shape) keys. With flash-linear-attention's default ``cache_results=True`` the
autotuner's timing tables (``*.autotune.json``) land in the same tree, so the benchmark
sweep is skipped as well.

The tree is keyed by GPU compute capability and the torch/triton versions:

* Compute capability, because Triton emits arch-specific SASS (``ptxas
  --gpu-name=sm_103a`` on B300) that is useless on any other arch. One Volume is
  shared by every recipe, so B200 (``sm100``), B300 (``sm103``) and H100 (``sm90``)
  jobs each get their own tree.
* torch and triton versions, because they are the toolchain identity of the trainer
  image that matters for the artifacts: Triton hashes its own version and the backend
  into every entry and Inductor invalidates on the torch build, so a shared tree
  would not be *wrong* — the key just keeps each toolchain's tree separate so lookups
  walk only entries they can use and a stale tree can be pruned by deleting one
  directory.

Kernel source changes (a flash-linear-attention or TransformerEngine bump) need no
key change: both caches hash the kernel source into the entry name, so old entries
are simply never read again.
"""

from __future__ import annotations

import os
import subprocess
from importlib import metadata
from pathlib import Path

TRITON_CACHE_ENV = "TRITON_CACHE_DIR"
INDUCTOR_CACHE_ENV = "TORCHINDUCTOR_CACHE_DIR"
UNKNOWN_GPU = "gpu-unknown"


def compute_capability() -> str:
    """``sm<major><minor>`` of the first visible GPU (``sm103`` on B300), via
    nvidia-smi so no CUDA context is created in the calling process. Falls back to
    ``gpu-unknown`` rather than failing the attempt: Triton and Inductor still key
    every entry by arch, so a shared tree is safe, only less tidy."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        first = out.strip().splitlines()[0].strip()
        major, _, minor = first.partition(".")
        return f"sm{int(major)}{int(minor or 0)}"
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        print(f"WARNING: could not determine GPU compute capability: {exc}")
        return UNKNOWN_GPU


def cache_key(
    *, compute_capability: str, torch_version: str, triton_version: str
) -> str:
    return f"{compute_capability}/torch-{torch_version}-triton-{triton_version}"


def environment(root: str | os.PathLike[str]) -> dict[str, str]:
    """Cache-directory env vars for this container's GPU and toolchain, with the
    directories created. Set these in the process that runs ``ray start``: Ray
    workers inherit the raylet's environment, and a ``runtime_env`` ``env_vars``
    overlay (as in miles' actor factory) is merged on top rather than replacing it."""
    key = cache_key(
        compute_capability=compute_capability(),
        torch_version=metadata.version("torch"),
        triton_version=metadata.version("triton"),
    )
    base = Path(root) / key
    env = {
        TRITON_CACHE_ENV: str(base / "triton"),
        INDUCTOR_CACHE_ENV: str(base / "inductor"),
    }
    for path in env.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return env
