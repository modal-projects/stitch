"""Persistent Triton/TorchInductor kernel cache.

Both default to container scratch, so every fresh container recompiles and
re-autotunes from zero. Pointing the cache dirs into a Volume, keyed by GPU compute
capability (SASS is arch-specific) and torch/triton version, lets later containers
reuse them. Kernel source bumps need no key change: both caches hash the source.
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
    """``sm<major><minor>`` of the first visible GPU via nvidia-smi (no CUDA context).
    Falls back to a shared tree rather than failing: entries are arch-hashed anyway."""
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
    """Cache-dir env vars for this container's GPU and toolchain (dirs created).
    Set them before ``ray start``: Ray workers inherit the raylet's environment."""
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
