"""Helpers for profiling public Hugging Face checkpoints on Modal Volumes."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from cookbook.common.hf_download import download_cached_snapshot


def download_snapshot(
    download_file: Any,
    repo_id: str,
    revision: str,
    *,
    volume: Any,
) -> str:
    """Download and durably commit a profiling snapshot across Modal workers."""

    return download_cached_snapshot(
        download_file,
        repo_id,
        revision,
        volume=volume,
    )


def materialize_checkpoint_view(source_dir: str, target_dir: str) -> None:
    """Expose cached weights and real sibling metadata as a local checkpoint."""

    source = Path(source_dir)
    target = Path(target_dir)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for path in source.rglob("*"):
        destination = target / path.relative_to(source)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.suffix == ".safetensors":
            destination.symlink_to(path.resolve())
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination, follow_symlinks=True)
