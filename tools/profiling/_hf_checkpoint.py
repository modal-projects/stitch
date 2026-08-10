"""Helpers for profiling public Hugging Face checkpoints on Modal Volumes."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path


def download_snapshot(
    repo_id: str,
    revision: str,
    cache_dir: str,
    *,
    commit: Callable[[], None],
) -> str:
    """Download and durably commit a model-sized snapshot in bounded batches."""

    from huggingface_hub import HfApi, snapshot_download

    files = HfApi().list_repo_files(repo_id=repo_id, revision=revision)
    weights = sorted(path for path in files if path.endswith(".safetensors"))
    batches = [sorted(set(files) - set(weights))]
    batches.extend(weights[offset : offset + 4] for offset in range(0, len(weights), 4))
    for index, batch in enumerate(batches, start=1):
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=batch,
            max_workers=4,
        )
        commit()
        print(f"Committed checkpoint download batch {index}/{len(batches)}")

    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    print(f"Downloaded {repo_id}@{revision} to {path}")
    return path


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
