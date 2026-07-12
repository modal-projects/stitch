"""Resolve the checkpoint used by both the engine and delta materializer."""

from __future__ import annotations

import os
from collections.abc import Callable


def is_hf_repo_id(spec: str) -> bool:
    """Return whether ``spec`` should be resolved through the Hugging Face cache."""
    return not os.path.isabs(spec)


def resolve_base_checkpoint(spec: str, *, snapshot_download: Callable[..., str]) -> str:
    """Resolve an HF repo locally, or return an absolute checkpoint path."""
    if is_hf_repo_id(spec):
        return snapshot_download(spec, local_files_only=True)
    return spec
