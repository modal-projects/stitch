"""Resolve a base-checkpoint spec to a local directory.

The standalone provider boots the engine from, and seeds every delta onto, one
base checkpoint the deployer names. The spec is either an HF repo id resolved
from the local hub cache (a public/cached base model) or an absolute directory
(an S3 CloudBucketMount, a prep volume, or an already-resolved cache snapshot).
Both forms are just a canonical HF-readable checkpoint dir; which one a customer
uses is their choice, so the provider assumes nothing about the base beyond that
it loads and that its tensors match the deltas built against it.
"""

from __future__ import annotations

import os
from collections.abc import Callable


def is_hf_repo_id(spec: str) -> bool:
    """True when ``spec`` is an HF repo id to resolve from the local cache, False
    when it is an explicit local directory. Anything that is not an absolute path
    is treated as a repo id, so the check does not depend on which volumes/mounts
    happen to be attached to the caller (the pool mounts the S3 base; the
    downloader function does not)."""
    return not os.path.isabs(spec)


def resolve_base_checkpoint(
    spec: str, *, snapshot_download: Callable[..., str]
) -> str:
    """Return the local directory the engine boots from and deltas seed onto.

    An absolute-path spec is that directory as-is; a repo-id spec is resolved
    from the local hub cache (``local_files_only=True`` — the cache is prewarmed
    by ``download_model``, so no network fetch happens on the serving path).
    """
    if is_hf_repo_id(spec):
        return snapshot_download(spec, local_files_only=True)
    return spec
