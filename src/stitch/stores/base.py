"""The ``Store`` port — where published versions and the ``latest`` pointer live.

Instances: ``stores/modal_volume.py`` (Modal Volume). Add S3 / NFS as new files.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from stitch.versions import VersionManifest, VersionRef


@runtime_checkable
class Store(Protocol):
    """A versioned checkpoint holder: the version bytes plus the monotonic
    pointer / run-epoch coordination."""

    def refresh(self) -> None:
        """Make other hosts' writes visible (Volume reload; no-op if strongly consistent)."""
        ...

    def read_pointer(self) -> VersionRef | None: ...

    def advance_pointer(self, ref: VersionRef) -> None:
        """Move ``latest`` to ``ref`` — the caller has already run ``decide_pointer_move``."""
        ...

    def claim(self, run_id: str) -> None:
        """Start a new run epoch at base, forking the version space."""
        ...

    def read_manifest(self, ref: VersionRef) -> VersionManifest: ...

    def publish(self, manifest: VersionManifest, files_dir: str) -> None:
        """Durably write a version's files; must be visible before ``advance_pointer``."""
        ...

    def open_version(self, ref: VersionRef) -> str:
        """A local directory of the version's files, guaranteed readable before returning."""
        ...
