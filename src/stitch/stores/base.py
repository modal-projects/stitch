"""The ``Store`` port — where published versions and the ``latest`` pointer live.

Instances subclass this base: ``stores/modal_volume.py`` (Modal Volume) and
``stores/s3.py`` (S3); add NFS etc. as new subclasses. Subclasses override the version +
pointer methods; ``commit`` has a no-op default.
"""

from __future__ import annotations

from stitch.versions import VersionManifest, VersionRef


class Store:
    """A versioned checkpoint holder: the version bytes plus the monotonic
    pointer / run-epoch coordination. Subclasses override every method."""

    def refresh(self) -> None:
        """Make other hosts' writes visible (Volume reload; no-op if strongly consistent)."""
        raise NotImplementedError

    def read_pointer(self) -> VersionRef | None:
        """The current ``latest`` pointer, or None if no run has been claimed."""
        raise NotImplementedError

    def advance_pointer(self, ref: VersionRef) -> None:
        """Move ``latest`` to ``ref`` — the caller has already run ``decide_pointer_move``."""
        raise NotImplementedError

    def claim(self, run_id: str) -> None:
        """Start a new run epoch at base, forking the version space."""
        raise NotImplementedError

    def read_manifest(self, ref: VersionRef) -> VersionManifest:
        """The manifest for ``ref``, derived from its on-disk HF index."""
        raise NotImplementedError

    def publish(self, manifest: VersionManifest, files_dir: str) -> None:
        """Durably write a version's files; must be visible before ``advance_pointer``."""
        raise NotImplementedError

    def materialize(self, ref: VersionRef) -> str:
        """Ensure the version's files are locally readable and return their directory
        (hides mount vs download)."""
        raise NotImplementedError

    def commit(self) -> None:
        """Durably flush this host's pending writes (e.g. one trainer rank's shard of a
        version). Default no-op — only a store whose writes aren't immediately durable
        (a Volume) needs it; the publish hook calls it on every rank."""
