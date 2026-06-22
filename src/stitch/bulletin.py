"""Bulletin board storage interfaces."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from stitch.protocol import (
    VersionManifest,
    atomic_write_text,
    format_snapshot_identity,
    parse_snapshot_identity,
    read_latest,
    version_dir,
    weight_identity,
    write_latest,
)


class BulletinBoard(Protocol):
    root: Path

    async def refresh(self) -> None: ...

    def read_latest(self) -> tuple[str | None, int]: ...

    def write_latest(self, run_id: str | None, version: int) -> None: ...

    def version_dir(self, run_id: str | None, version: int) -> Path: ...

    def read_manifest(self, run_id: str | None, version: int) -> VersionManifest: ...

    def publish_manifest(
        self,
        manifest: VersionManifest,
        version_path: str | Path | None = None,
        *,
        run_id: str | None = None,
    ) -> None: ...


class FilesystemBulletinBoard:
    """Filesystem-backed bulletin board.

    A refresh callback can be provided by storage providers such as Modal Volume.

    Two on-disk layouts are supported:

    - ``"stitch"`` (default): the engine-neutral protocol — ``versions/`` -nested
      version dirs, a JSON ``latest.json`` pointer, and a ``manifest.json`` per
      version.
    - ``"slime"``: slime's native disk-delta publish output (and the customer's
      object-store layout). Each run's chain lives under ``<run_id>/weight_v{N:06d}/``
      and the single ``latest`` pointer holds the self-identifying snapshot identity
      ``<run_id>/weight_v{N:06d}`` (a bare ``weight_v{N:06d}`` for the degenerate
      run-less layout). The manifest is read from each version's
      ``model.safetensors.index.json``. Run-id partitioning is what makes sequential
      runs collision-free: a new run writes a fresh ``<run_id>/`` chain and the
      pointer moves to it (a new run is not a rewind), so a finished run's chain can
      never be overwritten or fast-forward a cold start. The pool reconciles against
      ``latest``; the front door (or the publish hook) advances it.
    """

    def __init__(
        self,
        root: str | Path,
        refresh: Callable[[], Any] | None = None,
        *,
        layout: str = "stitch",
    ) -> None:
        if layout not in ("stitch", "slime"):
            raise ValueError(f"unknown bulletin layout {layout!r}")
        self.root = Path(root)
        self._refresh = refresh
        self.layout = layout

    async def refresh(self) -> None:
        if self._refresh is None:
            return
        result = await asyncio.to_thread(self._refresh)
        if inspect.isawaitable(result):
            await result

    def read_latest(self) -> tuple[str | None, int]:
        """The active snapshot pointer as ``(run_id, version)``.

        slime: parse ``<run_id>/weight_v{N}`` (or a bare/legacy pointer) from the
        ``latest`` file; missing/empty/unparseable -> ``(None, 0)``. stitch: the
        JSON pointer is run-less, so ``(None, <version>)``.
        """
        if self.layout == "slime":
            path = self.root / "latest"
            if not path.exists():
                return (None, 0)
            return parse_snapshot_identity(path.read_text(encoding="utf-8"))
        return (None, read_latest(self.root))

    def write_latest(self, run_id: str | None, version: int) -> None:
        if self.layout == "slime":
            atomic_write_text(self.root / "latest", format_snapshot_identity(run_id, version))
        else:
            write_latest(self.root, version)

    def version_dir(self, run_id: str | None, version: int) -> Path:
        if self.layout == "slime":
            base = self.root / run_id if run_id else self.root
            return base / weight_identity(version)
        return version_dir(self.root, version)

    def read_manifest(self, run_id: str | None, version: int) -> VersionManifest:
        if self.layout == "slime":
            return VersionManifest.from_slime_index(self.version_dir(run_id, version))
        return VersionManifest.read(self.version_dir(run_id, version) / "manifest.json")

    def publish_manifest(
        self,
        manifest: VersionManifest,
        version_path: str | Path | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        if self.layout == "slime":
            # The version dir + index.json already exist (written by slime/the
            # uploader under <run_id>/); publishing is only advancing the pointer.
            self.write_latest(run_id, manifest.version)
            return
        target = Path(version_path) if version_path is not None else self.version_dir(run_id, manifest.version)
        manifest.write(target / "manifest.json")
        self.write_latest(run_id, manifest.version)
