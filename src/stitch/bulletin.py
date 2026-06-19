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
    read_latest,
    version_dir,
    weight_identity,
    write_latest,
)


class BulletinBoard(Protocol):
    root: Path

    async def refresh(self) -> None: ...

    def read_latest(self) -> int: ...

    def write_latest(self, version: int) -> None: ...

    def version_dir(self, version: int) -> Path: ...

    def read_manifest(self, version: int) -> VersionManifest: ...

    def publish_manifest(self, manifest: VersionManifest, version_path: str | Path | None = None) -> None: ...


class FilesystemBulletinBoard:
    """Filesystem-backed bulletin board.

    A refresh callback can be provided by storage providers such as Modal Volume.

    Two on-disk layouts are supported:

    - ``"stitch"`` (default): the engine-neutral protocol — ``versions/`` -nested
      version dirs, a JSON ``latest.json`` pointer, and a ``manifest.json`` per
      version.
    - ``"slime"``: slime's native disk-delta publish output (and the customer's
      object-store layout) — flat ``weight_v{N:06d}/`` dirs directly under the
      root, a raw ``latest`` pointer file (``"NNNNNN"``), and the manifest read
      from each version's ``model.safetensors.index.json``. This is what the
      rollout pool reconciles against; the front door advances ``latest``.
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

    def read_latest(self) -> int:
        if self.layout == "slime":
            path = self.root / "latest"
            if not path.exists():
                return 0
            text = path.read_text(encoding="utf-8").strip()
            return int(text) if text else 0
        return read_latest(self.root)

    def write_latest(self, version: int) -> None:
        if self.layout == "slime":
            atomic_write_text(self.root / "latest", f"{int(version):06d}")
        else:
            write_latest(self.root, version)

    def version_dir(self, version: int) -> Path:
        if self.layout == "slime":
            return self.root / weight_identity(version)
        return version_dir(self.root, version)

    def read_manifest(self, version: int) -> VersionManifest:
        if self.layout == "slime":
            return VersionManifest.from_slime_index(self.version_dir(version))
        return VersionManifest.read(self.version_dir(version) / "manifest.json")

    def publish_manifest(self, manifest: VersionManifest, version_path: str | Path | None = None) -> None:
        if self.layout == "slime":
            # The version dir + index.json already exist (written by slime/the
            # uploader); publishing is only advancing the monotonic pointer.
            self.write_latest(manifest.version)
            return
        target = Path(version_path) if version_path is not None else self.version_dir(manifest.version)
        manifest.write(target / "manifest.json")
        self.write_latest(manifest.version)
