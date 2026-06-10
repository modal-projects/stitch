"""Bulletin board storage interfaces."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from stitch.protocol import VersionManifest, read_latest, version_dir, write_latest


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
    """

    def __init__(self, root: str | Path, refresh: Callable[[], Any] | None = None) -> None:
        self.root = Path(root)
        self._refresh = refresh

    async def refresh(self) -> None:
        if self._refresh is None:
            return
        result = await asyncio.to_thread(self._refresh)
        if inspect.isawaitable(result):
            await result

    def read_latest(self) -> int:
        return read_latest(self.root)

    def write_latest(self, version: int) -> None:
        write_latest(self.root, version)

    def version_dir(self, version: int) -> Path:
        return version_dir(self.root, version)

    def read_manifest(self, version: int) -> VersionManifest:
        return VersionManifest.read(self.version_dir(version) / "manifest.json")

    def publish_manifest(self, manifest: VersionManifest, version_path: str | Path | None = None) -> None:
        target = Path(version_path) if version_path is not None else self.version_dir(manifest.version)
        manifest.write(target / "manifest.json")
        self.write_latest(manifest.version)
