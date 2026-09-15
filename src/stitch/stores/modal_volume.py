"""``ModalVolumeStore`` — the ``Store`` instance backed by a Modal Volume.

``root`` is one run's directory. The training framework owns
``<root>/updates/`` and may recreate it while initializing; Stitch owns the
self-identifying ``<root>/latest`` commit pointer. Durability is an explicit
Volume commit and cross-host visibility is a reload.

``volume_path`` is the same root relative to the Volume. When supplied, pointer
reads and publication verification use its API without reloading open files.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path, PurePosixPath

from stitch.stores.base import Store
from stitch.types import VersionManifest, VersionRef

_POINTER = "latest"


class ModalVolumeStore(Store):
    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        volume_name: str | None = None,
        volume_path: str | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        self.root = Path(root)
        self.volume_name = volume_name
        self.run_id = run_id
        self.volume_path = (
            PurePosixPath(volume_path) if volume_path is not None else None
        )
        if self.volume_path is not None and (
            not volume_name
            or self.volume_path.is_absolute()
            or ".." in self.volume_path.parts
        ):
            raise ValueError(
                "volume_path requires a Volume name and a relative path without '..'"
            )

    def refresh(self) -> None:
        if self.volume_name:
            _volume(self.volume_name).reload()

    def read_pointer(self) -> VersionRef | None:
        try:
            if self.volume_path is not None:
                text = b"".join(
                    _volume(self.volume_name).read_file(
                        str(self.volume_path / _POINTER)
                    )
                ).decode()
            else:
                text = (self.root / _POINTER).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        text = text.strip()
        return VersionRef.parse(text) if text else None

    def verify_committed_version(
        self, ref: VersionRef, files_dir: str
    ) -> VersionManifest | None:
        """Verify an in-place version through the API without reloading open files.

        Return None for an external staging directory or an unconfigured API path;
        those publications use the ordinary mounted-source copy and validation.
        Every writer host must have committed before this method is called.
        """
        directory = self._version_dir(ref)
        if self.volume_path is None or Path(files_dir).resolve() != directory.resolve():
            return None
        from modal.types import FileEntryType

        volume = _volume(self.volume_name)
        remote = self.volume_path / "updates" / directory.name
        index_name = "model.safetensors.index.json"
        persisted = b"".join(volume.read_file(str(remote / index_name)))
        if persisted != (directory / index_name).read_bytes():
            raise ValueError("Committed index differs from the completed local index")
        manifest = VersionManifest.from_hf_index(directory, run_id=self.run_id)
        if manifest.ref != ref:
            raise ValueError(
                f"Checkpoint index identifies {manifest.ref.identity}, expected {ref.identity}"
            )
        for name in manifest.files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
                raise ValueError(f"Invalid checkpoint shard path: {name!r}")
        files = {
            str(PurePosixPath(entry.path.lstrip("/")).relative_to(remote))
            for entry in volume.iterdir(str(remote), recursive=True)
            if entry.type == FileEntryType.FILE
        }
        missing = sorted(set(manifest.files) - files)
        if missing:
            raise FileNotFoundError(
                "Incomplete committed version: missing " + ", ".join(missing)
            )
        return manifest

    def advance_pointer(self, ref: VersionRef) -> None:
        if ref.run_id != self.run_id:
            raise ValueError(
                f"store is scoped to run {self.run_id!r}, got {ref.run_id!r}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.root / _POINTER, ref.identity)
        if self.volume_name:
            _volume(self.volume_name).commit()

    def claim(self, boot: VersionRef) -> None:
        if not boot.run_id:
            raise ValueError(
                "claim requires a run_id (the run's per-launch epoch token)"
            )
        self.advance_pointer(boot)

    def read_manifest(self, ref: VersionRef) -> VersionManifest:
        return VersionManifest.from_hf_index(self._version_dir(ref), run_id=ref.run_id)

    def publish(self, manifest: VersionManifest, files_dir: str) -> None:
        # The framework usually writes straight into the volume, so copy only when files_dir
        # isn't the version dir already.
        target = self._version_dir(manifest.ref)
        source = Path(files_dir)
        if source.resolve() != target.resolve():
            import shutil

            shutil.copytree(source, target, dirs_exist_ok=True)
        if self.volume_name:
            _volume(self.volume_name).commit()

    def materialize(self, ref: VersionRef) -> str:
        # The reconciler refreshed the mount before reading the pointer and manifest.
        # Repeating the Volume reload here adds no visibility and can serialize I/O.
        return str(self._version_dir(ref))

    def commit(self) -> None:
        """Durably flush pending writes on this host (e.g. one trainer rank's shard of a
        version's files). Not part of the Store port — a Modal-Volume affordance the
        publish hook uses on non-writer ranks; a no-op without a backing volume."""
        if self.volume_name:
            _volume(self.volume_name).commit()

    def _version_dir(self, ref: VersionRef) -> Path:
        if ref.run_id != self.run_id:
            raise ValueError(
                f"store is scoped to run {self.run_id!r}, got {ref.run_id!r}"
            )
        return self.root / "updates" / Path(ref.identity).name


def _volume(name: str):
    import modal

    return modal.Volume.from_name(name, version=2, create_if_missing=True)


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())  # durable before the rename (stitch#30)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
