"""``ModalVolumeStore`` — the ``Store`` instance backed by a Modal Volume.

``root`` is one run's directory. The training framework owns
``<root>/updates/`` and may recreate it while initializing; Stitch owns the
self-identifying ``<root>/latest`` commit pointer. Checkpoint bytes become
durable through a mounted-Volume commit; the small pointer uses the Volume API's
transactional upload so readers observe either the old value or the new one.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from io import BytesIO
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
        volume_root: str | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        if (volume_name is None) != (volume_root is None):
            raise ValueError("volume_name and volume_root must be configured together")
        self.root = Path(root)
        self.volume_name = volume_name
        self.volume_root = (
            PurePosixPath(volume_root) if volume_root is not None else None
        )
        if self.volume_root is not None and (
            self.volume_root.is_absolute()
            or self.volume_root == PurePosixPath(".")
            or ".." in self.volume_root.parts
        ):
            raise ValueError("volume_root must be a non-empty relative path without '..'")
        self.run_id = run_id

    def refresh(self) -> None:
        if self.volume_name:
            _volume(self.volume_name).reload()

    def read_pointer(self) -> VersionRef | None:
        try:
            if self.volume_root is not None:
                text = b"".join(
                    _volume(self.volume_name).read_file(
                        str(self.volume_root / _POINTER)
                    )
                ).decode("utf-8")
            else:
                text = (self.root / _POINTER).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        text = text.strip()
        return VersionRef.parse(text) if text else None

    def advance_pointer(self, ref: VersionRef) -> None:
        if ref.run_id != self.run_id:
            raise ValueError(
                f"store is scoped to run {self.run_id!r}, got {ref.run_id!r}"
            )
        if self.volume_root is not None:
            # A mounted rename followed by commit can expose a zero-filled file
            # while the new blocks become visible. The upload transaction swaps
            # this control-plane value as one durable object.
            with _volume(self.volume_name).batch_upload(force=True) as upload:
                upload.put_file(
                    BytesIO(ref.identity.encode("utf-8")),
                    str(self.volume_root / _POINTER),
                )
            return
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.root / _POINTER, ref.identity)

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
