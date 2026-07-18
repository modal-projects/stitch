"""``ModalVolumeStore`` — the ``Store`` instance backed by a Modal Volume.

Each run's chain lives under ``<root>/<run_id>/weight_vNNNNNN/`` (run-less:
``<root>/weight_vNNNNNN/``) as HF-safetensors + delta metadata, and a ``latest``
text file holds the self-identifying pointer identity. Durability is an explicit
Volume commit; cross-host visibility is a reload. With ``volume_name=None`` it is a
plain local directory, so the class is exercisable without Modal.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path

from stitch.stores.base import Store
from stitch.types import VersionManifest, VersionRef

_POINTER = "latest"


class ModalVolumeStore(Store):
    def __init__(self, root: str | Path, *, volume_name: str | None = None) -> None:
        self.root = Path(root)
        self.volume_name = volume_name

    def refresh(self) -> None:
        if self.volume_name:
            _volume(self.volume_name).reload()

    def read_pointer(self) -> VersionRef | None:
        path = self.root / _POINTER
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8").strip()
        return VersionRef.parse(text) if text else None

    def advance_pointer(self, ref: VersionRef) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.root / _POINTER, ref.identity)
        if self.volume_name:
            _volume(self.volume_name).commit()

    def claim(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("claim requires a run_id (the run's per-launch epoch token)")
        self.advance_pointer(VersionRef(run_id, 0))

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
        self.refresh()
        return str(self._version_dir(ref))

    def commit(self) -> None:
        """Durably flush pending writes on this host (e.g. one trainer rank's shard of a
        version's files). Not part of the Store port — a Modal-Volume affordance the
        publish hook uses on non-writer ranks; a no-op without a backing volume."""
        if self.volume_name:
            _volume(self.volume_name).commit()

    def _version_dir(self, ref: VersionRef) -> Path:
        return self.root / ref.identity


def pull_weights_pre_read_hook(source_dir: str, target_version: int) -> None:
    """Engine-side ``--custom-pull-weights-pre-read-hook``: reload the delta Volume onto
    THIS host exactly once so the engine's pull can read the published version.

    One reload, not a loop: looping thrashes the Modal-v2 mount (turned ~3s delta pulls into
    100-500s and tripped the engine's 300s watchdog). Completeness is verified downstream —
    the engine size-checks each staged delta and fails fast so the sidecar retries with a
    fresh reload. Volume name comes from ``DELTA_VOLUME_NAME`` (set on the serving container)."""
    del source_dir  # unused: the reload is by volume name, not path
    volume_name = os.environ.get("DELTA_VOLUME_NAME", "")
    if not volume_name or target_version <= 0:
        return
    _volume(volume_name).reload()


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
