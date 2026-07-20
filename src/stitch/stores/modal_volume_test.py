"""ModalVolumeStore harness, driven through the real ``publish_version`` flow.

Covers everything provable without Modal (volume_name=None → a local dir): the pointer
round-trip, the claim → delta-chain path, and the external-staging copy. The volume-backed
path is validated e2e."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from stitch.publish import publish_version
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.types import VersionKind, VersionRef


def _write_version(root: Path, ref: VersionRef, *, base: int | None = None, diff: str | None = None) -> str:
    d = root / ref.identity
    d.mkdir(parents=True)
    meta: dict = {"version": ref.version}
    if diff:
        meta.update({"delta_encoding": diff, "base_version": base, "compression_format": "zstd", "checksum_format": "xxh3-128"})
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": meta, "weight_map": {"w": "model-00001.safetensors"}})
    )
    (d / "model-00001.safetensors").write_bytes(b"\x00")
    return str(d)


def test_publish_full_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ModalVolumeStore(root)
        assert store.read_pointer() is None
        vdir = _write_version(root, VersionRef("r1", 1))  # framework wrote it in place
        ref = publish_version(store, None, vdir, run_id="r1")
        assert ref == VersionRef("r1", 1)
        assert store.read_pointer() == VersionRef("r1", 1)  # pointer parses back to the ref
        man = store.read_manifest(ref)
        assert man.kind is VersionKind.FULL
        assert (Path(store.materialize(ref)) / "model.safetensors.index.json").exists()


def test_claim_then_delta_chain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ModalVolumeStore(root)
        store.claim("r1")
        assert store.read_pointer() == VersionRef("r1", 0)  # base before any publish
        publish_version(store, None, _write_version(root, VersionRef("r1", 1)), run_id="r1")
        publish_version(store, None, _write_version(root, VersionRef("r1", 2), base=1, diff="xor"), run_id="r1")
        assert store.read_pointer() == VersionRef("r1", 2)
        man = store.read_manifest(VersionRef("r1", 2))
        assert man.kind is VersionKind.DELTA


def test_copies_when_files_dir_is_external() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, staging = Path(tmp) / "store", Path(tmp) / "staging"
        root.mkdir()
        store = ModalVolumeStore(root)
        src = _write_version(staging, VersionRef("r1", 1))  # a staging dir, not the store layout
        publish_version(store, None, src, run_id="r1")
        assert (root / "r1" / "weight_v000001" / "model.safetensors.index.json").exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"modal_volume harness: {len(tests)} PASS")
