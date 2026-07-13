"""Instance harness: the real ModalVolumeStore / SGLangEngine / ModalFlashPool.

Covers everything provable without Modal/sglang/GPU — port conformance, the store
driven through the real ``publish_version`` flow, and the engine's request/response
version stamping. The HTTP and volume-backed paths are validated e2e."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from stitch.engines.base import Engine
from stitch.engines.sglang import SGLangEngine
from stitch.pools.base import Pool
from stitch.pools.modal_flash import ModalFlashPool
from stitch.publish import publish_version
from stitch.stores.base import Store
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.versions import VersionKind, VersionRef


def _write_version(root: Path, ref: VersionRef, *, base: int | None = None, diff: str | None = None) -> str:
    d = root / ref.identity
    d.mkdir(parents=True)
    meta: dict = {"version": ref.version}
    if diff:
        meta.update({"diff": diff, "base_version": base, "compression": "zstd", "checksum": "xxh3-128"})
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": meta, "weight_map": {"w": "model-00001.safetensors"}})
    )
    (d / "model-00001.safetensors").write_bytes(b"\x00")
    return str(d)


# ── port conformance ──────────────────────────────────────────────────────────
def test_instances_satisfy_their_ports() -> None:
    assert isinstance(ModalVolumeStore("/tmp/store"), Store)
    assert isinstance(SGLangEngine("http://engine", "/ckpt"), Engine)
    assert isinstance(ModalFlashPool("app", "Server"), Pool)


# ── ModalVolumeStore, driven through the real publish_version ───────────────────
def test_store_publish_full_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ModalVolumeStore(root)
        assert store.read_pointer() is None
        vdir = _write_version(root, VersionRef("r1", 1))  # framework wrote it in place
        ref = publish_version(store, None, vdir, run_id="r1")
        assert ref == VersionRef("r1", 1)
        assert store.read_pointer() == VersionRef("r1", 1)  # pointer parses back to the ref
        man = store.read_manifest(ref)
        assert man.kind is VersionKind.FULL and man.base_version is None
        assert (Path(store.open_version(ref)) / "model.safetensors.index.json").exists()


def test_store_claim_then_delta_chain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ModalVolumeStore(root)
        store.claim("r1")
        assert store.read_pointer() == VersionRef("r1", 0)  # base before any publish
        publish_version(store, None, _write_version(root, VersionRef("r1", 1)), run_id="r1")
        publish_version(store, None, _write_version(root, VersionRef("r1", 2), base=1, diff="xor"), run_id="r1")
        assert store.read_pointer() == VersionRef("r1", 2)
        man = store.read_manifest(VersionRef("r1", 2))
        assert man.kind is VersionKind.DELTA and man.base_version == 1 and man.diff == "xor"


def test_store_copies_when_files_dir_is_external() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, staging = Path(tmp) / "store", Path(tmp) / "staging"
        root.mkdir()
        store = ModalVolumeStore(root)
        src = _write_version(staging, VersionRef("r1", 1))  # a staging dir, not the store layout
        publish_version(store, None, src, run_id="r1")
        assert (root / "r1" / "weight_v000001" / "model.safetensors.index.json").exists()


# ── SGLangEngine version stamping (pure dict mutation) ──────────────────────────
def test_engine_stamp_request_namespaces_by_version() -> None:
    engine = SGLangEngine("http://engine", "/ckpt")
    req: dict = {"text": "hi"}
    engine.stamp_request(req, VersionRef("r1", 7))
    assert req["extra_key"] == "wv7;r1/"  # version + run namespace, no user key
    listed: dict = {"extra_key": ["a", "b"]}
    engine.stamp_request(listed, VersionRef(None, 3))
    assert listed["extra_key"] == ["wv3;a", "wv3;b"]  # run-less, per-element


def test_engine_stamp_response_generate_vs_openai() -> None:
    engine = SGLangEngine("http://engine", "/ckpt")
    gen: dict = {"text": "x", "meta_info": {}}
    engine.stamp_response(gen, VersionRef("r1", 4), VersionRef("r1", 5))
    assert gen["meta_info"] == {"weight_version": "4", "weight_version_start": 4, "weight_version_end": 5}
    openai: dict = {"choices": []}
    engine.stamp_response(openai, VersionRef("r1", 4), VersionRef("r1", 4))
    assert openai["weight_version_start"] == 4 and openai["weight_version_end"] == 4
    assert "meta_info" not in openai and "weight_version" not in openai


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"instances harness: {len(tests)} PASS")
