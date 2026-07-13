"""Harness for S3Store over an in-memory fake S3 (no boto3 / no network).

Exercises the pointer round-trip and the publish -> read_manifest -> materialize path the
reconciler drives, so the store's S3 key layout + download-to-cache logic is verified without
a bucket. Run directly: PYTHONPATH=src python tests/s3_store_test.py
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

from stitch.stores.s3 import S3Store
from stitch.versions import VersionKind, VersionManifest, VersionRef


class _FakeS3:
    """Dict-backed stand-in for a boto3 S3 client (only the calls S3Store makes)."""

    class exceptions:  # noqa: N801 — mirrors boto3 client.exceptions.NoSuchKey
        class NoSuchKey(Exception):
            pass

    def __init__(self) -> None:
        self.objs: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket, Key, Body):
        self.objs[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objs:
            raise self.exceptions.NoSuchKey()
        return {"Body": io.BytesIO(self.objs[(Bucket, Key)])}

    def upload_file(self, Filename, Bucket, Key):
        self.objs[(Bucket, Key)] = Path(Filename).read_bytes()

    def download_file(self, Bucket, Key, Filename):
        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        Path(Filename).write_bytes(self.objs[(Bucket, Key)])

    def get_paginator(self, _name):
        objs = self.objs

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                contents = [{"Key": k, "Size": len(v)} for (b, k), v in objs.items() if b == Bucket and k.startswith(Prefix)]
                return [{"Contents": contents}]

        return _Paginator()


def _store(tmp: str) -> S3Store:
    store = S3Store("s3://bkt/deltas", cache_dir=Path(tmp) / "cache")
    store._client = _FakeS3()  # inject the fake so _s3() never imports boto3
    return store


def _write_local_version(root: Path, ref: VersionRef) -> str:
    """A version dir as the trainer would write it locally (delta index + one shard)."""
    d = root / ref.identity
    d.mkdir(parents=True)
    (d / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"version": ref.version, "base_version": ref.version - 1, "diff": "xor",
                         "compression": "zstd", "checksum": "xxh3-128"},
            "weight_map": {"w": "model-00001-of-00001.safetensors"},
        })
    )
    (d / "model-00001-of-00001.safetensors").write_bytes(b"\x00\x11\x22\x33")
    return str(d)


def test_s3_pointer_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        assert store.read_pointer() is None
        store.claim("run-x")
        assert store.read_pointer() == VersionRef("run-x", 0)
        store.advance_pointer(VersionRef("run-x", 3))
        assert store.read_pointer() == VersionRef("run-x", 3)


def test_s3_publish_manifest_materialize() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        ref = VersionRef("run-x", 1)
        local = _write_local_version(Path(tmp) / "trainer", ref)  # trainer's local write
        store.publish(VersionManifest.from_hf_index(local, run_id="run-x"), local)

        manifest = store.read_manifest(ref)  # reads the index back from S3
        assert manifest.ref == ref
        assert manifest.kind is VersionKind.DELTA
        assert manifest.delta_encoding == "xor" and manifest.base_version == 0

        version_dir = Path(store.materialize(ref))  # downloads the chain into the cache
        assert version_dir == store.cache_dir / ref.identity
        assert (version_dir / "model-00001-of-00001.safetensors").read_bytes() == b"\x00\x11\x22\x33"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"s3 store harness: {len(tests)} PASS")
