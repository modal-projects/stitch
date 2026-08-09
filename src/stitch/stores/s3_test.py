"""Tests for the S3-backed Store using an in-memory S3 client."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from stitch.publish import publish_version
from stitch.stores.s3 import S3Store
from stitch.types import VersionKind, VersionRef


class _FakeS3:
    """Dict-backed subset of the boto3 S3 client used by ``S3Store``."""

    class exceptions:  # noqa: N801 - mirrors boto3's client.exceptions namespace
        class NoSuchKey(Exception):
            pass

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.downloads: list[tuple[str, str]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        try:
            body = self.objects[(Bucket, Key)]
        except KeyError:
            raise self.exceptions.NoSuchKey from None
        return {"Body": io.BytesIO(body)}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        self.downloads.append((bucket, key))
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[(bucket, key)])

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        objects = self.objects

        class _Paginator:
            def paginate(self, *, Bucket: str, Prefix: str):
                return [
                    {
                        "Contents": [
                            {"Key": key, "Size": len(value)}
                            for (bucket, key), value in objects.items()
                            if bucket == Bucket and key.startswith(Prefix)
                        ]
                    }
                ]

        return _Paginator()


def _store(tmp_path: Path, client: _FakeS3 | None = None) -> S3Store:
    store = S3Store(
        "s3://bucket/experiments/run-x",
        cache_dir=tmp_path / "cache",
        run_id="run-x",
    )
    store._client = client or _FakeS3()
    return store


def _write_version(root: Path, ref: VersionRef, *, base: int | None = None) -> str:
    version_dir = root / Path(ref.identity).name
    version_dir.mkdir(parents=True)
    metadata: dict[str, object] = {"version": ref.version}
    if base is not None:
        metadata.update(
            {
                "base_version": base,
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "xxh3-128",
            }
        )
    (version_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "weight_map": {"weight": "model-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    (version_dir / "model-00001.safetensors").write_bytes(
        f"version-{ref.version}".encode()
    )
    return str(version_dir)


def test_pointer_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.read_pointer() is None
    store.claim(VersionRef("run-x", 0))
    assert store.read_pointer() == VersionRef("run-x", 0)
    store.advance_pointer(VersionRef("run-x", 3))
    assert store.read_pointer() == VersionRef("run-x", 3)
    assert store._client.objects[("bucket", "experiments/run-x/latest")] == (
        b"run-x/weight_v000003"
    )


def test_rejects_a_version_from_another_run(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="scoped to run 'run-x'"):
        store.advance_pointer(VersionRef("run-y", 1))


def test_publish_manifest_and_materialize(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = _write_version(tmp_path / "trainer", VersionRef("run-x", 1))

    ref = publish_version(store, None, source, run_id="run-x")

    assert ref == VersionRef("run-x", 1)
    assert store.read_manifest(ref).kind is VersionKind.FULL
    version_dir = Path(store.materialize(ref))
    assert version_dir == store.cache_dir / "updates" / "weight_v000001"
    assert (version_dir / "model-00001.safetensors").read_bytes() == b"version-1"
    assert (
        "bucket",
        "experiments/run-x/updates/weight_v000001/model-00001.safetensors",
    ) in store._client.objects


def test_materialize_downloads_the_chain_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for version in (1, 2):
        source = _write_version(
            tmp_path / "trainer",
            VersionRef("run-x", version),
            base=version - 1 if version > 1 else None,
        )
        publish_version(store, None, source, run_id="run-x")

    version_dir = Path(store.materialize(VersionRef("run-x", 2)))

    assert (version_dir.parent / "weight_v000001").is_dir()
    assert (version_dir / "model-00001.safetensors").read_bytes() == b"version-2"
    downloads = list(store._client.downloads)
    store.materialize(VersionRef("run-x", 2))
    assert store._client.downloads == downloads


def test_runs_sharing_a_bucket_have_independent_state(tmp_path: Path) -> None:
    client = _FakeS3()
    first = S3Store(
        "s3://bucket/experiments/run-a",
        cache_dir=tmp_path / "run-a",
        run_id="run-a",
    )
    second = S3Store(
        "s3://bucket/experiments/run-b",
        cache_dir=tmp_path / "run-b",
        run_id="run-b",
    )
    first._client = second._client = client

    first.claim(VersionRef("run-a", 0))
    second.claim(VersionRef("run-b", 0))
    first.advance_pointer(VersionRef("run-a", 1))

    assert first.read_pointer() == VersionRef("run-a", 1)
    assert second.read_pointer() == VersionRef("run-b", 0)


def test_rejects_an_object_key_that_escapes_the_cache(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store._client.objects[("bucket", "experiments/run-x/updates/../outside-cache")] = (
        b"unsafe"
    )

    with pytest.raises(ValueError, match="unsafe S3 object key"):
        store.materialize(VersionRef("run-x", 1))


@pytest.mark.parametrize("root", ["", "s3://", "https://bucket/prefix"])
def test_rejects_an_invalid_root(tmp_path: Path, root: str) -> None:
    with pytest.raises(ValueError, match="S3 root"):
        S3Store(root, cache_dir=tmp_path, run_id="run-x")


def test_rejects_an_empty_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id is required"):
        S3Store("s3://bucket/prefix", cache_dir=tmp_path, run_id="")
