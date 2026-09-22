"""Publisher harness: the distributed publication protocol against fake comms/stores.

Runs without Modal/torch: ranks are scripted in-process, mounted stores are local
dirs (volume_name=None), and S3 is a fake client. The multi-rank scripts emulate
what a real collective would return — every rank sees every gathered value.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from base64 import b64encode
from pathlib import Path

import pytest

from stitch import publisher
from stitch.publisher import Publisher, TrainerComms
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.stores.s3 import S3Store, UploadReceipt
from stitch.types import PointerRewind, VersionRef


class _FakePool:
    def __init__(self) -> None:
        self.woke: list = []

    def discover_replicas(self):
        return ["http://r1"]

    def wake(self, replicas, ref):
        self.woke.append(ref)


class _RecordingComms(TrainerComms):
    """Scriptable in-process comms: ``gather`` emulates the collective's result."""

    def __init__(
        self,
        events: list[str] | None = None,
        rank: int | None = None,
        is_host_leader: bool = True,
        gather=None,
    ) -> None:
        self._events = events
        self._rank = rank
        self._leader = is_host_leader
        self._gather = gather or (lambda value: [value])

    def rank(self):
        return self._rank

    def all_gather_object(self, value):
        if self._events is not None:
            self._events.append("gather")
        return self._gather(value)

    def is_host_leader(self):
        return self._leader


class _RecordingVolumeStore(ModalVolumeStore):
    def __init__(self, events: list[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._events = events

    def commit(self) -> None:
        self._events.append("commit")
        super().commit()

    def refresh(self) -> None:
        self._events.append("refresh")
        super().refresh()


class _FakeS3:
    """Dict-backed subset of the boto3 S3 client used by ``S3Store``."""

    class exceptions:  # noqa: N801 - mirrors boto3's client.exceptions namespace
        class NoSuchKey(Exception):
            pass

    class ConditionalWriteFailed(Exception):
        def __init__(self) -> None:
            self.response = {
                "Error": {"Code": "PreconditionFailed"},
                "ResponseMetadata": {"HTTPStatusCode": 412},
            }

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.checksums: dict[tuple[str, str], str] = {}
        self.etags: dict[tuple[str, str], str] = {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        IfMatch: str | None = None,
        IfNoneMatch: str | None = None,
    ) -> None:
        object_id = (Bucket, Key)
        if IfNoneMatch == "*" and object_id in self.objects:
            raise self.ConditionalWriteFailed
        if IfMatch is not None and self.etags.get(object_id) != IfMatch:
            raise self.ConditionalWriteFailed
        self._write_object(object_id, Body)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        try:
            value = self.objects[(Bucket, Key)]
        except KeyError:
            raise self.exceptions.NoSuchKey from None
        return {"Body": io.BytesIO(value), "ETag": self.etags[(Bucket, Key)]}

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, object] | None = None,
    ) -> None:
        object_id = (bucket, key)
        body = Path(filename).read_bytes()
        self._write_object(object_id, body)
        self.metadata[object_id] = dict((ExtraArgs or {}).get("Metadata", {}))
        if (ExtraArgs or {}).get("ChecksumAlgorithm") == "SHA256":
            self.checksums[object_id] = b64encode(
                hashlib.sha256(body).digest()
            ).decode()

    def head_object(
        self, *, Bucket: str, Key: str, ChecksumMode: str | None = None
    ) -> dict[str, object]:
        object_id = (Bucket, Key)
        if object_id not in self.objects:
            raise self.exceptions.NoSuchKey
        return {
            "ContentLength": len(self.objects[object_id]),
            "ETag": self.etags[object_id],
            "Metadata": self.metadata.get(object_id, {}),
            **(
                {"ChecksumSHA256": self.checksums[object_id]}
                if ChecksumMode == "ENABLED" and object_id in self.checksums
                else {}
            ),
        }

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[(bucket, key)])

    def _write_object(self, object_id: tuple[str, str], body: bytes) -> None:
        self.objects[object_id] = body
        self.metadata.pop(object_id, None)
        self.checksums.pop(object_id, None)
        self.etags[object_id] = f'"{hashlib.sha256(body).hexdigest()[:32]}"'


def _write_version(root: Path, ref: VersionRef) -> str:
    d = root / "updates" / Path(ref.identity).name
    d.mkdir(parents=True)
    (d / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"version": ref.version},
                "weight_map": {"w": "model-00001.safetensors"},
            }
        )
    )
    (d / "model-00001.safetensors").write_bytes(b"weights")
    return str(d)


def _s3_store(tmp: Path, client: _FakeS3) -> S3Store:
    store = S3Store("s3://bucket/experiments/run-abc", cache_dir=tmp, run_id="run-abc")
    store._client = client
    return store


def test_claim_starts_run_at_boot_version() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ModalVolumeStore(Path(tmp), run_id="run-abc")
        pool = _FakePool()
        Publisher(store, pool, run_id="run-abc").claim(boot_version=119)
        assert store.read_pointer() == VersionRef("run-abc", 119)
        assert pool.woke == [VersionRef("run-abc", 119)]


def test_claim_is_rank_zero_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ModalVolumeStore(Path(tmp), run_id="run-abc")
        pool = _FakePool()
        comms = _RecordingComms(rank=7)
        Publisher(store, pool, run_id="run-abc", comms=comms).claim()
        assert store.read_pointer() is None
        assert pool.woke == []


def test_publish_mounted_orders_commit_refresh_publish() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        events: list[str] = []
        store = _RecordingVolumeStore(events, root, run_id="run-abc")
        pool = _FakePool()
        comms = _RecordingComms(events=events)
        original_publish = publisher.publish_version

        def record_publish(store, pool, version_dir, *, run_id):
            events.append("publish")
            return original_publish(store, pool, version_dir, run_id=run_id)

        publisher.publish_version = record_publish
        try:
            vdir = _write_version(root, VersionRef("run-abc", 1))
            Publisher(store, pool, run_id="run-abc", comms=comms).publish(vdir)
        finally:
            publisher.publish_version = original_publish

        assert events == ["gather", "commit", "gather", "refresh", "publish", "gather"]
        assert store.read_pointer() == VersionRef("run-abc", 1)
        assert pool.woke == [VersionRef("run-abc", 1)]


def test_publish_mounted_commits_once_per_mount() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        events: list[str] = []
        store = _RecordingVolumeStore(events, root, run_id="run-abc")

        def gather_with_rank_zero(value):
            if isinstance(value, tuple) and len(value) == 4:
                return [(True, False, None, None), value]
            return [value]

        comms = _RecordingComms(
            events=events, rank=1, is_host_leader=False, gather=gather_with_rank_zero
        )
        vdir = _write_version(root, VersionRef("run-abc", 1))
        Publisher(store, _FakePool(), run_id="run-abc", comms=comms).publish(vdir)

        assert "commit" not in events
        assert events.count("gather") == 3  # snapshot, commit errors, publish errors
        assert store.read_pointer() is None  # only rank 0 publishes


def test_publish_mounted_leaves_an_already_published_version_immutable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        events: list[str] = []
        store = _RecordingVolumeStore(events, root, run_id="run-abc")
        store.advance_pointer(VersionRef("run-abc", 1))
        vdir = _write_version(root, VersionRef("run-abc", 1))

        Publisher(store, _FakePool(), run_id="run-abc").publish(vdir)

        assert "commit" not in events
        assert store.read_pointer() == VersionRef("run-abc", 1)


def test_publish_rejects_a_rewind_on_every_rank() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        events: list[str] = []
        store = _RecordingVolumeStore(events, root, run_id="run-abc")

        def gather_with_rank_zero(value):
            if isinstance(value, tuple) and len(value) == 4:
                return [(True, False, VersionRef("run-abc", 5), None), value]
            return [value]

        comms = _RecordingComms(events=events, rank=7, gather=gather_with_rank_zero)
        vdir = _write_version(root, VersionRef("run-abc", 3))

        with pytest.raises(PointerRewind):
            Publisher(store, _FakePool(), run_id="run-abc", comms=comms).publish(vdir)
        assert "commit" not in events


def test_publish_baseline_commits_a_mounted_store_without_publishing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        events: list[str] = []
        store = _RecordingVolumeStore(events, root, run_id="run-abc")
        pool = _FakePool()
        updates = root / "updates"
        updates.mkdir()

        Publisher(store, pool, run_id="run-abc").publish(str(updates))

        assert events == ["commit"]  # a durability boundary, not a version
        assert store.read_pointer() is None
        assert pool.woke == []


def test_publish_baseline_is_a_noop_for_an_upload_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        client = _FakeS3()
        store = _s3_store(Path(tmp), client)
        pool = _FakePool()
        updates = Path(tmp) / "updates"
        updates.mkdir()

        Publisher(store, pool, run_id="run-abc").publish(str(updates))

        assert client.objects == {}
        assert pool.woke == []


def test_publish_uploads_directly_to_s3() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        client = _FakeS3()
        store = _s3_store(Path(tmp), client)
        pool = _FakePool()
        vdir = _write_version(Path(tmp), VersionRef("run-abc", 1))

        Publisher(store, pool, run_id="run-abc").publish(vdir)

        assert store.read_pointer() == VersionRef("run-abc", 1)
        assert pool.woke == [VersionRef("run-abc", 1)]
        assert (
            "bucket",
            "experiments/run-abc/updates/weight_v000001/model-00001.safetensors",
        ) in client.objects


def test_publish_uploaded_verifies_receipts_from_multiple_hosts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        client = _FakeS3()
        ref = VersionRef("run-abc", 1)
        root = Path(tmp)
        local_dir = root / "updates" / "weight_v000001"
        remote_dir = root / "remote" / "weight_v000001"
        local_dir.mkdir(parents=True)
        remote_dir.mkdir(parents=True)
        (local_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"version": 1},
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    },
                }
            )
        )
        (local_dir / "model-00001-of-00002.safetensors").write_bytes(b"local")
        (remote_dir / "model-00002-of-00002.safetensors").write_bytes(b"remote")
        remote_receipt = _s3_store(root / "remote-cache", client).upload_version_files(
            ref, remote_dir
        )

        def gather_with_remote(value):
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], UploadReceipt)
            ):
                return [value, (remote_receipt, None)]
            return [value]

        store = _s3_store(root, client)
        comms = _RecordingComms(gather=gather_with_remote)
        Publisher(store, _FakePool(), run_id="run-abc", comms=comms).publish(
            str(local_dir)
        )

        assert store.read_pointer() == ref


def test_publish_upload_fails_when_any_rank_errors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        client = _FakeS3()
        store = _s3_store(Path(tmp), client)

        def gather_with_failure(value):
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], UploadReceipt)
            ):
                return [value, (None, "rank 3: connection reset")]
            return [value]

        vdir = _write_version(Path(tmp), VersionRef("run-abc", 1))
        comms = _RecordingComms(gather=gather_with_failure)

        with pytest.raises(RuntimeError, match="checkpoint upload failed"):
            Publisher(store, _FakePool(), run_id="run-abc", comms=comms).publish(vdir)
        assert store.read_pointer() is None  # latest never points at a partial upload


def test_pointer_snapshot_demands_exactly_one_rank_zero() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ModalVolumeStore(Path(tmp), run_id="run-abc")

        def gather_with_two_rank_zeros(value):
            if isinstance(value, tuple) and len(value) == 4:
                return [(True, False, None, None), (True, False, None, None)]
            return [value]

        vdir = _write_version(Path(tmp), VersionRef("run-abc", 1))
        comms = _RecordingComms(gather=gather_with_two_rank_zeros)

        with pytest.raises(RuntimeError, match="exactly one rank 0"):
            Publisher(store, None, run_id="run-abc", comms=comms).publish(vdir)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"publisher harness: {len(tests)} PASS")
