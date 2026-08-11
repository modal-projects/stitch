"""Harness for the shared hook shims.

Runs without Modal/torch: the real ``_store`` builds a local dir (volume_name=None from
the temp root) and only ``_pool`` is faked. Run directly:
  PYTHONPATH=src:. python cookbook/common/hooks_test.py
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import tempfile
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbook.common import hooks
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.stores.s3 import S3Store, UploadReceipt
from stitch.types import VersionRef


class _FakePool:
    def __init__(self) -> None:
        self.woke: list = []

    def discover_replicas(self):
        return ["http://r1"]

    def wake(self, replicas, ref):
        self.woke.append(ref)


class _FakeS3:
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


def _args(root: str, run_id: str = "run-abc", **extra):
    return SimpleNamespace(
        update_weight_disk_dir=f"{root}/updates",
        run_id=run_id,
        **extra,
    )


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


def test_commit_and_wake_publishes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pool = _FakePool()
        events = []
        original_gather = hooks.process.dist_all_gather_object
        original_container_leader = hooks.process.dist_is_container_leader
        original_publish = hooks.publish_version
        original_commit = ModalVolumeStore.commit
        original_refresh = ModalVolumeStore.refresh

        def publish_after_refresh(*args, **kwargs):
            events.append("publish")
            return original_publish(*args, **kwargs)

        def commit_before_refresh(store):
            events.append("commit")
            return original_commit(store)

        def refresh_after_commit(store):
            events.append("refresh")
            return original_refresh(store)

        hooks._pool = lambda args: pool  # rank is None in tests -> treated as writer

        def gather(value):
            events.append("gather")
            return [value]

        hooks.process.dist_all_gather_object = gather
        hooks.process.dist_is_container_leader = lambda: True
        hooks.publish_version = publish_after_refresh
        ModalVolumeStore.commit = commit_before_refresh
        ModalVolumeStore.refresh = refresh_after_commit
        try:
            vdir = _write_version(root, VersionRef("run-abc", 1))
            hooks.commit_and_wake(_args(str(root)), vdir)
        finally:
            hooks.process.dist_all_gather_object = original_gather
            hooks.process.dist_is_container_leader = original_container_leader
            hooks.publish_version = original_publish
            ModalVolumeStore.commit = original_commit
            ModalVolumeStore.refresh = original_refresh
        assert events == [
            "gather",
            "commit",
            "gather",
            "refresh",
            "publish",
            "gather",
        ]
        assert ModalVolumeStore(root, run_id="run-abc").read_pointer() == VersionRef(
            "run-abc", 1
        )
        assert pool.woke == [VersionRef("run-abc", 1)]


def test_commit_and_wake_commits_once_per_container() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original_rank = hooks.process.dist_rank
        original_gather = hooks.process.dist_all_gather_object
        original_container_leader = hooks.process.dist_is_container_leader
        original_commit = ModalVolumeStore.commit

        hooks.process.dist_rank = lambda: 1

        def gather_with_rank_zero(value):
            if isinstance(value, tuple) and len(value) == 4:
                return [(True, False, None, None), value]
            return [value]

        hooks.process.dist_all_gather_object = gather_with_rank_zero
        hooks.process.dist_is_container_leader = lambda: False

        def unexpected_commit(_store) -> None:
            raise AssertionError("a non-leader rank must not commit the shared mount")

        ModalVolumeStore.commit = unexpected_commit
        try:
            vdir = _write_version(root, VersionRef("run-abc", 1))
            hooks.commit_and_wake(_args(str(root)), vdir)
        finally:
            hooks.process.dist_rank = original_rank
            hooks.process.dist_all_gather_object = original_gather
            hooks.process.dist_is_container_leader = original_container_leader
            ModalVolumeStore.commit = original_commit


def test_commit_and_wake_publishes_directly_to_s3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _FakeS3()
    pool = _FakePool()
    create_store = hooks.storage.create_store

    def create_store_with_client(*args, **kwargs):
        store = create_store(*args, **kwargs)
        if isinstance(store, S3Store):
            store._client = client
        return store

    monkeypatch.setattr(hooks.storage, "create_store", create_store_with_client)
    monkeypatch.setattr(hooks, "_pool", lambda _args: pool)
    monkeypatch.setattr(
        ModalVolumeStore,
        "commit",
        lambda _store: (_ for _ in ()).throw(
            AssertionError("S3 publication must not commit a Modal Volume")
        ),
    )
    args = _args(
        str(tmp_path),
        stitch_store_backend="s3",
        stitch_s3_root="s3://bucket/experiments/run-abc",
    )
    version_dir = _write_version(tmp_path, VersionRef("run-abc", 1))

    hooks.commit_and_wake(args, version_dir)

    assert hooks._store(args).read_pointer() == VersionRef("run-abc", 1)
    assert pool.woke == [VersionRef("run-abc", 1)]
    assert (
        "bucket",
        "experiments/run-abc/updates/weight_v000001/model-00001.safetensors",
    ) in client.objects


def test_commit_and_wake_verifies_receipts_from_multiple_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _FakeS3()
    ref = VersionRef("run-abc", 1)
    local_dir = tmp_path / "updates" / "weight_v000001"
    remote_dir = tmp_path / "remote" / "weight_v000001"
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
    remote_store = S3Store(
        "s3://bucket/experiments/run-abc",
        cache_dir=tmp_path,
        run_id="run-abc",
    )
    remote_store._client = client
    remote_receipt = remote_store.upload_version_files(ref, remote_dir)
    create_store = hooks.storage.create_store

    def create_store_with_client(*args, **kwargs):
        store = create_store(*args, **kwargs)
        if isinstance(store, S3Store):
            store._client = client
        return store

    def gather_with_remote_receipt(value):
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], UploadReceipt)
        ):
            return [value, (remote_receipt, None)]
        return [value]

    monkeypatch.setattr(hooks.storage, "create_store", create_store_with_client)
    monkeypatch.setattr(
        hooks.process, "dist_all_gather_object", gather_with_remote_receipt
    )
    monkeypatch.setattr(hooks.process, "dist_is_container_leader", lambda: True)
    monkeypatch.setattr(hooks, "_pool", lambda _args: _FakePool())
    args = _args(
        str(tmp_path),
        stitch_store_backend="s3",
        stitch_s3_root="s3://bucket/experiments/run-abc",
    )

    hooks.commit_and_wake(args, str(local_dir))

    assert hooks._store(args).read_pointer() == ref


def test_commit_and_wake_s3_baseline_does_not_touch_a_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _FakeS3()
    create_store = hooks.storage.create_store

    def create_store_with_client(*args, **kwargs):
        store = create_store(*args, **kwargs)
        if isinstance(store, S3Store):
            store._client = client
        return store

    monkeypatch.setattr(hooks.storage, "create_store", create_store_with_client)
    monkeypatch.setattr(
        ModalVolumeStore,
        "commit",
        lambda _store: (_ for _ in ()).throw(
            AssertionError("S3 baseline must not commit a Modal Volume")
        ),
    )
    updates = tmp_path / "updates"
    updates.mkdir()
    args = _args(
        str(tmp_path),
        stitch_store_backend="s3",
        stitch_s3_root="s3://bucket/experiments/run-abc",
    )

    hooks.commit_and_wake(args, str(updates))


def test_commit_and_wake_baseline_is_noop() -> None:
    # Regression: baseline commit hands the run dir (no index); keying on the dir name must no-op, not crash on a missing index.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pool = _FakePool()
        hooks._pool = lambda args: pool
        updates = root / "updates"
        updates.mkdir(parents=True, exist_ok=True)
        original_barrier = hooks.process.dist_barrier

        def unexpected_barrier() -> None:
            raise AssertionError("run-directory commit must not rendezvous")

        hooks.process.dist_barrier = unexpected_barrier
        try:
            hooks.commit_and_wake(_args(str(root)), str(updates))
        finally:
            hooks.process.dist_barrier = original_barrier
        assert ModalVolumeStore(root, run_id="run-abc").read_pointer() is None
        assert pool.woke == []


def test_commit_and_wake_does_not_mutate_an_already_published_version() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ModalVolumeStore(root, run_id="run-abc")
        store.advance_pointer(VersionRef("run-abc", 1))
        vdir = _write_version(root, VersionRef("run-abc", 1))
        original_commit = ModalVolumeStore.commit

        def unexpected_commit(_store) -> None:
            raise AssertionError("an immutable published version must not be rewritten")

        ModalVolumeStore.commit = unexpected_commit
        try:
            hooks.commit_and_wake(_args(str(root)), vdir)
        finally:
            ModalVolumeStore.commit = original_commit

        assert store.read_pointer() == VersionRef("run-abc", 1)


def test_claim_pool_claims_version_zero_for_fresh_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pool = _FakePool()
        hooks._pool = lambda args: pool
        hooks.claim_pool(_args(tmp))
        assert ModalVolumeStore(
            Path(tmp), run_id="run-abc"
        ).read_pointer() == VersionRef("run-abc", 0)
        assert pool.woke == [VersionRef("run-abc", 0)]


def test_claim_pool_preserves_resumed_boot_version() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pool = _FakePool()
        hooks._pool = lambda args: pool
        hooks.claim_pool(_args(tmp), boot_version=119)
        assert ModalVolumeStore(
            Path(tmp), run_id="run-abc"
        ).read_pointer() == VersionRef("run-abc", 119)
        assert pool.woke == [VersionRef("run-abc", 119)]


def test_store_rejects_a_non_updates_directory() -> None:
    with pytest.raises(ValueError, match="must end in /updates"):
        hooks._store(
            SimpleNamespace(
                update_weight_disk_dir="/stitch/run-abc/deltas",
                run_id="run-abc",
            )
        )


def test_request_hook_min_lag() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ModalVolumeStore(root, run_id="run-abc").advance_pointer(
            VersionRef("run-abc", 10)
        )  # published latest
        hooks._latest = hooks._CachedPointer()  # fresh cache reading this store
        args = _args(
            str(root),
            rollout_request_weight_version_lag=2,
            rollout_request_retry_attempts=900,
        )
        request = {"payload": {}}
        asyncio.run(
            hooks.gated_rollout_request_hook(
                args,
                SimpleNamespace(group_index=1, routing_key="sample-key"),
                request,
            )
        )
        assert request["payload"]["weight_version"] == {
            "min_version": 8,
            "exact_version": None,
        }
        assert request["headers"]["Modal-Session-ID"] == "group-1"
        assert request["max_retries"] == 900


def test_request_hook_reads_shared_mount_without_reload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ModalVolumeStore(root, run_id="run-abc")
        store.advance_pointer(VersionRef("run-abc", 1))
        pointer = hooks._CachedPointer()
        original_refresh = ModalVolumeStore.refresh

        def unexpected_refresh(_store) -> None:
            raise AssertionError(
                "the request gate must not reload the publisher's mount"
            )

        ModalVolumeStore.refresh = unexpected_refresh
        try:
            args = _args(str(root), experiment_volume_name="weights")
            assert asyncio.run(pointer.get(args, ttl=0)) == 1
            store.advance_pointer(VersionRef("run-abc", 2))
            assert asyncio.run(pointer.get(args, ttl=0)) == 2
        finally:
            ModalVolumeStore.refresh = original_refresh


def test_sample_affinity_key_fallbacks() -> None:
    assert hooks.sample_affinity_key(SimpleNamespace(group_index=7)) == "group-7"
    assert (
        hooks.sample_affinity_key(SimpleNamespace(routing_key="trajectory"))
        == "trajectory"
    )
    assert hooks.sample_affinity_key(SimpleNamespace(session_id="legacy")) == "legacy"
    assert hooks.sample_affinity_key(SimpleNamespace()) is None


def test_request_hook_exact_and_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ModalVolumeStore(root, run_id="run-abc").advance_pointer(
            VersionRef("run-abc", 10)
        )
        hooks._latest = hooks._CachedPointer()
        exact_req = {"payload": {}}
        asyncio.run(
            hooks.gated_rollout_request_hook(
                _args(
                    str(root),
                    rollout_request_weight_version_mode="exact",
                    rollout_request_weight_version_lag=1,
                ),
                SimpleNamespace(session_id=None),
                exact_req,
            )
        )
        assert exact_req["payload"]["weight_version"] == {
            "min_version": None,
            "exact_version": 9,
        }

        hooks._latest = hooks._CachedPointer()
        none_req = {"payload": {}}
        asyncio.run(
            hooks.gated_rollout_request_hook(
                _args(str(root), rollout_request_weight_version_mode="none"),
                SimpleNamespace(session_id=None),
                none_req,
            )
        )
        assert none_req["payload"]["weight_version"] == {
            "min_version": None,
            "exact_version": None,
        }


def test_request_hook_cache_switches_runs() -> None:
    with (
        tempfile.TemporaryDirectory() as first,
        tempfile.TemporaryDirectory() as second,
    ):
        ModalVolumeStore(first, run_id="run-a").advance_pointer(VersionRef("run-a", 7))
        ModalVolumeStore(second, run_id="run-b").advance_pointer(VersionRef("run-b", 3))
        hooks._latest = hooks._CachedPointer()

        async def read(root: str, run_id: str) -> dict:
            request = {"payload": {}}
            await hooks.gated_rollout_request_hook(
                _args(root, run_id=run_id, rollout_request_weight_version_lag=0),
                SimpleNamespace(session_id=None),
                request,
            )
            return request["payload"]["weight_version"]

        assert asyncio.run(read(first, "run-a")) == {
            "min_version": 7,
            "exact_version": None,
        }
        assert asyncio.run(read(second, "run-b")) == {
            "min_version": 3,
            "exact_version": None,
        }


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"common hooks harness: {len(tests)} PASS")
