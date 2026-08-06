"""Harness for the shared hook shims.

Runs without Modal/torch: the real ``_store`` builds a local dir (volume_name=None from
the temp root) and only ``_pool`` is faked. Run directly:
  PYTHONPATH=src:. python cookbook/common/hooks_test.py
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from cookbook.common import hooks
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.types import VersionRef


class _FakePool:
    def __init__(self) -> None:
        self.woke: list = []

    def discover_replicas(self):
        return ["http://r1"]

    def wake(self, replicas, ref):
        self.woke.append(ref)


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
        original_publish = hooks.publish_version
        original_commit = ModalVolumeStore.commit

        def publish_after_barrier(*args, **kwargs):
            events.append("publish")
            return original_publish(*args, **kwargs)

        def commit_before_barrier(store):
            events.append("commit")
            return original_commit(store)

        hooks._pool = lambda args: pool  # rank is None in tests -> treated as writer
        def gather(value):
            events.append("gather")
            return [value]

        hooks.process.dist_all_gather_object = gather
        hooks.publish_version = publish_after_barrier
        ModalVolumeStore.commit = commit_before_barrier
        try:
            vdir = _write_version(root, VersionRef("run-abc", 1))
            hooks.commit_and_wake(_args(str(root)), vdir)
        finally:
            hooks.process.dist_all_gather_object = original_gather
            hooks.publish_version = original_publish
            ModalVolumeStore.commit = original_commit
        assert events == ["gather", "commit", "gather", "publish", "gather"]
        assert ModalVolumeStore(root, run_id="run-abc").read_pointer() == VersionRef(
            "run-abc", 1
        )
        assert pool.woke == [VersionRef("run-abc", 1)]


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


def test_claim_pool_resets_to_base() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pool = _FakePool()
        hooks._pool = lambda args: pool
        hooks.claim_pool(_args(tmp))
        assert ModalVolumeStore(
            Path(tmp), run_id="run-abc"
        ).read_pointer() == VersionRef("run-abc", 0)
        assert pool.woke == [VersionRef("run-abc", 0)]


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
            rollout_session_affinity_header="Modal-Session-ID",
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


def test_request_hook_reads_local_pointer_without_refresh() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ModalVolumeStore(root, run_id="run-abc")
        store.advance_pointer(VersionRef("run-abc", 1))
        hooks._latest = hooks._CachedPointer()
        original_refresh = ModalVolumeStore.refresh

        def unexpected_refresh(_store) -> None:
            raise AssertionError(
                "the trainer-local request gate must not reload the Volume"
            )

        ModalVolumeStore.refresh = unexpected_refresh
        try:
            assert asyncio.run(hooks._latest.get(_args(str(root)), ttl=0)) == 1
            store.advance_pointer(VersionRef("run-abc", 2))
            assert asyncio.run(hooks._latest.get(_args(str(root)), ttl=0)) == 2
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
