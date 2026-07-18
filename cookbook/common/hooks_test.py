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
    # transport root = parent of update_weight_disk_dir, so the Store roots at `root`.
    return SimpleNamespace(update_weight_disk_dir=f"{root}/{run_id}", run_id=run_id, **extra)


def _write_version(root: Path, ref: VersionRef) -> str:
    d = root / ref.identity
    d.mkdir(parents=True)
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"version": ref.version}, "weight_map": {"w": "model-00001.safetensors"}})
    )
    return str(d)


def test_commit_and_wake_publishes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pool = _FakePool()
        hooks._pool = lambda args: pool  # rank is None in tests -> treated as writer
        vdir = _write_version(root, VersionRef("run-abc", 1))
        hooks.commit_and_wake(_args(str(root)), vdir)
        assert ModalVolumeStore(root).read_pointer() == VersionRef("run-abc", 1)
        assert pool.woke == [VersionRef("run-abc", 1)]


def test_commit_and_wake_baseline_is_noop() -> None:
    # The framework fires the hook at baseline/pointer-commit with the RUN dir (name is the
    # run_id, no version written). Keying on the dir name means we flush the volume but read
    # no index and publish nothing — the regression was reading an index that wasn't there.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pool = _FakePool()
        hooks._pool = lambda args: pool
        run_dir = root / "run-abc"
        run_dir.mkdir(parents=True)
        hooks.commit_and_wake(_args(str(root)), str(run_dir))
        assert ModalVolumeStore(root).read_pointer() is None
        assert pool.woke == []


def test_claim_pool_resets_to_base() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pool = _FakePool()
        hooks._pool = lambda args: pool
        hooks.claim_pool(_args(tmp))
        assert ModalVolumeStore(Path(tmp)).read_pointer() == VersionRef("run-abc", 0)
        assert pool.woke == [VersionRef("run-abc", 0)]


def test_request_hook_min_lag() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ModalVolumeStore(root).advance_pointer(VersionRef("run-abc", 10))  # published latest
        hooks._latest = hooks._CachedPointer()  # fresh cache reading this store
        args = _args(str(root), rollout_request_weight_version_lag=2,
                     rollout_request_retry_attempts=900, rollout_session_affinity_header="Modal-Session-ID")
        request = {"payload": {}}
        asyncio.run(hooks.gated_rollout_request_hook(args, SimpleNamespace(session_id="grp-1"), request))
        assert request["payload"]["weight_version"] == {"min_version": 8, "exact_version": None}
        assert request["headers"]["Modal-Session-ID"] == "grp-1"
        assert request["max_retries"] == 900


def test_request_hook_exact_and_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ModalVolumeStore(root).advance_pointer(VersionRef("run-abc", 10))
        hooks._latest = hooks._CachedPointer()
        exact_req = {"payload": {}}
        asyncio.run(hooks.gated_rollout_request_hook(
            _args(str(root), rollout_request_weight_version_mode="exact", rollout_request_weight_version_lag=1),
            SimpleNamespace(session_id=None), exact_req))
        assert exact_req["payload"]["weight_version"] == {"min_version": None, "exact_version": 9}

        hooks._latest = hooks._CachedPointer()
        none_req = {"payload": {}}
        asyncio.run(hooks.gated_rollout_request_hook(
            _args(str(root), rollout_request_weight_version_mode="none"),
            SimpleNamespace(session_id=None), none_req))
        assert none_req["payload"]["weight_version"] == {"min_version": None, "exact_version": None}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"common hooks harness: {len(tests)} PASS")
