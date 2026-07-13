"""Publish-side harness: publish_version / claim_run / constrain_request against fakes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from stitch.publish import claim_run, constrain_request, publish_version
from stitch.versions import PointerRewind, VersionRef


class FakeStore:
    def __init__(self, pointer: VersionRef | None = None) -> None:
        self._pointer = pointer
        self.published: list = []

    def read_pointer(self):
        return self._pointer

    def publish(self, manifest, files_dir):
        self.published.append(manifest)

    def advance_pointer(self, ref):
        self._pointer = ref

    def claim(self, run_id):
        self._pointer = VersionRef(run_id, 0)


class FakePool:
    def __init__(self) -> None:
        self.woke: list = []

    def discover_replicas(self):
        return ["http://r1"]

    def wake(self, replicas, ref):
        self.woke.append(ref)


def _version_dir(tmp: str, *, version: int, base: int | None = None, diff: str | None = None) -> str:
    d = Path(tmp) / f"weight_v{version:06d}"
    d.mkdir()
    meta: dict = {"version": version}
    if diff:
        meta.update({"diff": diff, "base_version": base, "compression": "zstd", "checksum": "xxh3-128"})
    index = {"metadata": meta, "weight_map": {"w": "model-00001.safetensors"}}
    (d / "model.safetensors.index.json").write_text(json.dumps(index))
    return str(d)


def test_publish_full() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store, pool = FakeStore(), FakePool()
        ref = publish_version(store, pool, _version_dir(tmp, version=1), run_id="r1")
        assert ref == VersionRef("r1", 1)
        assert store._pointer == VersionRef("r1", 1)
        man = store.published[0]
        assert man.kind.value == "full" and man.base_version is None
        assert pool.woke == [VersionRef("r1", 1)]


def test_publish_delta() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FakeStore(VersionRef("r1", 1))
        publish_version(store, None, _version_dir(tmp, version=2, base=1, diff="xor"), run_id="r1")
        man = store.published[0]
        assert man.kind.value == "delta" and man.base_version == 1 and man.diff == "xor"
        assert store._pointer == VersionRef("r1", 2)


def test_publish_rewind_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FakeStore(VersionRef("r1", 3))
        try:
            publish_version(store, None, _version_dir(tmp, version=1), run_id="r1")
            raise AssertionError("expected PointerRewind")
        except PointerRewind:
            pass


def test_claim_run() -> None:
    store, pool = FakeStore(), FakePool()
    claim_run(store, pool, "r2")
    assert store._pointer == VersionRef("r2", 0)
    assert pool.woke == [VersionRef("r2", 0)]


def test_constrain_lag() -> None:
    payload, headers = {}, {}
    constrain_request(payload, headers, latest=10, lag=2, session_id="g1", affinity_header="X-Session")
    assert payload["weight_version"] == {"min_version": 8, "exact_version": None}
    assert headers["X-Session"] == "g1"


def test_constrain_exact() -> None:
    payload, headers = {}, {}
    constrain_request(payload, headers, exact=5)
    assert payload["weight_version"] == {"min_version": None, "exact_version": 5}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"publish harness: {len(tests)} PASS")
