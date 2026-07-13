"""In-memory core harness (the Phase-1 gate): the real Reconciler + AdmissionGate
against fake Store / Engine — no Modal, sglang, or GPU. Runnable directly
(``python tests/reconcile_test.py``) or under pytest."""

from __future__ import annotations

import asyncio

from stitch.engines.base import Engine
from stitch.stores.base import Store
from stitch.sync import ConstraintUnmet, Reconciler
from stitch.versions import (
    VersionConstraint,
    VersionKind,
    VersionManifest,
    VersionRef,
    SyncState,
)


class FakeStore(Store):
    def __init__(self, pointer: VersionRef | None = None, *manifests: VersionManifest) -> None:
        self._pointer = pointer
        self._manifests = {(m.ref.run_id, m.ref.version): m for m in manifests}
        self.refreshed = 0

    def refresh(self) -> None:
        self.refreshed += 1

    def read_pointer(self) -> VersionRef | None:
        return self._pointer

    def read_manifest(self, ref: VersionRef) -> VersionManifest:
        return self._manifests[(ref.run_id, ref.version)]

    def materialize(self, ref: VersionRef) -> str:
        return f"/fake/{ref.identity}"

    def advance_pointer(self, ref: VersionRef) -> None:
        self._pointer = ref

    def claim(self, run_id: str) -> None:
        self._pointer = VersionRef(run_id, 0)

    def publish(self, manifest: VersionManifest, files_dir: str) -> None:
        self._manifests[(manifest.ref.run_id, manifest.ref.version)] = manifest


class FakeEngine(Engine):
    def __init__(self) -> None:
        self.calls: list[str] = []          # ordered pause/flush/commit/reset/resume + stage
        self.staged: list[VersionRef] = []
        self.committed: list[VersionRef] = []

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        self.staged.append(manifest.ref)
        self.calls.append(f"stage:{manifest.ref.version}")

    async def commit(self, ref: VersionRef) -> None:
        self.committed.append(ref)
        self.calls.append(f"commit:{ref.version}")

    async def flush(self) -> None:
        self.calls.append("flush")

    async def pause(self) -> None:
        self.calls.append("pause")

    async def resume(self) -> None:
        self.calls.append("resume")

    async def reset(self) -> None:
        self.calls.append("reset")

    def stamp_request(self, request, served) -> None:
        pass

    def stamp_response(self, response, served, current) -> None:
        pass

    def base_url(self) -> str:
        return "http://engine"


def _full(run: str, version: int) -> VersionManifest:
    return VersionManifest(VersionRef(run, version), VersionKind.FULL, ["model.safetensors"])


def _delta(run: str, version: int, *, files: list[str]) -> VersionManifest:
    return VersionManifest(
        VersionRef(run, version), VersionKind.DELTA, files, base_version=version - 1, delta_encoding="xor"
    )


def _run(coro) -> None:
    asyncio.run(coro)


# ── reconcile ────────────────────────────────────────────────────────────────
def test_fresh_reconcile() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(store=FakeStore(VersionRef("r1", 3), _full("r1", 3)), engine=engine, commit_mode="quiesce")
        await r.startup()
        assert r.applied == VersionRef("r1", 3)
        assert engine.staged[-1] == VersionRef("r1", 3)
        assert VersionRef("r1", 3) in engine.committed
        assert r.sync_state is SyncState.IDLE
        assert engine.calls.index("flush") < engine.calls.index("commit:3")  # quiesce flushes first

    _run(go())


def test_catch_up() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(store=FakeStore(VersionRef("r1", 5), _full("r1", 5)), engine=engine)  # default commit_mode
        r.applied = VersionRef("r1", 3)
        await r.reconcile()
        assert r.applied == VersionRef("r1", 5)
        assert engine.committed == [VersionRef("r1", 5)]  # one composed stage+reload, not per-version
        assert "flush" not in engine.calls  # in_place is the default: no drain/flush, pause+reload+resume

    _run(go())


def test_run_switch_resets_in_place() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(store=FakeStore(VersionRef("r2", 2), _full("r2", 2)), engine=engine, commit_mode="in_place")
        r.applied = VersionRef("r1", 5)
        await r.reconcile()
        assert r.applied == VersionRef("r2", 2)
        assert "reset" in engine.calls  # was patched -> reseed base for the new run
        assert engine.calls.index("pause") < engine.calls.index("reset") < engine.calls.index("resume")
        assert "flush" not in engine.calls  # in_place never flushes

    _run(go())


def test_empty_delta_skips_reload() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(store=FakeStore(VersionRef("r1", 4), _delta("r1", 4, files=[])), engine=engine)
        r.applied = VersionRef("r1", 3)
        await r.reconcile()
        assert r.applied == VersionRef("r1", 4)
        assert engine.staged == [VersionRef("r1", 4)]
        assert engine.committed == []  # no reload for a zero-file delta
        assert r.metrics.get("skipped_reload") is True

    _run(go())


# ── admission gate ───────────────────────────────────────────────────────────
def test_admit_satisfied() -> None:
    async def go() -> None:
        r = Reconciler(store=FakeStore(), engine=FakeEngine())
        r.applied = VersionRef("r1", 5)
        async with r.admit(VersionConstraint(min_version=3)) as served:
            assert served == VersionRef("r1", 5)

    _run(go())


def test_admit_rejected_triggers_wake() -> None:
    async def go() -> None:
        r = Reconciler(store=FakeStore(VersionRef("r1", 5), _full("r1", 5)), engine=FakeEngine())
        r.applied = VersionRef("r1", 2)
        try:
            async with r.admit(VersionConstraint(min_version=5)):
                raise AssertionError("should have rejected")
        except ConstraintUnmet as e:
            assert e.error["type"] == "WeightVersionNotReady"
            assert e.error["target_version"] == 5
        assert r._task is not None  # the 409 kicked off a catch-up reconcile

    _run(go())


def test_version_flips_before_resume() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(store=FakeStore(VersionRef("r1", 4), _full("r1", 4)), engine=engine, commit_mode="in_place")
        r.applied = VersionRef("r1", 3)
        seen: dict[str, VersionRef | None] = {}
        base_resume = engine.resume

        async def resume_spy() -> None:
            seen["applied"] = r.applied
            await base_resume()

        engine.resume = resume_spy  # type: ignore[method-assign]
        await r.reconcile()
        assert seen["applied"] == VersionRef("r1", 4)  # flipped under the gate, before resume

    _run(go())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"reconcile harness: {len(tests)} PASS")
