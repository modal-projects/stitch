"""In-memory core harness (the Phase-1 gate): the real Reconciler + AdmissionGate
against fake Store / Engine — no Modal, sglang, or GPU. Runnable directly
(``python src/stitch/sync_test.py``) or under pytest."""

from __future__ import annotations

import asyncio
import queue
import threading

from stitch.engines.base import Engine
from stitch.stores.base import Store
from stitch.sync import ConstraintUnmet, Reconciler
from stitch.types import (
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
        self.calls: list[str] = []
        self.staged: list[VersionRef] = []
        self.committed: list[VersionRef] = []
        self.commit_weight_names: list[list[str] | None] = []

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        self.staged.append(manifest.ref)
        self.calls.append(f"stage:{manifest.ref.version}")

    async def commit(
        self, ref: VersionRef, *, flush_cache: bool = False, weight_names: list[str] | None = None
    ) -> None:
        self.committed.append(ref)
        self.commit_weight_names.append(weight_names)
        self.calls.append(f"commit:{ref.version}")

    async def flush_cache(self) -> None:
        self.calls.append("flush_cache")

    async def pause(self) -> None:
        self.calls.append("pause")

    async def resume(self) -> None:
        self.calls.append("resume")

    async def reset(self) -> None:
        self.calls.append("reset")

    async def prefetch(self) -> None:
        self.calls.append("prefetch")

    def stamp_request(self, request, served) -> None:
        pass

    def stamp_response(self, response, served, current) -> None:
        pass

    def base_url(self) -> str:
        return "http://engine"


def _full(run: str, version: int) -> VersionManifest:
    return VersionManifest(VersionRef(run, version), VersionKind.FULL, ["model.safetensors"])


def _delta(
    run: str, version: int, *, files: list[str], tensor_names: list[str] | None = None
) -> VersionManifest:
    return VersionManifest(VersionRef(run, version), VersionKind.DELTA, files, tensor_names or [])


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
        assert "flush_cache" not in engine.calls  # flushing is not automatic; it rides commit(flush_cache=…)

    _run(go())


def test_startup_prefetches_base() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(store=FakeStore(), engine=engine)  # unclaimed pool: reconcile is a no-op
        await r.startup()
        await r._prefetch_task  # let the background base-seed finish
        assert "prefetch" in engine.calls

    _run(go())


def test_catch_up() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(store=FakeStore(VersionRef("r1", 5), _full("r1", 5)), engine=engine)
        r.applied = VersionRef("r1", 3)
        await r.reconcile()
        assert r.applied == VersionRef("r1", 5)
        assert engine.committed == [VersionRef("r1", 5)]  # one composed stage+reload, not per-version
        assert engine.commit_weight_names == [None]  # FULL target reseeds -> full reload, not partial

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


def test_run_switch_drains_rolling_requests() -> None:
    # Base reset is incompatible: even in in_place, no rolling request crosses the wipe (drain_all; stitch#32).
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(store=FakeStore(VersionRef("r2", 1), _full("r2", 1)), engine=engine, commit_mode="in_place")
        r.applied = VersionRef("r1", 5)
        release = asyncio.Event()
        late_served: list[VersionRef | None] = []

        async def rolling() -> None:
            async with r.admit(None):
                await release.wait()

        async def late() -> None:
            async with r.admit(None) as served:
                late_served.append(served)

        req = asyncio.create_task(rolling())
        await asyncio.sleep(0)  # admitted before the switch begins
        sync = asyncio.create_task(r.reconcile())
        for _ in range(1000):  # bounded: without drain_all the switch completes without draining
            if r._committing:
                break
            await asyncio.sleep(0.001)
        late_task = asyncio.create_task(late())
        await asyncio.sleep(0.05)
        assert "reset" not in engine.calls  # the wipe waits for the rolling request
        assert not late_served  # and nothing is admitted while draining
        release.set()
        await asyncio.gather(req, sync, late_task)
        assert "reset" in engine.calls
        assert late_served[0] is not None and late_served[0].run_id == "r2"  # admitted post-wipe

    _run(go())


def test_rolling_requests_cross_in_place_commit() -> None:
    # Counterpart: a compatible in_place commit applies while rolling traffic keeps decoding; only a base reset drains.
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(store=FakeStore(VersionRef("r1", 4), _full("r1", 4)), engine=engine, commit_mode="in_place")
        r.applied = VersionRef("r1", 3)
        release = asyncio.Event()

        async def rolling() -> None:
            async with r.admit(None):
                await release.wait()

        req = asyncio.create_task(rolling())
        await asyncio.sleep(0)
        await r.reconcile()  # completes while the rolling request is still in flight
        assert r.applied == VersionRef("r1", 4)
        release.set()
        await req

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


def test_touched_names_union_reaches_commit() -> None:
    # A multi-version catch-up hands the engine the UNION of touched tensor names across the
    # applied→target range, so an O(delta) reload refreshes every tensor any delta changed —
    # not just the target's. (This is the plumbing a partial reload needs; without it the
    # engine names no tensors and pays a full reload.)
    async def go() -> None:
        engine = FakeEngine()
        store = FakeStore(
            VersionRef("r1", 5),
            _delta("r1", 4, files=["f1"], tensor_names=["a", "b"]),
            _delta("r1", 5, files=["f1", "f2"], tensor_names=["b", "c"]),
        )
        r = Reconciler(store=store, engine=engine)
        r.applied = VersionRef("r1", 3)
        await r.reconcile()
        assert r.applied == VersionRef("r1", 5)
        assert engine.commit_weight_names == [["a", "b", "c"]]  # union of v4 + v5, deduped + sorted
        assert r.metrics.get("reload_names") == 3

    _run(go())


def test_delta_without_touched_names_full_reloads_not_skips() -> None:
    # A store that records a non-empty delta but NOT its touched tensor names must full-reload,
    # never skip: an empty touched-set reads as "nothing changed" and would advance the version
    # onto stale weights (a downstream DeltaStore regression). weight_names=None => full reload.
    async def go() -> None:
        engine = FakeEngine()
        store = FakeStore(VersionRef("r1", 5), _delta("r1", 5, files=["f1"], tensor_names=[]))
        r = Reconciler(store=store, engine=engine)
        r.applied = VersionRef("r1", 4)
        await r.reconcile()
        assert r.applied == VersionRef("r1", 5)
        assert engine.committed == [VersionRef("r1", 5)]  # reloaded, not skipped
        assert engine.commit_weight_names == [None]       # full reload (touched names unknown)
        assert r.metrics.get("skipped_reload") is not True

    _run(go())


def test_periodic_reconcile_recovers_missed_wake() -> None:
    async def go() -> None:
        engine = FakeEngine()
        store = FakeStore(VersionRef("r1", 3), _full("r1", 3), _full("r1", 5))
        r = Reconciler(store=store, engine=engine, reconcile_interval=0.02)
        await r.startup()
        assert r.applied == VersionRef("r1", 3)
        # Publish advances latest but its wake never lands; only the background loop catches up.
        store.advance_pointer(VersionRef("r1", 5))
        await asyncio.sleep(0.1)
        assert r.applied == VersionRef("r1", 5)  # the backstop caught up on its own
        await r.shutdown()

    _run(go())


def test_reconcile_interval_zero_disables_backstop() -> None:
    async def go() -> None:
        store = FakeStore(VersionRef("r1", 3), _full("r1", 3), _full("r1", 5))
        r = Reconciler(store=store, engine=FakeEngine(), reconcile_interval=0.0)
        await r.startup()
        store.advance_pointer(VersionRef("r1", 5))
        await asyncio.sleep(0.1)
        assert r.applied == VersionRef("r1", 3)  # no backstop: stays until a wake/409
        assert r._periodic_task is None
        await r.shutdown()

    _run(go())


def test_stage_waits_for_prefetch() -> None:
    # Prefetch and stage both write the checkpoint; the stage must wait for the prefetch so they never race.
    async def go() -> None:
        engine = FakeEngine()
        release = asyncio.Event()
        base_prefetch = engine.prefetch

        async def slow_prefetch() -> None:
            await release.wait()
            await base_prefetch()

        engine.prefetch = slow_prefetch  # type: ignore[method-assign]
        r = Reconciler(store=FakeStore(VersionRef("r1", 2), _full("r1", 2)), engine=engine, reconcile_interval=0.0)
        r.applied = VersionRef("r1", 0)  # same run, behind -> stage v2 (no run switch)
        task = asyncio.create_task(r.startup())
        await asyncio.sleep(0.05)  # startup fires the prefetch; reconcile reaches the await
        assert "stage:2" not in engine.calls  # blocked on the (still-running) prefetch
        release.set()
        await task
        assert engine.calls.index("prefetch") < engine.calls.index("stage:2")
        await r.shutdown()

    _run(go())


# ── convergence liveness ─────────────────────────────────────────────────────
# Backstop self-heals what wake-only cannot (stitch#45): each converges with it, never with interval=0.


async def _converged(r: Reconciler, target: VersionRef, timeout: float = 1.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if r.applied == target and r.sync_state is SyncState.IDLE:
            return True
        await asyncio.sleep(0.02)
    return False


class FlakyStore(FakeStore):
    """read_manifest fails once, then heals — a transient store-side error."""

    failures = 1

    def read_manifest(self, ref: VersionRef) -> VersionManifest:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transient store error")
        return super().read_manifest(ref)


async def _heals_transient_error(interval: float) -> bool:
    r = Reconciler(store=FlakyStore(VersionRef("r1", 1), _full("r1", 1)), engine=FakeEngine(),
                   reconcile_interval=interval)
    await r.startup()  # the pass hits the error -> ERROR; the store is healed from here on
    async with r.admit(None):
        pass  # unconstrained traffic never 409s, so it nudges nothing
    ok = await _converged(r, VersionRef("r1", 1))
    await r.shutdown()
    return ok


def test_transient_error_recovery_needs_backstop() -> None:
    assert asyncio.run(_heals_transient_error(0.05))
    assert not asyncio.run(_heals_transient_error(0))  # ERROR retries only on external wake


class HostViewStore(FakeStore):
    """advance_pointer lands on the durable *remote*; read_pointer sees it only after
    refresh() snapshots remote -> local (Volume reload semantics). refresh_gate lets a
    test hold a pass open on a pre-publish snapshot."""

    refresh_gate: queue.Queue[threading.Event] | None = None

    def __init__(self, pointer: VersionRef | None = None, *manifests: VersionManifest) -> None:
        super().__init__(None, *manifests)
        self.remote_pointer = pointer

    def refresh(self) -> None:
        self._pointer = self.remote_pointer
        if self.refresh_gate is not None:
            self.refresh_gate.put(release := threading.Event())
            release.wait(timeout=10)

    def advance_pointer(self, ref: VersionRef) -> None:
        self.remote_pointer = ref


async def _heals_dropped_wake(interval: float) -> bool:
    """A wake IS delivered, but mid-pass: wake() no-ops against the running task, whose
    caught-up recheck already snapshotted pre-publish state — the wake is lost."""
    store = HostViewStore(VersionRef("r1", 1), _full("r1", 1), _full("r1", 2))
    r = Reconciler(store=store, engine=FakeEngine(), reconcile_interval=interval)
    await r.startup()  # converges to v1, ungated

    gate = store.refresh_gate = queue.Queue()
    r.wake()  # start an idle pass
    (await asyncio.to_thread(gate.get, True, 10)).set()  # release its pass-start refresh
    recheck = await asyncio.to_thread(gate.get, True, 10)  # its recheck: snapshotted v1, held
    store.advance_pointer(VersionRef("r1", 2))
    r.wake()  # v2's wake: delivered mid-pass -> dropped; the pass idles on its v1 snapshot
    recheck.set()
    store.refresh_gate = None
    ok = await _converged(r, VersionRef("r1", 2))
    await r.shutdown()
    return ok


def test_dropped_wake_recovery_needs_backstop() -> None:
    assert asyncio.run(_heals_dropped_wake(0.05))
    assert not asyncio.run(_heals_dropped_wake(0))


def test_constrained_409_recovers_without_backstop() -> None:
    """The event-driven channel: a min_version 409 self-wakes a stale ERROR replica."""

    async def go() -> None:
        r = Reconciler(store=FlakyStore(VersionRef("r1", 1), _full("r1", 1)), engine=FakeEngine(),
                       reconcile_interval=0)
        await r.startup()
        assert r.sync_state is SyncState.ERROR
        try:
            async with r.admit(VersionConstraint(min_version=1)):
                raise AssertionError("should have rejected")
        except ConstraintUnmet:
            pass
        assert await _converged(r, VersionRef("r1", 1))

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


def test_unapplied_replica_rejects() -> None:
    # Non-blocking startup serves /health before the first sync lands a version; a request in
    # that window has no served version to stamp, so it must 409 (retryable), not serve unversioned.
    async def go() -> None:
        r = Reconciler(store=FakeStore(), engine=FakeEngine())
        assert r.applied is None
        try:
            async with r.admit(None):
                raise AssertionError("should have rejected")
        except ConstraintUnmet as e:
            assert e.error["type"] == "WeightVersionNotReady"

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
