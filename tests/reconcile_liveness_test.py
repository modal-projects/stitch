"""Convergence-liveness conformance tests for the wake-driven Reconciler.

The property under test (call it *self-convergence*): **every replica eventually
reaches the store's durable pointer, given only a healthy store — no further
publishes, no external nudges, no particular traffic mix.** v1's sidecar provided
it with a background reconcile loop (`servers/sglang.py`, `background_sync_interval`);
v2 dropped the loop, leaving three triggers only: a delivered `/wake`, a
constraint-409 (`Reconciler._on_reject`), and process startup (`sync.py`).

The xfail(strict=True) tests are RED today: each is a scenario where the durable
pointer is ahead, the store is healthy, and no mechanism ever converges the
replica. If a self-convergence mechanism lands (a periodic re-arm, a wake
generation counter, ...), they XPASS as errors — promote them to plain tests.
The green companions pin down the recovery channels that DO exist, so together
the file states the exact boundary of the current guarantee:

    convergence <=> (a wake is delivered while the pass can still see its effects)
                    or (traffic carries a version constraint that 409s)

Run the reds loud with: ``uv run pytest tests/reconcile_liveness_test.py --runxfail``
"""

from __future__ import annotations

import asyncio
import queue
import threading

import pytest

from reconcile_test import FakeEngine, FakeStore, _full
from stitch.sync import ConstraintUnmet, Reconciler
from stitch.versions import SyncState, VersionConstraint, VersionManifest, VersionRef


class FlakyStore(FakeStore):
    """FakeStore whose read_manifest fails N times, then heals — a transient
    store-side error (e.g. a Volume reload hiccup) that resolves on its own."""

    def __init__(self, pointer: VersionRef, *manifests: VersionManifest, failures: int = 1) -> None:
        super().__init__(pointer, *manifests)
        self.failures = failures

    def read_manifest(self, ref: VersionRef) -> VersionManifest:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("transient store error")
        return super().read_manifest(ref)


class HostViewStore(FakeStore):
    """FakeStore with the mounted-volume visibility split: ``advance_pointer``
    lands on the durable *remote*; a replica's ``read_pointer`` sees it only after
    ``refresh()`` snapshots remote -> local (Modal Volume reload semantics).
    ``refresh_gate`` lets a test hold a pass's snapshot open at a precise point."""

    def __init__(self, pointer: VersionRef | None = None, *manifests: VersionManifest) -> None:
        super().__init__(pointer, *manifests)
        self.remote_pointer = pointer
        self.refresh_gate: queue.Queue[threading.Event] | None = None

    def refresh(self) -> None:
        super().refresh()
        self._pointer = self.remote_pointer  # the reload: snapshot the durable state
        if self.refresh_gate is not None:
            release = threading.Event()
            self.refresh_gate.put(release)
            release.wait(timeout=10)  # released by the test; timeout only guards a hang

    def advance_pointer(self, ref: VersionRef) -> None:
        self.remote_pointer = ref


async def _converged(r: Reconciler, target: VersionRef, timeout: float = 1.0) -> bool:
    """True if the replica reaches (applied == target, IDLE) within ``timeout``."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if r.applied == target and r.sync_state is SyncState.IDLE:
            return True
        await asyncio.sleep(0.02)
    return False


def _run(coro) -> None:
    asyncio.run(coro)


# ── RED: self-convergence scenarios wake-driven-only cannot recover ──────────
@pytest.mark.xfail(
    strict=True,
    reason="no self-driven reconcile: after an ERROR pass, reconcile() returns and only an "
    "external wake retries — a healed store is never revisited (sync.py reconcile/wake)",
)
def test_transient_store_error_recovers_unaided() -> None:
    """Pointer at v1, one transient store error at boot, store heals immediately.
    Unconstrained traffic flows. The replica must eventually serve v1."""

    async def go() -> None:
        store = FlakyStore(VersionRef("r1", 1), _full("r1", 1), failures=1)
        r = Reconciler(store=store, engine=FakeEngine())
        await r.startup()  # the pass hits the transient error -> SyncState.ERROR
        for _ in range(3):  # store is healed; unconstrained traffic doesn't nudge anything
            async with r.admit(None):
                pass
        assert await _converged(r, VersionRef("r1", 1))

    _run(go())


@pytest.mark.xfail(
    strict=True,
    reason="publish-time pool wake is best-effort (publish.py _wake swallows failures; "
    "pools/modal_flash.py logs and drops) and unconstrained requests never 409, so a "
    "replica that misses the wake serves stale until an unrelated future publish",
)
def test_lost_publish_wake_converges() -> None:
    """A publish advances the durable pointer but its wake is lost in transit.
    Unconstrained traffic flows. The replica must eventually serve v1."""

    async def go() -> None:
        store = FakeStore(VersionRef("r1", 0))
        r = Reconciler(store=store, engine=FakeEngine())
        await r.startup()  # converges to the claimed base (r1, 0)
        store.publish(_full("r1", 1), "/fake/src")
        store.advance_pointer(VersionRef("r1", 1))  # durable; the wake never arrives
        for _ in range(3):
            async with r.admit(None):
                pass
        assert await _converged(r, VersionRef("r1", 1))

    _run(go())


@pytest.mark.xfail(
    strict=True,
    reason="wake() is a no-op while a pass runs (sync.py), and the pass's caught-up recheck "
    "reads a snapshot taken before the publish — the delivered wake is consumed by a pass "
    "that cannot see its effects (v1's wake-generation counter closed exactly this)",
)
def test_wake_delivered_during_pass_finale_converges() -> None:
    """Even with perfect wake delivery: a wake lands while the previous publish's
    pass is finishing on a pre-publish snapshot. The replica must reach v3."""

    async def go() -> None:
        engine = FakeEngine()
        store = HostViewStore(VersionRef("r1", 1), _full("r1", 1), _full("r1", 2), _full("r1", 3))
        r = Reconciler(store=store, engine=engine)
        await r.startup()  # converges to v1, ungated

        gate = store.refresh_gate = queue.Queue()
        store.advance_pointer(VersionRef("r1", 2))
        r.wake()  # publish v2's wake: starts a pass
        (await asyncio.to_thread(gate.get, True, 10)).set()  # pass-start refresh snapshots v2
        recheck = await asyncio.to_thread(gate.get, True, 10)  # hold the caught-up recheck open
        store.advance_pointer(VersionRef("r1", 3))
        r.wake()  # publish v3's wake: delivered mid-pass -> dropped by the running-task check
        assert r._task is not None and not r._task.done()  # scenario validity, not the property
        recheck.set()
        store.refresh_gate = None
        await r._task  # the recheck's snapshot showed v2: the pass idles at v2

        assert await _converged(r, VersionRef("r1", 3))

    _run(go())


# ── GREEN: the recovery channels that do exist (the guarantee's boundary) ────
def test_explicit_wake_recovers_error_state() -> None:
    """A delivered wake after the store heals converges the ERROR replica."""

    async def go() -> None:
        store = FlakyStore(VersionRef("r1", 1), _full("r1", 1), failures=1)
        r = Reconciler(store=store, engine=FakeEngine())
        await r.startup()
        assert r.sync_state is SyncState.ERROR
        r.wake()
        assert await _converged(r, VersionRef("r1", 1), timeout=5.0)

    _run(go())


def test_constrained_409_recovers_error_state() -> None:
    """A min_version request 409s against the stale replica; the rejection's
    self-wake converges it."""

    async def go() -> None:
        store = FlakyStore(VersionRef("r1", 1), _full("r1", 1), failures=1)
        r = Reconciler(store=store, engine=FakeEngine())
        await r.startup()
        assert r.sync_state is SyncState.ERROR
        with pytest.raises(ConstraintUnmet):
            async with r.admit(VersionConstraint(min_version=1)):
                pass
        assert await _converged(r, VersionRef("r1", 1), timeout=5.0)

    _run(go())


def test_constrained_409_recovers_lost_wake() -> None:
    """Same lost-wake scenario as the red test, but the traffic carries a
    staleness floor — the 409 self-wake is the one channel that heals it."""

    async def go() -> None:
        store = FakeStore(VersionRef("r1", 0))
        r = Reconciler(store=store, engine=FakeEngine())
        await r.startup()
        store.publish(_full("r1", 1), "/fake/src")
        store.advance_pointer(VersionRef("r1", 1))  # wake lost, as in the red test
        with pytest.raises(ConstraintUnmet):
            async with r.admit(VersionConstraint(min_version=1)):
                pass
        assert await _converged(r, VersionRef("r1", 1), timeout=5.0)

    _run(go())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
