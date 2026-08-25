"""In-memory core harness (the Phase-1 gate): the real Reconciler + AdmissionGate
against fake Store / Engine — no Modal, sglang, or GPU. Runnable directly
(``python src/stitch/sync_test.py``) or under pytest."""

from __future__ import annotations

import asyncio
import time
import queue
import threading
from functools import partial

import pytest

from stitch.engines.base import Engine
from stitch.errors import UnrecoverableEngineError
from stitch.stores.base import Store
from stitch.sync import AdmissionGate, ConstraintUnmet, Reconciler
from stitch.types import (
    SyncState,
    VersionConstraint,
    VersionKind,
    VersionManifest,
    VersionRef,
)

_make_reconciler = partial(Reconciler, run_id="r1")


class FakeStore(Store):
    def __init__(
        self, pointer: VersionRef | None = None, *manifests: VersionManifest
    ) -> None:
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

    def claim(self, boot: VersionRef) -> None:
        self._pointer = boot

    def publish(self, manifest: VersionManifest, files_dir: str) -> None:
        self._manifests[(manifest.ref.run_id, manifest.ref.version)] = manifest


class FakeEngine(Engine):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.staged: list[VersionRef] = []
        self.committed: list[VersionRef] = []
        self.initialized_versions: list[int] = []

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        self.staged.append(manifest.ref)
        self.calls.append(f"stage:{manifest.ref.version}")

    async def commit(
        self,
        manifest: VersionManifest,
        *,
        flush_cache: bool = False,
    ) -> None:
        self.committed.append(manifest.ref)
        self.calls.append(f"commit:{manifest.ref.version}")

    async def flush_cache(self) -> None:
        self.calls.append("flush_cache")

    async def pause(self) -> None:
        self.calls.append("pause")

    async def resume(self) -> None:
        self.calls.append("resume")

    async def reset(self) -> None:
        self.calls.append("reset")

    async def initialize_update_destination(self, boot_version: int = 0) -> None:
        self.initialized_versions.append(boot_version)
        self.calls.append("initialize_update_destination")

    def stamp_request(self, request, served) -> None:
        pass

    def stamp_response(self, response, served, current) -> None:
        pass

    def base_url(self) -> str:
        return "http://engine"


def _full(run: str, version: int) -> VersionManifest:
    return VersionManifest(
        VersionRef(run, version), VersionKind.FULL, ["model.safetensors"]
    )


def _delta(run: str, version: int, *, files: list[str]) -> VersionManifest:
    return VersionManifest(VersionRef(run, version), VersionKind.DELTA, files)


def _run(coro) -> None:
    asyncio.run(coro)


# ── reconcile ────────────────────────────────────────────────────────────────
def test_fresh_reconcile() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 3), _full("r1", 3)),
            engine=engine,
            commit_mode="quiesce",
        )
        await r.startup()
        assert r.applied == VersionRef("r1", 3)
        assert engine.staged[-1] == VersionRef("r1", 3)
        assert VersionRef("r1", 3) in engine.committed
        assert r.sync_state is SyncState.IDLE
        assert r.ready
        assert (
            "flush_cache" not in engine.calls
        )  # flushing is not automatic; it rides commit(flush_cache=…)

    _run(go())


def test_reconcile_latches_ready() -> None:
    async def go() -> None:
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 2), _full("r1", 2)), engine=FakeEngine()
        )
        assert (
            not r.ready
        )  # /health keeps a replica out of Flash rotation until it has caught up
        await r.reconcile()
        assert r.ready

    _run(go())


def test_startup_initializes_update_destination() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = _make_reconciler(
            store=FakeStore(), engine=engine
        )  # unclaimed pool: reconcile is a no-op
        await r.startup()
        assert "initialize_update_destination" in engine.calls
        assert r.ready

    _run(go())


@pytest.mark.parametrize(
    "ready,sync_state,expects_progress",
    [
        (False, SyncState.IDLE, False),
        (False, SyncState.FETCHING, False),
        (False, SyncState.HOLDING, False),
        (False, SyncState.STAGING, False),
        (False, SyncState.COMMITTING, False),
        (False, SyncState.ERROR, False),
        (True, SyncState.IDLE, True),
        (True, SyncState.FETCHING, True),
        (True, SyncState.HOLDING, True),
        (True, SyncState.STAGING, True),
        (True, SyncState.COMMITTING, False),
        (True, SyncState.ERROR, True),
    ],
)
def test_engine_progress_expectation_follows_serving_lifecycle(
    ready: bool, sync_state: SyncState, expects_progress: bool
) -> None:
    reconciler = _make_reconciler(store=FakeStore(), engine=FakeEngine())
    reconciler.ready = ready
    reconciler.sync_state = sync_state

    assert reconciler.expects_engine_progress() is expects_progress


def test_catch_up() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 5), _full("r1", 5)), engine=engine
        )
        r.applied = VersionRef("r1", 3)
        await r.reconcile()
        assert r.applied == VersionRef("r1", 5)
        assert engine.committed == [
            VersionRef("r1", 5)
        ]  # one staged chain and one commit

    _run(go())


def test_latest_advance_during_stage_coalesces_before_commit() -> None:
    async def go() -> None:
        manifests = [
            _delta("r1", version, files=[f"v{version}"]) for version in range(7, 11)
        ]
        store = FakeStore(VersionRef("r1", 7), *manifests)
        engine = FakeEngine()
        stage_started = asyncio.Event()
        finish_stage = asyncio.Event()
        base_stage = engine.stage

        async def slow_stage(manifest: VersionManifest, source_dir: str) -> None:
            await base_stage(manifest, source_dir)
            if manifest.ref.version == 7:
                stage_started.set()
                await finish_stage.wait()

        engine.stage = slow_stage  # type: ignore[method-assign]
        r = _make_reconciler(store=store, engine=engine)
        r.applied = VersionRef("r1", 6)

        syncing = asyncio.create_task(r.reconcile())
        await stage_started.wait()
        store.advance_pointer(VersionRef("r1", 10))
        finish_stage.set()
        await syncing

        assert engine.staged == [VersionRef("r1", 7), VersionRef("r1", 10)]
        assert engine.committed == [VersionRef("r1", 10)]
        assert r.applied == VersionRef("r1", 10)
        assert r.metrics["initial_target_version"] == 7
        assert r.metrics["target_version"] == 10
        assert r.metrics["coalesced_versions"] == 3

    _run(go())


def test_continuous_publishing_cannot_starve_commit() -> None:
    async def go() -> None:
        store = FakeStore(
            VersionRef("r1", 1),
            *(_delta("r1", version, files=[f"v{version}"]) for version in range(1, 4)),
        )
        engine = FakeEngine()
        stage = engine.stage

        async def advancing_stage(
            manifest: VersionManifest,
            source_dir: str,
        ) -> None:
            await stage(manifest, source_dir)
            if manifest.ref.version < 3:
                store.advance_pointer(VersionRef("r1", manifest.ref.version + 1))

        engine.stage = advancing_stage  # type: ignore[method-assign]
        r = _make_reconciler(store=store, engine=engine)
        r.applied = VersionRef("r1", 0)

        caught_up = await r._reconcile_once()

        assert not caught_up
        assert engine.staged == [VersionRef("r1", 1), VersionRef("r1", 2)]
        assert engine.committed == [VersionRef("r1", 2)]
        assert r.applied == VersionRef("r1", 2)
        assert store.read_pointer() == VersionRef("r1", 3)

    _run(go())


def test_coalesce_does_not_cross_run_lineage() -> None:
    async def go() -> None:
        store = FakeStore(
            VersionRef("r1", 1),
            _delta("r1", 1, files=["old"]),
            _delta("r2", 1, files=["new"]),
        )
        engine = FakeEngine()
        stage = engine.stage

        async def switching_stage(
            manifest: VersionManifest,
            source_dir: str,
        ) -> None:
            await stage(manifest, source_dir)
            store.advance_pointer(VersionRef("r2", 1))

        engine.stage = switching_stage  # type: ignore[method-assign]
        r = _make_reconciler(store=store, engine=engine)
        r.applied = VersionRef("r1", 0)

        assert not await r._reconcile_once()
        assert engine.staged == [VersionRef("r1", 1)]
        assert engine.committed == [VersionRef("r1", 1)]
        assert r.applied == VersionRef("r1", 1)

    _run(go())


def test_coalesce_observation_failure_commits_staged_target() -> None:
    class FailSecondRefreshStore(FakeStore):
        def refresh(self) -> None:
            super().refresh()
            if self.refreshed == 2:
                raise RuntimeError("transient store error")

    async def go() -> None:
        store = FailSecondRefreshStore(
            VersionRef("r1", 1),
            _delta("r1", 1, files=["v1"]),
        )
        engine = FakeEngine()
        r = _make_reconciler(store=store, engine=engine)
        r.applied = VersionRef("r1", 0)

        assert await r._reconcile_once()
        assert engine.staged == [VersionRef("r1", 1)]
        assert engine.committed == [VersionRef("r1", 1)]
        assert r.applied == VersionRef("r1", 1)
        assert r.metrics["coalesce_error"] == "transient store error"

    _run(go())


def test_run_switch_resets_in_place() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = _make_reconciler(
            store=FakeStore(VersionRef("r2", 2), _full("r2", 2)),
            engine=engine,
            commit_mode="in_place",
        )
        r.applied = VersionRef("r1", 5)
        await r.reconcile()
        assert r.applied == VersionRef("r2", 2)
        assert "reset" in engine.calls  # was patched -> restore boot checkpoint
        assert (
            engine.calls.index("pause")
            < engine.calls.index("reset")
            < engine.calls.index("resume")
        )
        assert (
            "flush_cache" not in engine.calls
        )  # cache flushing is an explicit commit policy

    _run(go())


def test_run_switch_exposes_paused_reset_to_engine_watchdog() -> None:
    async def go() -> None:
        engine = FakeEngine()
        reset_started = asyncio.Event()
        finish_reset = asyncio.Event()

        async def slow_reset() -> None:
            engine.calls.append("reset")
            reset_started.set()
            await finish_reset.wait()

        engine.reset = slow_reset  # type: ignore[method-assign]
        r = _make_reconciler(
            store=FakeStore(),
            engine=engine,
            commit_mode="in_place",
        )
        r.applied = VersionRef("r0", 5)

        switch = asyncio.create_task(r._switch_run("r1"))
        await reset_started.wait()
        assert engine.calls == ["pause", "reset"]
        assert r.sync_state is SyncState.COMMITTING
        assert not r.expects_engine_progress()

        finish_reset.set()
        await switch

    _run(go())


def test_run_switch_drains_rolling_requests() -> None:
    # Base reset is incompatible: even in in_place, no rolling request crosses the wipe (drain_all; stitch#32).
    async def go() -> None:
        engine = FakeEngine()
        r = _make_reconciler(
            store=FakeStore(VersionRef("r2", 1), _full("r2", 1)),
            engine=engine,
            commit_mode="in_place",
        )
        r.applied = VersionRef("r1", 5)
        release = asyncio.Event()
        late_served: list[VersionRef | None] = []

        async def rolling() -> None:
            async with r.gate.admit(None):
                await release.wait()

        async def late() -> None:
            async with r.gate.admit(None) as served:
                late_served.append(served)

        req = asyncio.create_task(rolling())
        await asyncio.sleep(0)  # admitted before the switch begins
        sync = asyncio.create_task(r.reconcile())
        for _ in range(
            1000
        ):  # bounded: without drain_all the switch completes without draining
            if r.gate._committing:
                break
            await asyncio.sleep(0.001)
        late_task = asyncio.create_task(late())
        await asyncio.sleep(0.05)
        assert "reset" not in engine.calls  # the wipe waits for the rolling request
        assert not late_served  # and nothing is admitted while draining
        release.set()
        await asyncio.gather(req, sync, late_task)
        assert "reset" in engine.calls
        assert (
            late_served[0] is not None and late_served[0].run_id == "r2"
        )  # admitted post-wipe

    _run(go())


def test_rolling_requests_cross_in_place_commit() -> None:
    # Counterpart: a compatible in_place commit applies while rolling traffic keeps decoding; only a boot reset drains.
    async def go() -> None:
        engine = FakeEngine()
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 4), _full("r1", 4)),
            engine=engine,
            commit_mode="in_place",
        )
        r.applied = VersionRef("r1", 3)
        release = asyncio.Event()

        async def rolling() -> None:
            async with r.gate.admit(None):
                await release.wait()

        req = asyncio.create_task(rolling())
        await asyncio.sleep(0)
        await r.reconcile()  # completes while the rolling request is still in flight
        assert r.applied == VersionRef("r1", 4)
        release.set()
        await req

    _run(go())


def test_new_request_waits_for_in_place_commit() -> None:
    async def go() -> None:
        engine = FakeEngine()
        commit_started = asyncio.Event()
        finish_commit = asyncio.Event()
        original_commit = engine.commit

        async def slow_commit(
            manifest: VersionManifest,
            *,
            flush_cache: bool = False,
        ) -> None:
            commit_started.set()
            await finish_commit.wait()
            await original_commit(manifest, flush_cache=flush_cache)

        engine.commit = slow_commit  # type: ignore[method-assign]
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 4), _full("r1", 4)),
            engine=engine,
            commit_mode="in_place",
        )
        r.applied = VersionRef("r1", 3)
        rolling_release = asyncio.Event()
        late_served: list[VersionRef | None] = []

        async def rolling() -> None:
            async with r.gate.admit():
                await rolling_release.wait()

        async def late() -> None:
            async with r.gate.admit() as served:
                late_served.append(served)

        rolling_task = asyncio.create_task(rolling())
        await asyncio.sleep(0)
        sync_task = asyncio.create_task(r.reconcile())
        await commit_started.wait()
        late_task = asyncio.create_task(late())
        await asyncio.sleep(0.01)
        assert not late_served

        finish_commit.set()
        await asyncio.gather(sync_task, late_task)
        assert late_served == [VersionRef("r1", 4)]
        assert not rolling_task.done()
        rolling_release.set()
        await rolling_task

    _run(go())


def test_empty_delta_skips_commit() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 4), _delta("r1", 4, files=[])),
            engine=engine,
        )
        r.applied = VersionRef("r1", 3)
        await r.reconcile()
        assert r.applied == VersionRef("r1", 4)
        assert engine.staged == []
        assert engine.committed == []  # no commit for a zero-file delta
        assert r.metrics.get("skipped_weight_update") is True

    _run(go())


def test_nonempty_delta_in_catch_up_range_commits() -> None:
    async def go() -> None:
        engine = FakeEngine()
        store = FakeStore(
            VersionRef("r1", 5),
            _delta("r1", 4, files=["f1"]),
            _delta("r1", 5, files=[]),
        )
        r = _make_reconciler(store=store, engine=engine)
        r.applied = VersionRef("r1", 3)
        await r.reconcile()
        assert r.applied == VersionRef("r1", 5)
        assert engine.committed == [VersionRef("r1", 5)]

    _run(go())


def test_nonempty_delta_commits() -> None:
    async def go() -> None:
        engine = FakeEngine()
        store = FakeStore(VersionRef("r1", 5), _delta("r1", 5, files=["f1"]))
        r = _make_reconciler(store=store, engine=engine)
        r.applied = VersionRef("r1", 4)
        await r.reconcile()
        assert r.applied == VersionRef("r1", 5)
        assert engine.committed == [VersionRef("r1", 5)]
        assert r.metrics.get("skipped_weight_update") is not True

    _run(go())


def test_periodic_reconcile_recovers_missed_wake() -> None:
    async def go() -> None:
        engine = FakeEngine()
        store = FakeStore(VersionRef("r1", 3), _full("r1", 3), _full("r1", 5))
        r = _make_reconciler(store=store, engine=engine, reconcile_interval=0.02)
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
        r = _make_reconciler(store=store, engine=FakeEngine(), reconcile_interval=0.0)
        await r.startup()
        store.advance_pointer(VersionRef("r1", 5))
        await asyncio.sleep(0.1)
        assert r.applied == VersionRef("r1", 3)  # no backstop: stays until a wake/409
        assert r._periodic_task is None
        await r.shutdown()

    _run(go())


def test_stage_waits_for_update_destination() -> None:
    async def go() -> None:
        engine = FakeEngine()
        started = asyncio.Event()
        release = asyncio.Event()
        initialize = engine.initialize_update_destination

        async def slow_initialize(boot_version: int = 0) -> None:
            started.set()
            await release.wait()
            await initialize(boot_version)

        engine.initialize_update_destination = slow_initialize  # type: ignore[method-assign]
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 2), _full("r1", 2)),
            engine=engine,
            reconcile_interval=0.0,
        )
        r.applied = VersionRef("r1", 0)  # same run, behind -> stage v2 (no run switch)
        task = asyncio.create_task(r.startup())
        await started.wait()
        assert "stage:2" not in engine.calls
        assert not r.ready
        release.set()
        await task
        assert engine.calls.index("initialize_update_destination") < engine.calls.index(
            "stage:2"
        )
        assert r.ready
        await r.shutdown()

    _run(go())


def test_boot_weights_wait_for_update_destination() -> None:
    async def go() -> None:
        engine = FakeEngine()
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_initialize(boot_version: int = 0) -> None:
            started.set()
            await release.wait()
            engine.initialized_versions.append(boot_version)
            engine.calls.append("initialize_update_destination")

        engine.initialize_update_destination = slow_initialize  # type: ignore[method-assign]
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 0)),
            engine=engine,
            reconcile_interval=0.0,
        )
        startup = asyncio.create_task(r.startup())
        await started.wait()
        assert not r.ready
        assert r.applied == VersionRef("r1", 0)
        assert "pause" not in engine.calls
        assert not r._destination_ready
        release.set()
        await startup
        assert r._destination_ready
        assert r.ready
        await r.shutdown()

    _run(go())


def test_store_refresh_waits_for_update_destination() -> None:
    async def go() -> None:
        engine = FakeEngine()
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_initialize(boot_version: int = 0) -> None:
            started.set()
            await release.wait()
            engine.initialized_versions.append(boot_version)
            engine.calls.append("initialize_update_destination")

        engine.initialize_update_destination = slow_initialize  # type: ignore[method-assign]

        class GuardedStore(FakeStore):
            def refresh(self) -> None:
                assert release.is_set(), (
                    "store refreshed while checkpoint files were open"
                )
                super().refresh()

        store = GuardedStore(VersionRef("r1", 0))
        r = _make_reconciler(store=store, engine=engine, reconcile_interval=0.0)
        startup = asyncio.create_task(r.startup())
        await started.wait()
        assert not r.ready
        assert store.refreshed == 0

        release.set()
        await startup
        assert store.refreshed == 1
        await r.shutdown()

    _run(go())


def test_update_fails_after_update_destination_initialization_fails() -> None:
    async def go() -> None:
        engine = FakeEngine()

        async def fail_initialize(boot_version: int = 0) -> None:
            raise RuntimeError("broken destination")

        engine.initialize_update_destination = fail_initialize  # type: ignore[method-assign]
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 2), _full("r1", 2)),
            engine=engine,
            reconcile_interval=0.0,
        )
        r.applied = VersionRef("r1", 0)
        terminal = asyncio.create_task(r.wait_for_terminal_error())
        await r.startup()
        assert not r.ready
        assert "stage:2" not in engine.calls
        with pytest.raises(
            UnrecoverableEngineError,
            match="weight update destination initialization failed.*broken destination",
        ):
            await terminal
        assert "broken destination" in r.server_info()["terminal_error"]
        await r.shutdown()

    _run(go())


def test_unrecoverable_reconcile_error_reaches_terminal_monitor() -> None:
    async def go() -> None:
        engine = FakeEngine()

        async def fail_stage(_manifest, _source_dir) -> None:
            raise UnrecoverableEngineError("engine process is gone")

        engine.stage = fail_stage  # type: ignore[method-assign]
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 2), _full("r1", 2)),
            engine=engine,
            reconcile_interval=0,
        )
        await r.startup()
        with pytest.raises(UnrecoverableEngineError, match="process is gone"):
            await r.wait_for_terminal_error()
        assert r.server_info()["terminal_error"] == "engine process is gone"
        await r.shutdown()

    _run(go())


def test_commit_failure_is_terminal() -> None:
    async def go() -> None:
        engine = FakeEngine()

        async def fail_commit(
            _manifest: VersionManifest,
            *,
            flush_cache: bool = False,
        ) -> None:
            raise RuntimeError("partial weight copy")

        engine.commit = fail_commit  # type: ignore[method-assign]
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 1), _delta("r1", 1, files=["v1"])),
            engine=engine,
            reconcile_interval=0,
        )
        await r.startup()

        with pytest.raises(
            UnrecoverableEngineError,
            match="weight commit failed; live engine state is uncertain",
        ):
            await r.wait_for_terminal_error()
        assert not r.ready
        assert r.applied == VersionRef("r1", 0)
        assert engine.calls == [
            "initialize_update_destination",
            "stage:1",
            "pause",
            "resume",
        ]
        await r.shutdown()

    _run(go())


# ── convergence liveness ─────────────────────────────────────────────────────
# The periodic backstop heals a wake that never arrives. A delivered wake is sufficient.


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
    r = _make_reconciler(
        store=FlakyStore(VersionRef("r1", 1), _full("r1", 1)),
        engine=FakeEngine(),
        reconcile_interval=interval,
    )
    await (
        r.startup()
    )  # the pass hits the error -> ERROR; the store is healed from here on
    async with r.gate.admit(None):
        pass  # unconstrained traffic never 409s, so it nudges nothing
    ok = await _converged(r, VersionRef("r1", 1))
    await r.shutdown()
    return ok


def test_transient_error_recovery_needs_backstop() -> None:
    assert asyncio.run(_heals_transient_error(0.05))
    assert not asyncio.run(
        _heals_transient_error(0)
    )  # ERROR retries only on external wake


class HostViewStore(FakeStore):
    """advance_pointer lands on the durable *remote*; read_pointer sees it only after
    refresh() snapshots remote -> local (Volume reload semantics). refresh_gate lets a
    test hold a pass open on a pre-publish snapshot."""

    refresh_gate: queue.Queue[threading.Event] | None = None

    def __init__(
        self, pointer: VersionRef | None = None, *manifests: VersionManifest
    ) -> None:
        super().__init__(None, *manifests)
        self.remote_pointer = pointer

    def refresh(self) -> None:
        self._pointer = self.remote_pointer
        if self.refresh_gate is not None:
            self.refresh_gate.put(release := threading.Event())
            release.wait(timeout=10)

    def advance_pointer(self, ref: VersionRef) -> None:
        self.remote_pointer = ref


async def _preserves_concurrent_wake(interval: float) -> bool:
    """A publish wake delivered during a pass schedules another authoritative read."""
    store = HostViewStore(VersionRef("r1", 1), _full("r1", 1), _full("r1", 2))
    r = _make_reconciler(store=store, engine=FakeEngine(), reconcile_interval=interval)
    await r.startup()  # converges to v1, ungated

    gate = store.refresh_gate = queue.Queue()
    r.wake()  # start an idle pass
    refresh = await asyncio.to_thread(gate.get, True, 10)  # snapshotted v1, held
    store.advance_pointer(VersionRef("r1", 2))
    r.wake()  # delivered mid-pass: request another pass
    store.refresh_gate = None
    refresh.set()
    ok = await _converged(r, VersionRef("r1", 2))
    await r.shutdown()
    return ok


def test_wake_during_reconcile_is_preserved_without_backstop() -> None:
    assert asyncio.run(_preserves_concurrent_wake(0))


def test_constrained_409_recovers_without_backstop() -> None:
    """The event-driven channel: a min_version 409 self-wakes a stale ERROR replica."""

    async def go() -> None:
        r = _make_reconciler(
            store=FlakyStore(VersionRef("r1", 1), _full("r1", 1)),
            engine=FakeEngine(),
            reconcile_interval=0,
        )
        await r.startup()
        assert r.sync_state is SyncState.ERROR
        try:
            async with r.gate.admit(VersionConstraint(min_version=1)):
                raise AssertionError("should have rejected")
        except ConstraintUnmet:
            pass
        assert await _converged(r, VersionRef("r1", 1))

    _run(go())


# ── admission gate ───────────────────────────────────────────────────────────
def test_admit_satisfied() -> None:
    async def go() -> None:
        r = _make_reconciler(store=FakeStore(), engine=FakeEngine())
        r.applied = VersionRef("r1", 5)
        async with r.gate.admit(VersionConstraint(min_version=3)) as served:
            assert served == VersionRef("r1", 5)

    _run(go())


def test_admit_rejected_triggers_wake() -> None:
    async def go() -> None:
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 5), _full("r1", 5)), engine=FakeEngine()
        )
        r.applied = VersionRef("r1", 2)
        try:
            async with r.gate.admit(VersionConstraint(min_version=5)):
                raise AssertionError("should have rejected")
        except ConstraintUnmet as e:
            assert e.error["type"] == "WeightVersionNotReady"
            assert e.error["target_version"] == 5
        assert r._task is not None  # the 409 kicked off a catch-up reconcile

    _run(go())


def test_reconciler_requires_nonempty_run_id() -> None:
    with pytest.raises(ValueError, match="run_id is required"):
        Reconciler(store=FakeStore(), engine=FakeEngine(), run_id="")


def test_run_scoped_replica_serves_boot_checkpoint_as_version_zero() -> None:
    async def go() -> None:
        r = Reconciler(store=FakeStore(), engine=FakeEngine(), run_id="r1")
        await r.startup()
        try:
            assert r.applied == VersionRef("r1", 0)
            assert r.ready
            async with r.gate.admit(VersionConstraint(exact_version=0)) as served:
                assert served == VersionRef("r1", 0)
        finally:
            await r.shutdown()

    _run(go())


def test_resumed_replica_serves_boot_checkpoint_at_saved_version() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = Reconciler(
            store=FakeStore(VersionRef("resumed", 119)),
            engine=engine,
            run_id="resumed",
            boot_version=119,
        )
        await r.startup()
        try:
            assert r.applied == VersionRef("resumed", 119)
            assert r.ready
            assert not engine.staged
            assert engine.initialized_versions == [119]
            async with r.gate.admit(VersionConstraint(exact_version=119)) as served:
                assert served == VersionRef("resumed", 119)
        finally:
            await r.shutdown()

    _run(go())


def test_resumed_replica_applies_only_versions_after_saved_checkpoint() -> None:
    async def go() -> None:
        engine = FakeEngine()
        target = _delta("resumed", 120, files=["delta"])
        r = Reconciler(
            store=FakeStore(VersionRef("resumed", 120), target),
            engine=engine,
            run_id="resumed",
            boot_version=119,
        )
        await r.reconcile()
        assert engine.staged == [VersionRef("resumed", 120)]
        assert engine.committed == [VersionRef("resumed", 120)]
        assert r.applied == VersionRef("resumed", 120)

    _run(go())


def test_version_flips_before_resume() -> None:
    async def go() -> None:
        engine = FakeEngine()
        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 4), _full("r1", 4)),
            engine=engine,
            commit_mode="in_place",
        )
        r.applied = VersionRef("r1", 3)
        seen: dict[str, VersionRef | None] = {}
        base_resume = engine.resume

        async def resume_spy() -> None:
            seen["applied"] = r.applied
            await base_resume()

        engine.resume = resume_spy  # type: ignore[method-assign]
        await r.reconcile()
        assert seen["applied"] == VersionRef(
            "r1", 4
        )  # flipped under the gate, before resume

    _run(go())


# ── version leases ───────────────────────────────────────────────────────────
def test_lease_lifecycle() -> None:
    """Acquire, renew, think-time hold, expiry, snapshot counts, then commit."""

    async def go() -> None:
        gate = AdmissionGate(
            served_version=lambda: VersionRef("r1", 3), version_lease_ttl=0.1
        )
        pin = VersionConstraint(exact_version=3)
        async with gate.admit(pin, lease_key="sess-1"):
            pass
        # The lease outlives its request: think-time still holds the version.
        assert gate.leases_snapshot() == {3: 1}
        await asyncio.sleep(0.06)
        async with gate.admit(pin, lease_key="sess-1"):
            pass
        await asyncio.sleep(0.06)
        assert gate.leases_snapshot() == {3: 1}
        await asyncio.sleep(0.1)
        assert gate.leases_snapshot() == {}

        gate = AdmissionGate(
            served_version=lambda: VersionRef("r1", 3), version_lease_ttl=30.0
        )
        for key in ("sess-1", "sess-2"):
            async with gate.admit(pin, lease_key=key):
                pass
        assert gate.leases_snapshot() == {3: 2}

        gate = AdmissionGate(
            served_version=lambda: VersionRef("r1", 5), version_lease_ttl=60.0
        )
        async with gate.admit(VersionConstraint(exact_version=5), lease_key="sess"):
            pass
        assert gate.leases_snapshot() == {5: 1}
        applied: list[str] = []

        async def apply() -> None:
            applied.append("applied")

        await asyncio.wait_for(
            gate.commit(apply=apply, on_applied=lambda: None, target_version=5),
            timeout=1.0,
        )
        assert applied == ["applied"], "lease at or above target does not block commit"

        gate = AdmissionGate(served_version=lambda: VersionRef("r1", 3))
        async with gate.admit(VersionConstraint(exact_version=3), lease_key="sess"):
            pass
        assert gate.leases_snapshot() == {}
        applied = []
        await asyncio.wait_for(
            gate.commit(apply=apply, on_applied=lambda: None, target_version=4),
            timeout=1.0,
        )
        assert applied == ["applied"], "ttl 0 skips the hold"

        r = _make_reconciler(
            store=FakeStore(VersionRef("r1", 3), _full("r1", 3)),
            engine=FakeEngine(),
            version_lease_ttl=30.0,
        )
        r.applied = VersionRef("r1", 3)
        async with r.gate.admit(VersionConstraint(exact_version=3), lease_key="sess-1"):
            pass
        info = r.server_info()
        assert info["applied_version"] == 3
        assert info["leases"] == {3: 1}
        r = _make_reconciler(store=FakeStore(), engine=FakeEngine())
        info = r.server_info()
        assert info["applied_version"] == 0
        assert info["leases"] == {}

    _run(go())


def test_lease_admitted_in_hold_close_gap_reenters_hold() -> None:
    """A lease slipping in between the hold returning and admission closing
    must re-enter the hold, never be committed past."""
    asyncio.run(_lease_slips_into_hold_close_gap())


async def _lease_slips_into_hold_close_gap() -> None:
    gate = AdmissionGate(commit_mode="in_place", version_lease_ttl=60.0)
    original_hold = gate.hold_for_leases
    slipped = False

    async def hold_then_slip(target: int) -> None:
        nonlocal slipped
        await original_hold(target)
        if not slipped:
            slipped = True
            # Admitted lease lands after the hold returns, before the close.
            gate._leases["late-session"] = (1, time.monotonic() + 60.0)

    gate.hold_for_leases = hold_then_slip  # type: ignore[method-assign]
    applied: list[bool] = []

    async def apply() -> None:
        applied.append(True)

    commit_task = asyncio.create_task(
        gate.commit(apply=apply, on_applied=lambda: None, target_version=2)
    )
    await asyncio.sleep(0.05)
    assert not applied, "commit must not proceed past the slipped lease"
    assert not gate._committing, "admission must stay open while re-holding"
    del gate._leases["late-session"]
    async with gate._cond:
        gate._cond.notify_all()
    await asyncio.wait_for(commit_task, timeout=2.0)
    assert applied, "commit proceeds once the slipped lease is gone"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"reconcile harness: {len(tests)} PASS")
