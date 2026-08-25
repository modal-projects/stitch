"""Sidecar engine-watchdog policy and process-propagation tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from stitch.engines.base import EngineHealth, EngineHealthStatus
from stitch.sync import Reconciler
from stitch.types import SyncState
from stitch.watchdog import (
    EngineWatchdog,
    SidecarWatchdog,
    TerminalFailureMonitor,
    UnrecoverableEngineError,
    run_server_with_watchdog,
)


class _HealthSequence:
    def __init__(self, *statuses: EngineHealthStatus) -> None:
        self._statuses = iter(statuses)
        self.checks = 0
        self.checked = asyncio.Event()

    async def check_health(self) -> EngineHealth:
        self.checks += 1
        try:
            status = next(self._statuses)
        except StopIteration:
            self.checked.set()
            status = EngineHealthStatus.HEALTHY
        return EngineHealth(status, status.value)


class _FakeServer:
    def __init__(self) -> None:
        self.should_exit = False
        self.stopped = asyncio.Event()

    async def serve(self) -> None:
        try:
            while not self.should_exit:
                await asyncio.sleep(0)
        finally:
            self.stopped.set()


def _watchdog(
    engine: _HealthSequence,
    *,
    expects_engine_progress: Callable[[], bool] = lambda: True,
    failure_threshold: int = 2,
) -> EngineWatchdog:
    return EngineWatchdog(
        engine,  # type: ignore[arg-type]
        expects_engine_progress=expects_engine_progress,
        interval=0.001,
        failure_threshold=failure_threshold,
    )


def test_watchdog_does_not_probe_when_progress_is_not_expected() -> None:
    async def go() -> None:
        engine = _HealthSequence(EngineHealthStatus.UNRESPONSIVE)
        task = asyncio.create_task(
            _watchdog(engine, expects_engine_progress=lambda: False).run()
        )
        await asyncio.sleep(0.01)
        assert not task.done()
        assert engine.checks == 0
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())


def test_probe_timeouts_during_lease_hold_count_toward_watchdog() -> None:
    """HOLDING is not a stage/commit: the engine still serves, so probe
    failures count toward the unrecoverable verdict."""

    async def go() -> None:
        reconciler = Reconciler(store=None, engine=None, run_id="r1")  # type: ignore[arg-type]
        reconciler.ready = True
        reconciler.sync_state = SyncState.HOLDING
        engine = _HealthSequence(
            EngineHealthStatus.UNRESPONSIVE,
            EngineHealthStatus.UNRESPONSIVE,
        )
        with pytest.raises(UnrecoverableEngineError, match="unresponsive"):
            await _watchdog(
                engine,
                expects_engine_progress=reconciler.expects_engine_progress,
            ).run()
        assert engine.checks == 2

    asyncio.run(go())


def test_transient_idle_failure_does_not_trip_watchdog() -> None:
    async def go() -> None:
        engine = _HealthSequence(
            EngineHealthStatus.UNRESPONSIVE,
            EngineHealthStatus.HEALTHY,
            EngineHealthStatus.UNRESPONSIVE,
        )
        task = asyncio.create_task(_watchdog(engine).run())
        await asyncio.wait_for(engine.checked.wait(), timeout=1)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())


def test_watchdog_failure_stops_server_and_reaches_process_boundary() -> None:
    async def go() -> None:
        server = _FakeServer()
        engine = _HealthSequence(
            EngineHealthStatus.UNREACHABLE,
            EngineHealthStatus.UNREACHABLE,
        )
        with pytest.raises(UnrecoverableEngineError):
            await run_server_with_watchdog(server, _watchdog(engine))
        assert server.stopped.is_set()

    asyncio.run(go())


def test_sidecar_watchdog_propagates_component_terminal_error() -> None:
    async def go() -> None:
        async def terminal_failure() -> None:
            raise UnrecoverableEngineError("cannot restore boot weights")

        engine = _HealthSequence(EngineHealthStatus.HEALTHY)
        watchdog = SidecarWatchdog(
            _watchdog(engine),
            TerminalFailureMonitor(terminal_failure),
        )
        with pytest.raises(UnrecoverableEngineError, match="restore boot weights"):
            await watchdog.run()

    asyncio.run(go())
