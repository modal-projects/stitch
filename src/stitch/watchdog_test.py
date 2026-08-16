"""Sidecar engine-watchdog policy and process-propagation tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from stitch.engines.base import EngineHealth, EngineHealthStatus
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
        self.checked = asyncio.Event()
        self.check_count = 0

    async def check_health(self) -> EngineHealth:
        self.check_count += 1
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
    engine_health_may_be_stale: Callable[[], bool] = lambda: False,
    wait_until_monitorable: Callable[[], Awaitable[None]] | None = None,
    failure_threshold: int = 2,
) -> EngineWatchdog:
    return EngineWatchdog(
        engine,  # type: ignore[arg-type]
        engine_health_may_be_stale=engine_health_may_be_stale,
        wait_until_monitorable=wait_until_monitorable,
        interval=0.001,
        failure_threshold=failure_threshold,
    )


def test_engine_watchdog_starts_after_engine_becomes_monitorable() -> None:
    async def go() -> None:
        monitorable = asyncio.Event()
        engine = _HealthSequence(
            EngineHealthStatus.UNRESPONSIVE,
            EngineHealthStatus.UNRESPONSIVE,
        )
        task = asyncio.create_task(
            _watchdog(
                engine,
                wait_until_monitorable=monitorable.wait,
            ).run()
        )

        await asyncio.sleep(0.01)
        assert engine.check_count == 0
        assert not task.done()

        monitorable.set()
        with pytest.raises(UnrecoverableEngineError, match="unresponsive"):
            await task

    asyncio.run(go())


def test_expected_unresponsiveness_during_weight_apply_is_recoverable() -> None:
    async def go() -> None:
        engine = _HealthSequence(
            EngineHealthStatus.UNRESPONSIVE,
            EngineHealthStatus.UNRESPONSIVE,
        )
        task = asyncio.create_task(
            _watchdog(engine, engine_health_may_be_stale=lambda: True).run()
        )
        await asyncio.wait_for(engine.checked.wait(), timeout=1)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

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


def test_unreachable_engine_is_terminal_even_during_weight_apply() -> None:
    async def go() -> None:
        engine = _HealthSequence(
            EngineHealthStatus.UNREACHABLE,
            EngineHealthStatus.UNREACHABLE,
        )
        with pytest.raises(UnrecoverableEngineError, match="unreachable"):
            await _watchdog(engine, engine_health_may_be_stale=lambda: True).run()

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

        monitorable = asyncio.Event()
        engine = _HealthSequence(EngineHealthStatus.HEALTHY)
        watchdog = SidecarWatchdog(
            _watchdog(engine, wait_until_monitorable=monitorable.wait),
            TerminalFailureMonitor(terminal_failure),
        )
        with pytest.raises(UnrecoverableEngineError, match="restore boot weights"):
            await watchdog.run()
        assert engine.check_count == 0

    asyncio.run(go())
