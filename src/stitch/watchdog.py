"""Supervise a sidecar's colocated engine and propagate terminal failures.

The reconciler owns weight-sync recovery and says when inference progress is
expected. This module owns engine supervision and the sidecar process boundary.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Protocol

from stitch.engines.base import Engine, EngineHealthStatus
from stitch.errors import UnrecoverableEngineError

logger = logging.getLogger(__name__)


class _Server(Protocol):
    should_exit: bool

    async def serve(self) -> None: ...


class _Watchdog(Protocol):
    async def run(self) -> None: ...


class TerminalFailureMonitor:
    """Adapt a component's terminal-failure signal into the watchdog interface."""

    def __init__(self, wait_for_failure: Callable[[], Awaitable[None]]) -> None:
        self._wait_for_failure = wait_for_failure

    async def run(self) -> None:
        await self._wait_for_failure()


class SidecarWatchdog:
    """Fail when any independently supervised sidecar component fails."""

    def __init__(self, *components: _Watchdog) -> None:
        if not components:
            raise ValueError("sidecar watchdog requires at least one component")
        self._components = components

    async def run(self) -> None:
        tasks = {
            asyncio.create_task(component.run(), name=f"watchdog-{index}")
            for index, component in enumerate(self._components)
        }
        try:
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            await next(iter(done))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class EngineWatchdog:
    """Turn repeated engine-health failures into one terminal sidecar error.

    Probe only while the reconciler promises inference progress. Destination
    initialization, staging, and commits supervise their own engine RPCs and may
    occupy the engine's HTTP loop, so a concurrent health request is not an
    independent signal then.
    """

    def __init__(
        self,
        engine: Engine,
        *,
        expects_engine_progress: Callable[[], bool],
        interval: float = 5.0,
        failure_threshold: int = 3,
    ) -> None:
        if interval <= 0:
            raise ValueError("watchdog interval must be positive")
        if failure_threshold < 1:
            raise ValueError("watchdog failure threshold must be positive")
        self._engine = engine
        self._expects_engine_progress = expects_engine_progress
        self._interval = interval
        self._failure_threshold = failure_threshold
        self._consecutive_failures = 0

    async def run(self) -> None:
        """Run until cancelled or the engine is conclusively unrecoverable."""
        while True:
            if not self._expects_engine_progress():
                self._consecutive_failures = 0
                await asyncio.sleep(self._interval)
                continue

            health = await self._engine.check_health()
            if health.status is EngineHealthStatus.HEALTHY:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                logger.warning(
                    "engine watchdog failure %d/%d status=%s detail=%s",
                    self._consecutive_failures,
                    self._failure_threshold,
                    health.status.value,
                    health.detail,
                )
                if self._consecutive_failures >= self._failure_threshold:
                    raise UnrecoverableEngineError(
                        "local engine is persistently "
                        f"{health.status.value} after {self._consecutive_failures} checks: "
                        f"{health.detail}"
                    )
            await asyncio.sleep(self._interval)


async def run_server_with_watchdog(
    server: _Server,
    watchdog: _Watchdog,
    *,
    shutdown_timeout: float = 10.0,
) -> None:
    """Run uvicorn and make a terminal watchdog error reach the process boundary."""
    server_task = asyncio.create_task(server.serve(), name="sidecar-server")
    watchdog_task = asyncio.create_task(watchdog.run(), name="sidecar-watchdog")
    try:
        done, _pending = await asyncio.wait(
            {server_task, watchdog_task}, return_when=asyncio.FIRST_COMPLETED
        )
    except BaseException:
        server_task.cancel()
        watchdog_task.cancel()
        await asyncio.gather(server_task, watchdog_task, return_exceptions=True)
        raise

    if watchdog_task in done:
        try:
            await watchdog_task
        except BaseException:
            logger.critical("sidecar watchdog terminated the process", exc_info=True)
            await _stop_server(server, server_task, timeout=shutdown_timeout)
            raise
        await _stop_server(server, server_task, timeout=shutdown_timeout)
        raise UnrecoverableEngineError("engine watchdog exited unexpectedly")

    watchdog_task.cancel()
    with suppress(asyncio.CancelledError):
        await watchdog_task
    await server_task


async def _stop_server(
    server: _Server, server_task: asyncio.Task[None], *, timeout: float
) -> None:
    server.should_exit = True
    try:
        await asyncio.wait_for(asyncio.shield(server_task), timeout=timeout)
    except TimeoutError:
        logger.error("sidecar server did not stop within %.1fs; cancelling it", timeout)
        server_task.cancel()
        with suppress(BaseException):
            await server_task
    except BaseException:
        logger.exception("sidecar server failed while handling watchdog shutdown")
