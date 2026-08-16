"""Keep the test suite usable from sandboxes that block sockets/IPC.

Some environments (agent sandboxes in particular) deny the syscalls event
loops and local test servers rely on: ``socket()``, ``socketpair()``, and the
cross-thread wakeups behind ``asyncio.to_thread``. Tests there don't merely
fail - the loop's own shutdown path wedges too, so the whole pytest process
hangs for minutes or forever and even a per-test SIGALRM can't unwind it
(asyncio's ``Runner.close`` re-blocks the same way while being interrupted).
This conftest handles that in two layers:

1. Each session starts by probing the environment. If sockets or an
   asyncio+executor round-trip don't work, the event-loop/socket test modules
   are *skipped* with an explicit reason instead of run into the wedge. On a
   healthy machine nothing is skipped and the suite is unchanged.
2. As a backstop, every test phase gets a repeating wall-clock watchdog. If
   anything unforeseen still wedges, the first interrupt fails the test with a
   full traceback dump; if even that can't unwind it, the session aborts
   loudly - a pytest run in this repo never just sits there.

Watchdog budget per phase in seconds: ``STITCH_TEST_TIMEOUT`` (default 30,
``0`` disables the watchdog; the sandbox probe/skip always applies).
"""

from __future__ import annotations

import faulthandler
import os
import signal
import socket
import sys
import threading

import pytest

TIMEOUT_ENV_VAR = "STITCH_TEST_TIMEOUT"
DEFAULT_TIMEOUT_S = 30.0

# Modules whose tests need a working event loop / real sockets.
SKIP_WHEN_SANDBOXED = ("src/stitch/service_test.py", "src/stitch/sync_test.py")

_sandbox_reason: str | None = None


def _probe_environment() -> str | None:
    """Return a human-readable reason the sandbox is broken, or None."""
    try:
        left, right = socket.socketpair()
        left.close()
        right.close()
    except OSError as exc:
        return f"sockets unavailable ({exc})"

    # The asyncio default executor relies on cross-thread loop wakeups; if the
    # sandbox breaks those, *every* asyncio.run wedges inside selectors.select.
    # Probe one full round-trip with a hard alarm so the probe itself can never
    # hang the session.
    def _alarm(signum: int, frame: object) -> None:
        raise TimeoutError("asyncio probe timed out")

    import asyncio

    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, 5.0)
    try:

        async def _roundtrip() -> str:
            return await asyncio.to_thread(str, "ok")

        asyncio.run(_roundtrip())
    except BaseException as exc:  # noqa: BLE001 - report anything the sandbox throws
        return f"asyncio event loop cannot complete a cross-thread round-trip ({exc!r})"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
    return None


def pytest_sessionstart(session: pytest.Session) -> None:
    global _sandbox_reason
    _sandbox_reason = _probe_environment()


def pytest_report_header(config: pytest.Config) -> list[str]:
    if _sandbox_reason is None:
        return []
    return [
        f"sandbox detected ({_sandbox_reason}); skipping event-loop/socket tests "
        "(run outside the sandbox for full coverage)"
    ]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _sandbox_reason is None:
        return
    skip = pytest.mark.skip(reason=f"sandboxed environment ({_sandbox_reason})")
    for item in items:
        if any(str(item.path).endswith(module) for module in SKIP_WHEN_SANDBOXED):
            item.add_marker(skip)


def _configured_timeout() -> float:
    raw = os.environ.get(TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_TIMEOUT_S
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return DEFAULT_TIMEOUT_S


class TestTimeoutError(Exception):
    """Raised in the test's thread when its wall-clock budget expires."""


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item):
    return (yield from _with_watchdog(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    return (yield from _with_watchdog(item))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item):
    return (yield from _with_watchdog(item))


def _with_watchdog(item: pytest.Item):
    """Arm a repeating SIGALRM around one runtest phase, disarm after.

    Repeating (not one-shot) because a single raise only breaks the *current*
    block: interrupting a wedged asyncio loop throws into ``Runner.close``,
    which promptly blocks again the same way. The first interrupt is a normal
    test error; a second one means even unwinding is wedged, so give up on
    graceful reporting and kill the session instead of waiting forever. The
    raise (rather than warn) matters: per PEP 475 the interrupted syscall is
    not retried, so the C-level wait unwinds immediately.
    """
    timeout = _configured_timeout()
    can_arm = (
        timeout > 0
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
    )
    if not can_arm:
        return (yield)

    fires = 0

    def _fire(signum: int, frame: object) -> None:
        nonlocal fires
        fires += 1
        faulthandler.dump_traceback(sys.stderr, all_threads=True)
        if fires == 1:
            raise TestTimeoutError(
                f"{item.nodeid} exceeded its {timeout:g}s budget "
                f"(bump {TIMEOUT_ENV_VAR} or set it to 0 to disable the watchdog)"
            )
        pytest.exit(
            f"watchdog: {item.nodeid} did not unwind after an interrupt; aborting the run",
            returncode=3,
        )

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, timeout, timeout)
    try:
        return (yield)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)
