"""Pool base harness: the ``*_async`` defaults delegate to the sync implementations
off the event loop, so any sync-only Pool subclass is usable from async callers."""

from __future__ import annotations

import asyncio
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from unittest.mock import patch

from stitch.pools.base import Pool
from stitch.types import VersionRef


class _SyncOnlyPool(Pool):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def gateway_url(self) -> str:
        self.calls.append(("gateway_url", threading.get_ident()))
        return "https://gw"

    def discover_replicas(self) -> list[str]:
        self.calls.append(("discover_replicas", threading.get_ident()))
        return ["https://r1", "https://r2"]

    def wake(self, replicas: list[str], ref: VersionRef) -> None:
        self.calls.append(("wake", threading.get_ident()))


async def _to_thread_without_threadsafe_loop_wakeup(function, /, *args, **kwargs):
    """Run a function off-loop without relying on the loop's self-pipe.

    Some restricted test sandboxes prohibit socket writes from worker threads.
    The standard ``asyncio.to_thread`` worker completes there, but cannot wake a
    sleeping selector. Polling the raw future at event-loop checkpoints keeps
    this unit test focused on the Pool delegation contract.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(function, *args, **kwargs)
        while not future.done():
            await asyncio.sleep(0)
        return future.result()


def _worker_thread_can_wake_event_loop() -> bool:
    """Whether this runtime permits a worker to write to a loop-style self-pipe."""
    reader, writer = socket.socketpair()
    errors: list[OSError] = []

    def write() -> None:
        try:
            writer.send(b"\0")
        except OSError as exc:
            errors.append(exc)

    try:
        worker = threading.Thread(target=write)
        worker.start()
        worker.join()
        return not errors
    finally:
        reader.close()
        writer.close()


def test_async_defaults_delegate_to_sync_impls_off_loop() -> None:
    pool = _SyncOnlyPool()

    async def drive() -> tuple[str, list[str]]:
        url = await pool.gateway_url_async()
        replicas = await pool.discover_replicas_async()
        await pool.wake_async(replicas, VersionRef("run", 1))
        return url, replicas

    to_thread_patch = (
        nullcontext()
        if _worker_thread_can_wake_event_loop()
        else patch.object(
            asyncio, "to_thread", _to_thread_without_threadsafe_loop_wakeup
        )
    )
    with to_thread_patch:
        url, replicas = asyncio.run(drive())  # the loop runs on THIS thread
    assert (url, replicas) == ("https://gw", ["https://r1", "https://r2"])
    assert [name for name, _ in pool.calls] == [
        "gateway_url",
        "discover_replicas",
        "wake",
    ]
    # the sync impls must have run on worker threads, never on the loop's thread
    assert all(ident != threading.get_ident() for _, ident in pool.calls)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"pool base harness: {len(tests)} PASS")
