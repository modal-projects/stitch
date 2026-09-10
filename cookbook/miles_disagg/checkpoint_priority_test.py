import asyncio
import threading
import time

import pytest

from cookbook.miles_disagg.checkpoint_priority import CheckpointIO, watch_serving
from stitch import service
from stitch.types import PoolState, ReplicaState, VersionRef


def test_old_completion_cannot_release_a_new_delta():
    io = CheckpointIO()
    first, second = io.pause(), io.pause()
    io.resume(first)
    with pytest.raises(TimeoutError, match="upload deadline"):
        with io.operation(time.monotonic() + 0.02):
            pytest.fail("new delta lost priority")
    io.resume(second)
    with io.operation(time.monotonic() + 1):
        assert not io.quiescent
    assert io.quiescent


def test_expired_lease_releases_checkpoint_even_if_monitor_is_stuck():
    io = CheckpointIO()
    token = io.pause()
    io.expire_after(token, 0.02)
    with io.operation(time.monotonic() + 1):
        assert not io.held(token)


def test_serving_hold_requires_all_replicas_on_the_same_run(monkeypatch):
    io = CheckpointIO()
    token = io.pause()
    queried = threading.Event()
    applied = threading.Event()

    async def readiness(pool, timeout):
        queried.set()
        return PoolState(
            [
                ReplicaState(ready=True, applied=VersionRef("run", 12)),
                ReplicaState(ready=True, applied=VersionRef("run", 11)),
                ReplicaState(
                    ready=True,
                    applied=VersionRef("run" if applied.is_set() else "other", 11),
                ),
            ]
        )

    monkeypatch.setattr(service, "readiness", readiness)
    thread = watch_serving(
        io,
        token,
        None,
        VersionRef("run", 11),
        host_rank=0,
        min_ready=2,
        timeout=2,
        interval=0.01,
    )
    assert queried.wait(1)
    assert io.held(token)
    applied.set()
    thread.join(1)
    assert not thread.is_alive()
    assert not io.held(token)


def test_monitor_timeout_and_request_errors_release_priority(monkeypatch, caplog):
    async def readiness(pool, timeout):
        raise OSError("pool discovery unavailable")

    monkeypatch.setattr(service, "readiness", readiness)
    io = CheckpointIO()
    token = io.pause()
    with caplog.at_level("INFO"):
        thread = watch_serving(
            io,
            token,
            None,
            VersionRef("run", 1),
            host_rank=0,
            min_ready=1,
            timeout=0.04,
            interval=0.01,
        )
        thread.join(1)
    assert not thread.is_alive()
    assert not io.held(token)
    assert "serving_timeout" in caplog.text
    assert "pool discovery unavailable" in caplog.text


def test_hung_request_cannot_extend_the_serving_timeout(monkeypatch):
    async def readiness(pool, timeout):
        await asyncio.sleep(60)

    monkeypatch.setattr(service, "readiness", readiness)
    io = CheckpointIO()
    token = io.pause()
    thread = watch_serving(
        io,
        token,
        None,
        VersionRef("run", 1),
        host_rank=0,
        min_ready=1,
        timeout=0.04,
        interval=0.01,
    )
    with io.operation(time.monotonic() + 1):
        pass
    thread.join(1)
    assert not thread.is_alive()
