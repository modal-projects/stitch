"""Cooperative checkpoint I/O scheduling around live weight updates."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CheckpointIO:
    """Stop admitting I/O while priority leases exist; never preempt an active call."""

    def __init__(self):
        self._condition = threading.Condition()
        self._leases: dict[object, float | None] = {}
        self._active = 0

    def pause(self) -> object:
        token = object()
        with self._condition:
            self._leases[token] = None
        return token

    def expire_after(self, token: object, seconds: float) -> None:
        with self._condition:
            if token in self._leases:
                self._leases[token] = time.monotonic() + seconds
            self._condition.notify_all()

    def resume(self, token: object) -> None:
        with self._condition:
            self._leases.pop(token, None)
            self._condition.notify_all()

    def held(self, token: object) -> bool:
        with self._condition:
            self._expire()
            return token in self._leases

    def _expire(self) -> None:
        now = time.monotonic()
        self._leases = {
            token: until
            for token, until in self._leases.items()
            if until is None or until > now
        }

    @property
    def quiescent(self) -> bool:
        with self._condition:
            return self._active == 0

    @contextmanager
    def operation(self, deadline: float):
        with self._condition:
            while True:
                self._expire()
                if time.monotonic() >= deadline:
                    raise TimeoutError("checkpoint I/O exceeded upload deadline")
                if not self._leases:
                    self._active += 1
                    break
                self._condition.wait(timeout=min(0.1, deadline - time.monotonic()))
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()


def watch_serving(
    io: CheckpointIO,
    token: object,
    pool,
    ref,
    *,
    host_rank: int,
    min_ready: int,
    timeout: float,
    interval: float = 5,
) -> threading.Thread:
    """Release a priority lease after serving applies the version, off the trainer thread.

    The gate also expires the lease itself, so a stuck discovery/request cannot
    indefinitely starve checkpoint persistence. A newer lease survives this one.
    """
    from stitch.service import readiness

    started = time.monotonic()
    io.expire_after(token, timeout)

    async def watch():
        count, required = 0, min_ready
        reason = "serving_timeout"
        error = None
        try:
            while io.held(token):
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    state = await asyncio.wait_for(
                        readiness(pool, timeout=min(5, remaining)),
                        timeout=min(10, remaining),
                    )
                    count = sum(
                        replica.ready
                        and replica.applied is not None
                        and replica.applied.run_id == ref.run_id
                        and replica.applied.version >= ref.version
                        for replica in state.replicas
                    )
                    required = max(min_ready, len(state.replicas))
                    if count >= required:
                        reason = "serving_applied"
                        error = None
                        break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                remaining = timeout - (time.monotonic() - started)
                await asyncio.sleep(min(interval, max(0, remaining)))
            if reason == "serving_timeout" and time.monotonic() - started < timeout:
                reason = "serving_superseded"
        finally:
            io.resume(token)
            logger.info(
                "CHECKPOINT %s",
                json.dumps(
                    dict(
                        phase=reason,
                        host_rank=host_rank,
                        version=ref.version,
                        ready=count,
                        required=required,
                        seconds=time.monotonic() - started,
                        error=error,
                    ),
                    sort_keys=True,
                ),
            )

    thread = threading.Thread(
        target=lambda: asyncio.run(watch()),
        name="checkpoint-delta-priority",
        daemon=True,
    )
    thread.start()
    return thread
