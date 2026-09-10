"""Cooperative checkpoint I/O scheduling around live weight updates."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class CheckpointIO:
    """Stop admitting I/O while priority leases exist; never preempt an active call."""

    def __init__(self):
        self._condition = threading.Condition()
        self._leases: set[object] = set()
        self._active = 0

    def pause(self) -> object:
        token = object()
        with self._condition:
            self._leases.add(token)
        return token

    def resume(self, token: object) -> None:
        with self._condition:
            self._leases.discard(token)
            self._condition.notify_all()

    @property
    def quiescent(self) -> bool:
        with self._condition:
            return self._active == 0

    @contextmanager
    def operation(self, deadline: float):
        with self._condition:
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("checkpoint I/O exceeded upload deadline")
                if not self._leases:
                    self._active += 1
                    break
                self._condition.wait(timeout=deadline - time.monotonic())
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()
