"""The run's control service: the newest version, which engines may take traffic, the barrier."""

import secrets
import threading
import time
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Clocks:
    """Static per run, in seconds."""

    # How long an engine may lag, catch up or miss the control service before it is given up.
    patience: float = 120
    # An open engine not heard from this long is gone.
    lease: float = 30
    # Without a trainer call this long, the engines may scale to zero.
    idle: float = 1800

    @property
    def wait(self) -> float:
        """How long a call waits for news: a third of the lease, so a lost call is resent within it."""
        return self.lease / 3

    @property
    def barrier(self) -> float:
        """The longest a publish waits: a fresh service collects engines for a lease, then each open engine
        serves the version, falls silent past the lease or behind past the patience."""
        return self.lease + max(self.lease, self.patience)


class Refused(Exception):
    """A trainer call the run cannot take: from another start, or a version not built on the head."""


class Failed(Exception):
    """A version every open engine failed to apply."""


@dataclass
class _Engine:
    version: int
    heard: float
    open: bool = False
    behind_since: float | None = None
    stats: dict | None = None


class ControlService:
    def __init__(self, clocks: Clocks):
        """The run's start and head are unknown until /start or, after a restart, the trainer's next call."""
        self.clocks = clocks
        self.head: int | None = None
        self.start_version: int | None = None
        self.start_id: str | None = None
        self.engines: dict[str, _Engine] = {}
        self.ending: dict[str, str] = {}
        self.failed: dict[str, int] = {}
        self.started = self.trainer_heard = time.monotonic()
        self.condition = threading.Condition()

    def open(self, engine_id: str, version: int, start_id: str | None) -> dict:
        """{"open": may the engine take traffic at `version`, "head": the newest, "start_id": the
        run's start} or {"exit": why}. `start_id`: the start it knew when applying `version`."""
        with self.condition:
            self.condition.wait_for(lambda: self.head is not None, self.clocks.wait)
            if reason := self._exit_reason(engine_id, version, start_id):
                return self._end(engine_id, reason)
            engine = self._heard(engine_id, version)
            if self.head is None:
                return {"open": False, "head": version, "start_id": self.start_id}
            engine.open = version == self.head
            self.condition.notify_all()
            return {"open": engine.open, "head": self.head, "start_id": self.start_id}

    def next(
        self, engine_id: str, version: int, seen: int, start_id: str | None
    ) -> dict:
        """The engine serves `version`: {"head": N, "start_id": the run's start} once N is not
        `seen` or the wait ends, or {"exit": why}"""
        with self.condition:
            if reason := self._exit_reason(engine_id, version, start_id):
                return self._end(engine_id, reason)
            # Only an engine taking traffic calls next: it is open even if this service restarted.
            self._heard(engine_id, version).open = True
            self.condition.notify_all()
            self.condition.wait_for(
                lambda: (
                    engine_id in self.ending
                    or (self.head is not None and self.head != seen)
                ),
                self.clocks.wait,
            )
            if reason := self._exit_reason(engine_id, version, start_id):
                return self._end(engine_id, reason)
            return {
                "head": seen if self.head is None else self.head,
                "start_id": self.start_id,
            }

    def served(self, engine_id: str, version: int, stats: dict, start_id: str) -> dict:
        """The engine serves `version` from now on, `stats` from applying it: {} or {"exit": why}"""
        with self.condition:
            if reason := self._exit_reason(engine_id, version, start_id):
                return self._end(engine_id, reason)
            engine = self._heard(engine_id, version)
            engine.open, engine.stats = True, stats
            self.condition.notify_all()
            return {}

    def fail(self, engine_id: str, version: int) -> None:
        """The engine failed to apply `version` and is exiting; its other calls in flight end it."""
        with self.condition:
            engine = self.engines.pop(engine_id, None)
            if engine is not None and engine.open:
                self.failed[engine_id] = version
            self.ending[engine_id] = f"failed to apply version {version}"
            self.condition.notify_all()

    def leave(self, engine_id: str) -> None:
        """The engine is shutting down: no barrier waits for it, and its call in flight ends it."""
        with self.condition:
            self.engines.pop(engine_id, None)
            self.ending[engine_id] = "it left"
            self.condition.notify_all()

    def start(self, version: int) -> dict:
        """The run starts, or resumes, at `version`: the versions after it are gone, so engines past
        it exit and the others go to it. The reply's start_id names this start in later calls."""
        with self.condition:
            self.trainer_heard = time.monotonic()
            self.start_version, self.start_id = version, secrets.token_hex(8)
            self._set_head(version)
            for engine_id, engine in self.engines.items():
                if engine.version > version:
                    self.ending[engine_id] = f"the run resumed at {version}"
                    engine.open = False
            self.condition.notify_all()
            return {"start_id": self.start_id, **self._status()}

    def publish(
        self, version: int, base_version: int, start_version: int, start_id: str
    ) -> dict:
        """Waits until every engine taking traffic serves `version` after `base_version`; returns their stats."""
        with self.condition:
            self.trainer_heard = time.monotonic()
            self._check_trainer(start_version, start_id, base_version)
            if version > self.head:
                if base_version != self.head:
                    raise Refused(
                        f"version {version} (base {base_version}) does not follow the run's {self.head}"
                    )
                self._set_head(version)
        return self._barrier(version, start_id)

    def status(
        self,
        start_version: int | None = None,
        start_id: str | None = None,
        head: int | None = None,
    ) -> dict:
        with self.condition:
            self.trainer_heard = time.monotonic()
            if start_id is not None:
                self._check_trainer(start_version, start_id, head)
            return self._status()

    def _check_trainer(
        self, start_version: int | None, start_id: str, head: int | None
    ) -> None:
        """A trainer call names the run's start and head: a restarted service takes them from it."""
        if self.start_id is None:
            self.start_version, self.start_id = start_version, start_id
        elif (start_version, start_id) != (self.start_version, self.start_id):
            raise Refused("the call is from another start than the run's")
        if self.head is None and head is not None:
            self._set_head(head)

    def idle_seconds(self) -> float:
        return time.monotonic() - self.trainer_heard

    def _barrier(self, version: int, start_id: str) -> dict:
        started = time.monotonic()
        with self.condition:
            # Every open engine calls once a lease: only a service up that long knows all of them.
            while (left := self.started + self.clocks.lease - time.monotonic()) > 0:
                self.condition.wait(left)
            # An open engine that failed at `version` before this call counts too.
            expected = set(self._serving())
            expected |= {i for i, failed in self.failed.items() if failed == version}
            while True:
                if self.start_id != start_id:
                    raise Refused("the call is from another start than the run's")
                self._sweep()
                if all(e.version >= version for e in self._serving().values()):
                    break
                self.condition.wait(1.0)
            serving = self._serving()
            if (
                expected
                and not serving
                and all(self.failed.get(i) == version for i in expected)
            ):
                raise Failed(f"every open engine failed to apply version {version}")
            return {
                "version": version,
                "seconds": time.monotonic() - started,
                "engines": {
                    i: {"version": e.version, "stats": e.stats}
                    for i, e in serving.items()
                },
            }

    def _serving(self) -> dict[str, _Engine]:
        """Open engines heard from within the lease, as of now; any other is gone."""
        now = time.monotonic()
        return {
            i: e
            for i, e in self.engines.items()
            if e.open and now - e.heard <= self.clocks.lease
        }

    def _heard(self, engine_id: str, version: int) -> _Engine:
        now = time.monotonic()
        engine = self.engines.setdefault(engine_id, _Engine(version, now))
        # A call sent before a later one may arrive after it; an engine's version only grows.
        engine.version, engine.heard = max(engine.version, version), now
        if self.head is None or version >= self.head:
            engine.behind_since = None
        elif engine.behind_since is None:
            engine.behind_since = now
        return engine

    def _set_head(self, version: int) -> None:
        self.head, now = version, time.monotonic()
        for engine in self.engines.values():
            if engine.version < version and engine.behind_since is None:
                engine.behind_since = now
        self.condition.notify_all()

    def _exit_reason(
        self, engine_id: str, version: int, start_id: str | None
    ) -> str | None:
        """Why the engine must end, if it must: it was told to, or it holds `version` from versions
        the run's start removed, applied above the start under another start."""
        if engine_id in self.ending:
            return self.ending.pop(engine_id)
        if (
            self.start_version is not None
            and version > self.start_version
            and start_id != self.start_id
        ):
            return f"at {version} from before the run's start at {self.start_version}"
        return None

    def _end(self, engine_id: str, reason: str) -> dict:
        self.engines.pop(engine_id, None)
        self.condition.notify_all()
        return {"exit": reason}

    def _sweep(self) -> None:
        """Drops open engines silent past the lease, others past the patience; ends open ones behind past it."""
        now = time.monotonic()
        for engine_id, engine in list(self.engines.items()):
            silent = now - engine.heard
            if silent > (self.clocks.lease if engine.open else self.clocks.patience):
                del self.engines[engine_id]
            elif (
                engine.open
                and engine.behind_since is not None
                and now - engine.behind_since > self.clocks.patience
            ):
                self.ending[engine_id] = (
                    f"not at version {self.head} within {self.clocks.patience} s"
                )
                engine.open = False
                self.condition.notify_all()

    def _status(self) -> dict:
        self._sweep()
        return {
            "head": self.head,
            "serving": Counter(e.version for e in self._serving().values()),
            "catching_up": sum(not e.open for e in self.engines.values()),
        }
