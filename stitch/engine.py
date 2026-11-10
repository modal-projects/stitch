"""Stitch's code in each engine container: keeps the engine at the version the control service names."""

import contextlib
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import modal
from autoinference_utils.endpoint import terminate_container

from stitch.control import Clocks

logger = logging.getLogger(__name__)


class Engine(Protocol):
    """What Stitch needs from an inference engine. `version` is the version it serves."""

    version: int

    def start(self) -> None: ...

    def warm_up(self) -> None: ...

    def watch(self) -> None:
        """From here on, the engine exiting or failing its health checks ends the container."""

    def prepare(self, manifest: Path) -> dict | None:
        """Reads and stages a version while serving; the weights stay as they are. It leaves none of the
        version's files open: Modal reloads a Volume only with none of its files open."""

    def apply(self, manifest: Path, version: int) -> dict | None:
        """Applies the prepared version inside a pause; a failure leaves the weights partly updated."""

    def abort_all(self) -> None:
        """Ends every request in flight."""

    def stop(self) -> None: ...


class ControlClient:
    """The control service as one engine calls it: a call unanswered in twice the wait is sent again."""

    def __init__(self, service: "modal.cls.Obj", engine_id: str, clocks: Clocks):
        self.service, self.engine_id, self.clocks = service, engine_id, clocks

    def open(self, version: int, start_id: str | None) -> dict:
        return self._call(self.service.open, version, start_id)

    def next(self, version: int, seen: int, start_id: str | None) -> dict:
        return self._call(self.service.next, version, seen, start_id)

    def served(self, version: int, stats: dict, start_id: str) -> dict:
        return self._call(self.service.served, version, stats, start_id)

    def fail(self, version: int) -> None:
        self._call(self.service.fail, version)

    def leave(self) -> None:
        # Not waited on: a container shutting down does not wait for the control service.
        self.service.leave.spawn(self.engine_id)

    def _call(self, method, *args):
        deadline = time.monotonic() + self.clocks.patience
        while True:
            try:
                # No cancel on a timeout: cancelling a non-async input shuts its container down.
                return method.spawn(self.engine_id, *args).get(
                    timeout=2 * self.clocks.wait
                )
            except modal.exception.ClientClosed:
                # This container is ending, not the control service slow.
                raise
            except (TimeoutError, modal.exception.Error):
                if time.monotonic() > deadline:
                    raise
                logger.warning(
                    "the control service did not answer, calling again", exc_info=True
                )
                time.sleep(2)


class EngineContainer:
    """One engine's Modal container: applies versions/weight_vNNNNNN/manifest.json in order."""

    def __init__(
        self,
        engine: Engine,
        control: ControlClient,
        versions: Path,
        reload: Callable[[], None],
    ):
        self.engine, self.control, self.versions = engine, control, versions
        self.reload = reload
        self.head = engine.version
        # The run's start as last heard; the version this engine serves and the start it knew
        # when applying it.
        self.start_id: str | None = None
        self.applied: tuple[int, str | None] = (engine.version, None)
        self.view_head: int | None = None
        self.newer = threading.Condition()
        self.exiting = False

    def enter(self) -> None:
        """For @modal.enter: Modal routes traffic to the container only once this returns."""
        self.engine.start()
        self.engine.warm_up()
        self.open()
        self.engine.watch()
        for loop in (self.listen, self.follow):
            threading.Thread(
                target=self._run_or_exit, args=(loop,), daemon=True
            ).start()

    def exit(self) -> None:
        self.exiting = True
        with contextlib.suppress(Exception):
            self.control.leave()
        self.engine.stop()

    def open(self) -> None:
        """Catches up one version per call to the control service until it lets this engine open."""
        while True:
            reply = _raise_on_exit(self.control.open(*self.applied))
            self.start_id = reply["start_id"] or self.start_id
            self.head = reply["head"]
            if reply["open"]:
                return
            if self.head > self.engine.version:
                self._apply_next()

    def listen(self) -> None:
        """Keeps one call to the control service open, also while applying; wakes `follow`."""
        while True:
            version, start_id = self.applied
            reply = _raise_on_exit(self.control.next(version, self.head, start_id))
            with self.newer:
                self.head = reply["head"]
                self.start_id = reply["start_id"] or self.start_id
                self.newer.notify_all()

    def follow(self) -> None:
        """Applies each version up to the newest, in order, and reports each one served."""
        while True:
            with self.newer:
                self.newer.wait_for(lambda: self.head > self.engine.version)
            version, stats = self._apply_next()
            _raise_on_exit(self.control.served(version, stats, self.applied[1]))

    def _run_or_exit(self, loop: Callable[[], None]) -> None:
        try:
            loop()
        except BaseException:
            if self.exiting:
                return
            logger.exception("this engine cannot follow the run")
            with contextlib.suppress(Exception):
                self.engine.abort_all()
            terminate_container()

    def _apply_next(self) -> tuple[int, dict]:
        start_id, head = self.start_id, self.head
        # A version is whole only in a view reloaded after it was announced; a new start may also
        # have removed versions an older view still shows.
        if (
            start_id != self.applied[1]
            or self.view_head is None
            or head > self.view_head
        ):
            self.reload()
            self.view_head = head
        published = {
            int(match[1]): entry
            for entry in self.versions.iterdir()
            if (match := re.fullmatch(r"weight_v(\d+)", entry.name))
        }
        ahead = [v for v in published if self.engine.version < v <= head]
        if not ahead:
            raise RuntimeError(
                f"no published version above {self.engine.version} up to the head {head}"
            )
        version = min(ahead)
        manifest = published[version] / "manifest.json"
        try:
            stats = {
                "prepare": self.engine.prepare(manifest),
                "apply": self.engine.apply(manifest, version),
            }
        except Exception as error:
            # A leaving engine's version fails because its engine stopped: nothing to report.
            if not self.exiting:
                with contextlib.suppress(Exception):
                    self.control.fail(version)
            raise RuntimeError(f"version {version} from {manifest}: {error}") from error
        self.applied = (version, start_id)
        logger.info("version %d: %s", version, stats)
        return version, stats


def _raise_on_exit(reply: dict) -> dict:
    if "exit" in reply:
        raise RuntimeError(f"the control service ended this engine: {reply['exit']}")
    return reply
