"""Stitch's code in each engine container: keeps the engine at the version the control service names."""

import contextlib
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import modal
from autoinference_utils.endpoint import terminate_container

from stitch.control import Clocks

logger = logging.getLogger(__name__)


# The receiver request's fields of a publication, which its manifest carries.
PUBLICATION_FIELDS = (
    "stream_id",
    "base_stream_id",
    "base_version",
    "target_version",
    "plan_digest",
    "payload_checksum_format",
)


class Engine(Protocol):
    """What Stitch needs from an inference engine. `version` is the version it serves, -1 before the run's first."""

    version: int

    def start(self) -> None: ...

    def warm_up(self) -> None: ...

    def watch(self) -> None:
        """From here on, the engine exiting or failing its health checks ends the container."""

    def committed(self, plan_digest: str) -> tuple[str | None, int]:
        """The (stream, version) its weights end at, (None, 0) for the checkpoint it loaded; `plan_digest` is the
        run's checkpoint plan, which its weights must share."""

    def apply(self, publication: dict, version: int | None) -> dict | None:
        """Prepares one publication while serving, then applies it inside a pause and labels the weights `version`
        (None until the last of a catch-up). A failure leaves the weights partly updated. It leaves none of the
        publication's files open: Modal reloads a Volume only with none of its files open."""

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


def publication_chain(
    versions: Path, head: dict, committed: tuple[str | None, int]
) -> list[dict]:
    """The publications from an engine's `committed` (stream, version) to the head's, oldest first.

    Each manifest names its base: the stream's previous version, the stream a resume continued, or
    the checkpoint (None, 0). A publication lies at <stream>/weight_vNNNNNN/manifest.json under `versions`.
    """
    chain, manifest = [], head["manifest"]
    point = (head["stream_id"], head["target_version"])
    while point != committed:
        if point == (None, 0):
            raise RuntimeError(
                f"weights at {committed} are not on the run's publications to {head['stream_id']}:"
                f"{head['target_version']}: the run resumed before them"
            )
        data = (versions / manifest).read_bytes()
        record = json.loads(data)
        digest = hashlib.sha256(data).hexdigest()
        if not chain and digest != head["manifest_sha256"]:
            raise RuntimeError(f"{manifest} is not the announced publication")
        chain.append(
            {
                "manifest": manifest,
                "manifest_path": str(versions / manifest),
                "manifest_sha256": digest,
                **{key: record[key] for key in PUBLICATION_FIELDS},
            }
        )
        point = (record["base_stream_id"], record["base_version"])
        manifest = f"{point[0]}/weight_v{point[1]:06d}/manifest.json"
    return chain[::-1]


class EngineContainer:
    """One engine's Modal container: applies the run's publications in order, up to the head's."""

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
        # The head's publication, as the control service last named it.
        self.publication: dict | None = None
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
        """Catches up to the head per call to the control service until it lets this engine open."""
        while True:
            reply = _raise_on_exit(self.control.open(*self.applied))
            self.start_id = reply["start_id"] or self.start_id
            self.head, self.publication = reply["head"], reply["publication"]
            if reply["open"]:
                return
            if self.head > self.engine.version:
                self._catch_up()

    def listen(self) -> None:
        """Keeps one call to the control service open, also while applying; wakes `follow`."""
        while True:
            version, start_id = self.applied
            reply = _raise_on_exit(self.control.next(version, self.head, start_id))
            with self.newer:
                self.head, self.publication = reply["head"], reply["publication"]
                self.start_id = reply["start_id"] or self.start_id
                self.newer.notify_all()

    def follow(self) -> None:
        """Applies the publications up to the newest version, in order, and reports it served."""
        while True:
            with self.newer:
                self.newer.wait_for(lambda: self.head > self.engine.version)
            version, stats = self._catch_up()
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

    def _catch_up(self) -> tuple[int, dict]:
        start_id, head, publication = self.start_id, self.head, self.publication
        if publication is None:
            raise RuntimeError(f"the run's version {head} names no publication")
        # A version is whole only in a view reloaded after it was announced; a new start may also
        # have removed versions an older view still shows.
        if (
            start_id != self.applied[1]
            or self.view_head is None
            or head > self.view_head
        ):
            self.reload()
            self.view_head = head
        try:
            committed = self.engine.committed(publication["plan_digest"])
            chain = publication_chain(self.versions, publication, committed)
            if not chain:
                raise RuntimeError(f"its weights are already at {committed}")
            for earlier in chain[:-1]:
                self.engine.apply(earlier, None)
            stats = self.engine.apply(chain[-1], head)
        except Exception as error:
            # A leaving engine's version fails because its engine stopped: nothing to report.
            if not self.exiting:
                with contextlib.suppress(Exception):
                    self.control.fail(head)
            raise RuntimeError(
                f"version {head} from {publication['manifest']}: {error}"
            ) from error
        self.applied = (head, start_id)
        logger.info("version %d: %d publications, %s", head, len(chain), stats)
        return head, stats


def _raise_on_exit(reply: dict) -> dict:
    if "exit" in reply:
        raise RuntimeError(f"the control service ended this engine: {reply['exit']}")
    return reply
