"""Stitch's code in each engine container: keeps the engine at the version the control service names."""

import contextlib
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
# Beside an in-place adapter's files: {"version": its count}, written by the trainer.
ADAPTER_VERSION_FILE = "version.json"


class Engine(Protocol):
    """What Stitch needs from an inference engine. `version` is the version it serves, -1 before the run's first."""

    version: int

    def start(self) -> None: ...

    def warm_up(self) -> None: ...

    def watch(self) -> None:
        """From here on, the engine exiting or failing its health checks ends the container."""

    def committed(self, plan_digest: str) -> tuple[str | None, int]:
        """The (stream, version) its full weights end at, (None, 0) for the checkpoint it loaded; `plan_digest` is
        the run's checkpoint plan, which its weights must share."""

    def apply(self, publication: dict, version: int | None) -> dict | None:
        """Prepares one publication while serving, then applies it inside a pause and labels the weights `version`
        (None until the last of a catch-up). A failure leaves the weights partly updated. It leaves none of the
        publication's files open: Modal reloads a Volume only with none of its files open."""

    def update_adapter(self, name: str, path: str, version: int) -> None:
        """Serves adapter `name` from `path` at `version` from its next step; a name not loaded only learns `path`."""

    def remove_adapter(self, name: str) -> None: ...

    def load(self) -> dict:
        """Requests the engine runs and queues now, and the slots: how many it runs at once."""

    def abort_all(self) -> None:
        """Ends every request in flight."""

    def stop(self) -> None: ...


class ControlClient:
    """The control service as one engine calls it: a call unanswered in twice the wait is sent again."""

    def __init__(self, service: "modal.cls.Obj", engine_id: str, clocks: Clocks):
        self.service, self.engine_id, self.clocks = service, engine_id, clocks

    def open(self, version: int, start_id: str | None) -> dict:
        return self._call(self.service.open, version, start_id)

    def next(
        self,
        version: int,
        seen: int,
        start_id: str | None,
        service_id: str | None,
        seen_changes: int,
    ) -> dict:
        return self._call(
            self.service.next, version, seen, start_id, service_id, seen_changes
        )

    def served(self, version: int, stats: dict, start_id: str) -> dict:
        return self._call(self.service.served, version, stats, start_id)

    def synced(
        self, version: int, changes: int, start_id: str | None, service_id: str
    ) -> dict:
        return self._call(self.service.synced, version, changes, start_id, service_id)

    def fail(self, version: int) -> None:
        self._call(self.service.fail, version)

    def report_load(self, load: dict) -> None:
        # Not awaited: the next report follows in 30 s.
        self.service.report_load.spawn(self.engine_id, load)

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


class AdapterStage:
    """Local copies, at the paths requests name, of the directories the trainer links under the run's `root`."""

    def __init__(
        self, source: Path, target: Path, root: Path, reload: Callable[[], None]
    ):
        self.source, self.target, self.root, self.reload = source, target, root, reload

    def current(self, paths: list[str] | None, links: bool) -> dict[str, str | None]:
        """Each of `paths` (every link with None) mapped to its directory on the Volume now, None if gone."""
        self.reload()
        if paths is None:
            linked = self._linked()
            return linked if links else {target: target for target in linked.values()}
        if links:
            return {path: self._target(path) for path in paths}
        return {path: path if self._source(path).is_dir() else None for path in paths}

    def copy(self, paths: Iterable[str]) -> set[str]:
        """The `paths` held locally afterwards: a directory gone from the Volume meanwhile is not."""
        # Each copy mostly waits on the Volume's reads, and a burst of saves stages together.
        with ThreadPoolExecutor(max_workers=16) as pool:
            paths = list(paths)
            return {
                path
                for path, held in zip(paths, pool.map(self._copy, paths), strict=True)
                if held
            }

    def remove(self, paths: Iterable[str]) -> None:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(partial(shutil.rmtree, ignore_errors=True), paths))

    def version(self, path: str) -> int:
        return json.loads((Path(path) / ADAPTER_VERSION_FILE).read_text())["version"]

    def name(self, link: str) -> str:
        return str(Path(link).relative_to(self.root))

    def _linked(self) -> dict[str, str]:
        linked, folders = {}, [self._source(str(self.root))]
        while folders:
            for entry in os.scandir(folders.pop()):
                # A dot entry is a link's directory or the trainer's own bookkeeping, never a name.
                if entry.name.startswith("."):
                    continue
                if entry.is_symlink():
                    if (target := self._target(self._local(entry.path))) is not None:
                        linked[self._local(entry.path)] = target
                elif entry.is_dir():
                    folders.append(Path(entry.path))
        return linked

    def _target(self, link: str) -> str | None:
        source = self._source(link)
        if not source.is_symlink():
            return None
        target = source.parent / os.readlink(source)
        return self._local(os.path.normpath(target)) if target.is_dir() else None

    def _copy(self, path: str) -> bool:
        if Path(path).exists():
            return True
        try:
            shutil.copytree(self._source(path), path)
        except FileNotFoundError:
            shutil.rmtree(path, ignore_errors=True)
            return False
        return True

    def _source(self, path: str) -> Path:
        return self.source / Path(path).relative_to(self.target)

    def _local(self, source: str) -> str:
        return str(self.target / Path(source).relative_to(self.source))


class EngineContainer:
    """One engine's Modal container: applies the run's publications in order, up to the head's."""

    def __init__(
        self,
        engine: Engine,
        control: ControlClient,
        versions: Path,
        reload: Callable[[], None],
        adapter_stage: AdapterStage | None = None,
        adapters_in_place: bool = False,
    ):
        self.engine, self.control, self.versions = engine, control, versions
        self.reload, self.adapter_stage = reload, adapter_stage
        self.adapters_in_place = adapters_in_place
        self.head = engine.version
        # The head's publication, as the control service last named it.
        self.publication: dict | None = None
        # The run's start as last heard; the version this engine serves and the start it knew
        # when applying it.
        self.start_id: str | None = None
        self.applied: tuple[int, str | None] = (engine.version, None)
        self.view_head: int | None = None
        # Each adapter held, by its link in place or its directory otherwise, and the local directory it serves.
        self.adapters: dict[str, str] = {}
        self.service_id: str | None = None
        self.adapter_changes = 0
        # Where adapters changed since the last sync; None: anywhere.
        self.changed_adapter_paths: list[str] | None = []
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
        threading.Thread(target=self.report, daemon=True).start()

    def report(self) -> None:
        while not self.exiting:
            try:
                self.control.report_load(self.engine.load())
            except Exception:
                logger.warning("could not report this engine's load", exc_info=True)
            time.sleep(30)

    def exit(self) -> None:
        self.exiting = True
        with contextlib.suppress(Exception):
            self.control.leave()
        self.engine.stop()

    def open(self) -> None:
        """Catches up to the head and syncs the adapters until the control service opens this engine on them."""
        synced = None
        while True:
            reply = _raise_on_exit(self.control.open(*self.applied))
            self.start_id = reply["start_id"] or self.start_id
            self.head, self.publication = reply["head"], reply["publication"]
            self.service_id = reply["service_id"]
            self.adapter_changes = reply["adapter_changes"]
            if reply["open"]:
                if synced == (self.service_id, self.adapter_changes):
                    return
                self._sync_adapters(None, self.adapter_changes, self.service_id)
                synced = (self.service_id, self.adapter_changes)
            elif self.head > self.engine.version:
                self._catch_up()

    def listen(self) -> None:
        """Keeps one call to the control service open, also while applying; wakes `follow`."""
        while True:
            version, start_id = self.applied
            reply = _raise_on_exit(
                self.control.next(
                    version, self.head, start_id, self.service_id, self.adapter_changes
                )
            )
            with self.newer:
                self.head, self.publication = reply["head"], reply["publication"]
                self.start_id = reply["start_id"] or self.start_id
                changed = reply["changed_adapter_paths"]
                if changed is None or self.changed_adapter_paths is None:
                    self.changed_adapter_paths = None
                else:
                    self.changed_adapter_paths += changed
                self.service_id = reply["service_id"]
                self.adapter_changes = reply["adapter_changes"]
                self.newer.notify_all()

    def follow(self) -> None:
        """Applies the publications up to the newest version and reports it served; syncs changed adapters."""
        while True:
            with self.newer:
                self.newer.wait_for(
                    lambda: (
                        self.head > self.engine.version
                        or self.changed_adapter_paths != []
                    )
                )
                changed, self.changed_adapter_paths = self.changed_adapter_paths, []
                through, service_id = self.adapter_changes, self.service_id
            if self.head > self.engine.version:
                version, stats = self._catch_up()
                _raise_on_exit(self.control.served(version, stats, self.applied[1]))
            if changed != []:
                self._sync_adapters(changed, through, service_id)

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

    def _sync_adapters(
        self, changed: list[str] | None, through: int, service_id: str
    ) -> None:
        """Brings the adapters at `changed`, or every one with None, to the Volume's; reports `through` synced."""
        if self.adapter_stage is not None:
            current = self.adapter_stage.current(changed, links=self.adapters_in_place)
            if changed is None:
                current |= {key: None for key in self.adapters.keys() - current.keys()}
            moved = {
                key: path
                for key, path in current.items()
                if self.adapters.get(key) != path
            }
            held = self.adapter_stage.copy(path for path in moved.values() if path)
            moved = {
                key: path for key, path in moved.items() if path in held or path is None
            }
            if self.adapters_in_place:
                with ThreadPoolExecutor(max_workers=16) as pool:
                    list(pool.map(self._switch_adapter, moved.items()))
            # After the switch: until then the engine may still read the replaced directory.
            self.adapter_stage.remove(
                self.adapters[key] for key in moved if key in self.adapters
            )
            for key, path in moved.items():
                if path is None:
                    self.adapters.pop(key, None)
                else:
                    self.adapters[key] = path
        _raise_on_exit(
            self.control.synced(self.applied[0], through, self.applied[1], service_id)
        )

    def _switch_adapter(self, change: tuple[str, str | None]) -> None:
        link, path = change
        name = self.adapter_stage.name(link)
        if path is None:
            self.engine.remove_adapter(name)
        else:
            self.engine.update_adapter(name, path, self.adapter_stage.version(path))

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
            chain = self._chain(publication)
            for earlier in chain[:-1]:
                self.engine.apply(earlier, None)
            stats = self.engine.apply(chain[-1], head)
        except Exception as error:
            # A leaving engine's version fails because its engine stopped: nothing to report.
            if not self.exiting:
                with contextlib.suppress(Exception):
                    self.control.fail(head)
            source = (
                publication["path"]
                if publication.get("kind") == "lora"
                else publication["manifest"]
            )
            raise RuntimeError(f"version {head} from {source}: {error}") from error
        self.applied = (head, start_id)
        logger.info("version %d: %d publications, %s", head, len(chain), stats)
        return head, stats

    def _chain(self, publication: dict) -> list[dict]:
        """The publications that bring the engine to the head's, oldest first."""
        if publication.get("kind") == "lora":
            # Each LoRA version is the whole adapter: the head's alone, whatever the engine holds.
            return [
                publication | {"lora_path": str(self.versions / publication["path"])}
            ]
        committed = self.engine.committed(publication["plan_digest"])
        chain = publication_chain(self.versions, publication, committed)
        if not chain:
            raise RuntimeError(f"its weights are already at {committed}")
        return chain


def _raise_on_exit(reply: dict) -> dict:
    if "exit" in reply:
        raise RuntimeError(f"the control service ended this engine: {reply['exit']}")
    return reply
