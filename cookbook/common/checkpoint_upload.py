"""Background upload of node-local trainer checkpoints to the run Volume.

The trainer writes its checkpoint exactly as before, each rank its own files, but under
a node-local root that mirrors the run directory instead of onto the Volume mount. One
uploader per container copies what its host wrote to the same paths on the Volume. It
uses the Volume API rather than the mount: a mount copy would hold files open there, and
every publish reloads the mount.

Only saves write under the local root: run-scoped files that outlive a save, such as
the event log and the rollout dumps, are pointed at the run directory instead. Every
save rewrites the Megatron tracker, so new files in a step with no new tracker on any
host mean another writer, and the step fails rather than take its files.

Readers trust two kinds of marker file: the Megatron tracker and the HF ``.complete``
markers. Markers go up last, only once every host's data files from the same save are
durable, so a marker still vouches only for complete bytes. A marker's content is taken
when its save is claimed, since the trainer rewrites the tracker in place at every save.
Before a re-save's publish, the ``.complete`` markers it will replace are removed from
the Volume, so no reader pairs an old marker with a partly overwritten export. The
tracker is never removed: it only ever moves forward. Directories are left in place;
the next save writes into them.

Every method is collective: all trainer ranks call it the same number of times, in the
same order. Upload threads never use torch.distributed.
"""

from __future__ import annotations

import io
import logging
import shutil
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from stitch.publisher import TrainerComms

logger = logging.getLogger(__name__)

TRACKER_NAME = "latest_checkpointed_iteration.txt"
COMPLETE_NAME = ".complete"
# Files whose presence readers take to mean "everything this names is complete".
MARKER_NAMES = frozenset({TRACKER_NAME, COMPLETE_NAME})


class CheckpointUploader:
    """Copy one trainer host's node-local checkpoint files to the Volume.

    ``local_root`` mirrors ``volume_root``, the run directory inside ``volume``. Only the
    container leader uploads; every rank joins the collectives.
    """

    def __init__(
        self,
        local_root: Path,
        *,
        volume: Any,
        volume_root: PurePosixPath,
        comms: TrainerComms,
        streams: int = 4,
        retry_delay_seconds: float = 10.0,
        stall_floor_seconds: float = 600.0,
        min_bytes_per_second: float = 20e6,
    ) -> None:
        self._local_root = local_root
        self._retry_delay_seconds = retry_delay_seconds
        # A batch upload has no deadline of its own, and one that stops making
        # progress would otherwise hold this host's later saves, and every marker,
        # forever. An attempt slower than this rate is abandoned for a fresh one.
        self._stall_floor_seconds = stall_floor_seconds
        self._min_bytes_per_second = min_bytes_per_second
        self._volume = volume
        self._volume_root = volume_root
        self._comms = comms
        self._streams = streams
        self._is_leader = comms.is_host_leader()
        self._step = 0
        # A path is new when its identity changes: the tracker is rewritten in place.
        self._claimed: dict[Path, tuple[int, int, int]] = {}
        # Leader state; batch ids are step numbers, identical on every rank.
        self._data: deque[tuple[int, Future]] = deque()
        self._held_markers: dict[int, list[tuple[Path, bytes]]] = {}
        self._marker_uploads: deque[Future] = deque()
        self._error: str | None = None
        self._data_executor: ThreadPoolExecutor | None = None
        self._marker_executor: ThreadPoolExecutor | None = None
        if self._is_leader:
            self._data_executor = ThreadPoolExecutor(1, "checkpoint-upload-data")
            # Separate, so markers never wait behind the next save's data.
            self._marker_executor = ThreadPoolExecutor(1, "checkpoint-upload-markers")
            local_root.mkdir(parents=True, exist_ok=True)
            # Sibling directories belong to earlier trainer attempts, which are gone.
            for stale in local_root.parent.iterdir():
                if stale != local_root and stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)

    def step(self) -> None:
        """Before a publish: start uploading the files written since the last call and
        publish the markers whose data every host has made durable. Collective; raises
        on every rank when new files arrived without a save."""
        self._step += 1
        data, markers = self._claim() if self._data_executor is not None else ([], [])
        claims = self._comms.all_gather_object(
            (
                [str(path) for path, _ in data + markers],
                any(path.name == TRACKER_NAME for path, _ in markers),
            )
        )
        if not any(saved for _, saved in claims):
            if stray := sorted(path for paths, _ in claims for path in paths):
                raise RuntimeError(
                    "files appeared under the local checkpoint root without a save, so "
                    "something other than a save writes there: " + ", ".join(stray)
                )
        if self._data_executor is not None:
            if markers:
                # Must finish before this publish can advance a pointer to the new save.
                self._retract(path for path, _ in markers)
                self._held_markers[self._step] = markers
            if data:
                self._data.append(
                    (self._step, self._data_executor.submit(self._upload_data, data))
                )
                logger.info(
                    "checkpoint upload: step %d queued %d files (%.1f GB)",
                    self._step,
                    len(data),
                    sum(size for _, size in data) / 1e9,
                )
        self._sync()

    def drain(self, poll_seconds: float = 5.0) -> None:
        """Wait until every host's files and markers are durable. Collective."""
        while self._sync():
            time.sleep(poll_seconds)

    def _sync(self) -> bool:
        """Gather every host's progress, raise any host's failure on every rank, and
        queue the markers every host is ready for. Returns whether work is pending."""
        if self._data_executor is not None:
            self._reap()
        states = self._comms.all_gather_object(
            (self._is_leader, self._durable_through(), self._pending(), self._error)
        )
        errors = [state[3] for state in states if state[3] is not None]
        if errors:
            raise RuntimeError("checkpoint upload failed:\n" + "\n".join(errors))
        durable = min(state[1] for state in states if state[0])
        if self._marker_executor is not None:
            for batch in sorted(self._held_markers):
                if batch > durable:
                    break
                self._marker_uploads.append(
                    self._marker_executor.submit(
                        self._upload_markers, self._held_markers.pop(batch)
                    )
                )
        return any(state[2] for state in states if state[0])

    def _durable_through(self) -> int:
        """The last step whose data files, and all earlier ones, are on the Volume."""
        return self._data[0][0] - 1 if self._data else self._step

    def _pending(self) -> int:
        return len(self._data) + len(self._held_markers) + len(self._marker_uploads)

    def _reap(self) -> None:
        while self._data and self._data[0][1].done():
            self._record(self._data.popleft()[1])
        while self._marker_uploads and self._marker_uploads[0].done():
            self._record(self._marker_uploads.popleft())

    def _record(self, future: Future) -> None:
        if (error := future.exception()) is not None and self._error is None:
            self._error = f"rank {self._comms.rank()}:\n" + "".join(
                traceback.format_exception(error)
            )

    def _claim(self) -> tuple[list[tuple[Path, int]], list[tuple[Path, bytes]]]:
        """Files written since the last call: data as (path, size), markers as (path,
        content). The trainer saves synchronously before it publishes, so each is closed."""
        data, markers = [], []
        for path in sorted(self._local_root.rglob("*")):
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
                identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if self._claimed.get(path) == identity:
                    continue
                if path.name in MARKER_NAMES:
                    markers.append((path, path.read_bytes()))
                else:
                    data.append((path, stat.st_size))
            except FileNotFoundError:
                continue  # an earlier upload removed it after this scan listed it
            self._claimed[path] = identity
        return data, markers

    def _retract(self, markers) -> None:
        """Remove the Volume copies of ``.complete`` markers a re-save will replace."""
        for marker in markers:
            if marker.name != COMPLETE_NAME:
                continue
            try:
                self._volume.remove_file(self._volume_path(marker))
            except FileNotFoundError:
                continue

    def _upload_data(self, files: Sequence[tuple[Path, int]]) -> None:
        started = time.monotonic()
        groups = _balanced(files, self._streams, key=lambda file: file[1])
        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            for future in [
                pool.submit(self._put, [(path, path) for path, _ in group])
                for group in groups
            ]:
                future.result()
        for path, _ in files:
            path.unlink()
        logger.info(
            "checkpoint upload: %d data files (%.1f GB) durable in %.0fs",
            len(files),
            sum(size for _, size in files) / 1e9,
            time.monotonic() - started,
        )

    def _upload_markers(self, markers: Sequence[tuple[Path, bytes]]) -> None:
        self._put(list(markers))
        # The tracker stays: Megatron owns and rewrites it; each save claims it anew.
        for path, _ in markers:
            if path.name == COMPLETE_NAME:
                path.unlink(missing_ok=True)
        logger.info(
            "checkpoint upload: markers durable: %s",
            ", ".join(self._volume_path(path) for path, _ in markers),
        )

    def _put(
        self, files: Sequence[tuple[Path, Path | bytes]], attempts: int = 3
    ) -> None:
        """Upload ``files``, each a local file or in-memory content, in one batch.

        A batch is idempotent, so a failed attempt is retried, and so is one that
        outlives its deadline: it is abandoned, not cancelled, and if it ever
        finishes it writes the same bytes to the same paths. Abandoned attempts die
        with this attempt's container, before any re-save, so a late one can at most
        move the tracker back to an earlier save that is itself complete."""
        nbytes = sum(
            len(source) if isinstance(source, bytes) else source.stat().st_size
            for _, source in files
        )
        deadline = self._stall_floor_seconds + nbytes / self._min_bytes_per_second
        for attempt in range(1, attempts + 1):
            failure: list[BaseException] = []
            worker = threading.Thread(
                target=self._batch, args=(files, failure), daemon=True
            )
            worker.start()
            worker.join(deadline)
            if not worker.is_alive() and not failure:
                return
            error = (
                failure[0]
                if failure
                else TimeoutError(
                    f"batch of {len(files)} files ({nbytes / 1e9:.1f} GB) still "
                    f"uploading after {deadline:.0f}s"
                )
            )
            if attempt == attempts:
                raise error
            logger.warning(
                "checkpoint upload: batch of %d files failed (attempt %d/%d): %s",
                len(files),
                attempt,
                attempts,
                error,
            )
            time.sleep(self._retry_delay_seconds * attempt)

    def _batch(
        self, files: Sequence[tuple[Path, Path | bytes]], failure: list[BaseException]
    ) -> None:
        try:
            with self._volume.batch_upload(force=True) as upload:
                for path, source in files:
                    upload.put_file(
                        io.BytesIO(source)
                        if isinstance(source, bytes)
                        else str(source),
                        self._volume_path(path),
                    )
        except BaseException as error:  # noqa: BLE001 — reported to the waiting caller
            failure.append(error)

    def _volume_path(self, path: Path) -> str:
        return (self._volume_root / path.relative_to(self._local_root)).as_posix()


def _balanced(items: Sequence, streams: int, *, key: Callable) -> list[list]:
    """Split ``items`` into at most ``streams`` groups of similar total ``key``."""
    groups: list[list] = [[] for _ in range(max(1, min(streams, len(items))))]
    loads = [0] * len(groups)
    for item in sorted(items, key=key, reverse=True):
        lightest = loads.index(min(loads))
        groups[lightest].append(item)
        loads[lightest] += key(item)
    return [group for group in groups if group]
