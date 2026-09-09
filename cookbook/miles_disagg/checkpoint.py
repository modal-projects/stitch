"""Copy immutable local checkpoints to a dedicated Volume off the training path.

Each host commits its own files and then a receipt. The first host publishes a
completion manifest only after every receipt is durable. Readers trust that
manifest, never the presence of a directory, HF marker, or Megatron tracker.
Background workers do not use torch.distributed or reload a mounted Volume.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path == PurePosixPath(".")
    ):
        raise ValueError(f"invalid checkpoint-relative path: {value!r}")
    return path.as_posix()


@dataclass(frozen=True)
class CheckpointUpload:
    local_root: Path
    run_id: str
    attempt_id: str
    iteration: int
    host_rank: int
    host_ranks: tuple[int, ...]
    hf_directory: str | None
    required_files: tuple[str, ...] = ()

    def __post_init__(self):
        for value in (self.run_id, self.attempt_id):
            if "/" in relative_path(value):
                raise ValueError("run and attempt IDs must be single path components")
        if (
            self.iteration < 0
            or not self.host_ranks
            or self.host_rank not in self.host_ranks
        ):
            raise ValueError("invalid checkpoint iteration or host membership")
        if tuple(sorted(set(self.host_ranks))) != self.host_ranks:
            raise ValueError("host ranks must be sorted and unique")
        if self.hf_directory is not None:
            relative_path(self.hf_directory)
        for path in self.required_files:
            relative_path(path)

    @property
    def root(self) -> str:
        return (
            f"{self.run_id}/attempts/{self.attempt_id}/snapshots/{self.iteration:07d}"
        )

    @property
    def checkpoint_directory(self) -> str:
        return f"checkpoints/iter_{self.iteration:07d}"

    @property
    def rollout_file(self) -> str:
        return f"checkpoints/rollout/global_dataset_state_dict_{self.iteration}.pt"

    @property
    def manifest_path(self) -> str:
        return f"{self.run_id}/completed/{self.iteration:07d}-{self.attempt_id}.json"

    def receipt_path(self, rank: int) -> str:
        return f"{self.root}/receipts/{self.iteration:07d}/{rank}.json"

    def files(self) -> list[Path]:
        directories = [self.checkpoint_directory]
        if self.hf_directory:
            directories.append(self.hf_directory)
        paths = [
            path
            for directory in directories
            for path in (self.local_root / directory).rglob("*")
            if path.is_file() and path.name != ".complete"
        ]
        rollout = self.local_root / self.rollout_file
        if rollout.is_file():
            paths.append(rollout)
        for path in paths:
            if path.is_symlink():
                raise ValueError(
                    f"checkpoint payload must contain regular files: {path}"
                )
        return sorted(paths)


class CheckpointUploader:
    """Persist one immutable snapshot at a time per trainer host."""

    def __init__(
        self,
        mount: Path,
        volume: Any,
        *,
        bytes_per_second: int = 256 * 1024**2,
        timeout_seconds: float = 6 * 60 * 60,
        poll_seconds: float = 2,
    ):
        if bytes_per_second < 0 or timeout_seconds <= 0 or poll_seconds <= 0:
            raise ValueError("invalid checkpoint upload limits")
        self.mount = mount
        self.volume = volume
        self.bytes_per_second = bytes_per_second
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="checkpoint-upload"
        )
        self._future: Future | None = None

    def check(self) -> None:
        if self._future is not None and self._future.done():
            try:
                self._future.result()
            except Exception as exc:
                raise RuntimeError(f"checkpoint upload failed: {exc}") from exc
            self._future = None

    @property
    def pending(self) -> int:
        self.check()
        return int(self._future is not None)

    @property
    def has_capacity(self) -> bool:
        return self.pending == 0

    def submit(self, upload: CheckpointUpload) -> None:
        if not self.has_capacity:
            raise RuntimeError("checkpoint uploader has no capacity")
        self._future = self._executor.submit(self._upload, upload)
        self._event("queued", upload)

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def _event(self, phase: str, upload: CheckpointUpload, **fields) -> None:
        logger.info(
            "CHECKPOINT %s",
            json.dumps(
                {
                    "phase": phase,
                    "iteration": upload.iteration,
                    "host_rank": upload.host_rank,
                    **fields,
                },
                sort_keys=True,
            ),
        )

    def _read_json(self, path: str) -> dict | None:
        try:
            return json.loads(b"".join(self.volume.read_file(path)))
        except FileNotFoundError:
            return None

    def _write_json(self, path: str, value: dict) -> None:
        destination = self.mount / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, sort_keys=True))
        temporary.replace(destination)

    def _upload(self, upload: CheckpointUpload) -> None:
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        copied = 0
        reported = started
        files = {}
        try:
            for source in upload.files():
                name = source.relative_to(upload.local_root).as_posix()
                destination = self.mount / upload.root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                checksum = hashlib.sha256()
                size = 0
                with source.open("rb") as reader, destination.open("wb") as writer:
                    while True:
                        chunk_started = time.monotonic()
                        chunk = reader.read(4 * 1024**2)
                        if not chunk:
                            break
                        writer.write(chunk)
                        checksum.update(chunk)
                        copied += len(chunk)
                        size += len(chunk)
                        now = time.monotonic()
                        if now > deadline:
                            raise TimeoutError(
                                "checkpoint copy exceeded upload deadline"
                            )
                        if self.bytes_per_second:
                            delay = len(chunk) / self.bytes_per_second - (
                                now - chunk_started
                            )
                            if delay > 0:
                                time.sleep(delay)
                        if now - reported >= 10:
                            self._event(
                                "copying",
                                upload,
                                bytes=copied,
                                age_seconds=now - started,
                            )
                            reported = now
                files[name] = {"size": size, "sha256": checksum.hexdigest()}
            self.volume.commit()
            receipt = {
                "iteration": upload.iteration,
                "attempt_id": upload.attempt_id,
                "host_rank": upload.host_rank,
                "host_ranks": list(upload.host_ranks),
                "files": files,
                "required_files": list(upload.required_files),
            }
            self._write_json(upload.receipt_path(upload.host_rank), receipt)
            self.volume.commit()
            self._event("host_committed", upload, bytes=copied)
            if upload.host_rank == upload.host_ranks[0]:
                self._complete(upload, deadline)
            while self._read_json(upload.manifest_path) is None:
                self._wait(deadline)
            shutil.rmtree(upload.local_root / upload.checkpoint_directory)
            if upload.hf_directory:
                shutil.rmtree(
                    upload.local_root / upload.hf_directory, ignore_errors=True
                )
            (upload.local_root / upload.rollout_file).unlink(missing_ok=True)
            self._event(
                "durable", upload, bytes=copied, seconds=time.monotonic() - started
            )
        except Exception:
            self._event("failed", upload, retained_local_root=str(upload.local_root))
            raise

    def _wait(self, deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for every checkpoint host to commit")
        time.sleep(min(self.poll_seconds, max(0, deadline - time.monotonic())))

    def _complete(self, upload: CheckpointUpload, deadline: float) -> None:
        receipts = {}
        while len(receipts) != len(upload.host_ranks):
            for rank in upload.host_ranks:
                if rank not in receipts:
                    receipt = self._read_json(upload.receipt_path(rank))
                    if receipt is not None:
                        if (
                            receipt["iteration"],
                            receipt["attempt_id"],
                            receipt["host_rank"],
                            tuple(receipt["host_ranks"]),
                        ) != (
                            upload.iteration,
                            upload.attempt_id,
                            rank,
                            upload.host_ranks,
                        ):
                            raise ValueError(
                                "checkpoint receipt belongs to a different snapshot"
                            )
                        receipts[rank] = receipt
            if len(receipts) != len(upload.host_ranks):
                self._wait(deadline)
        files = {}
        required = set()
        for receipt in receipts.values():
            for path, entry in receipt["files"].items():
                if path in files:
                    raise ValueError(
                        f"checkpoint file has multiple host writers: {path}"
                    )
                files[path] = entry
            required.update(receipt["required_files"])
        missing = required - files.keys()
        if missing:
            raise ValueError(
                f"checkpoint is missing referenced files: {sorted(missing)}"
            )
        root = self.mount / upload.root
        # These are compatibility metadata for Megatron/HF loaders. The manifest
        # below is the sole publication boundary used by cookbook discovery.
        if upload.hf_directory:
            (root / upload.hf_directory / ".complete").touch()
        tracker = root / "checkpoints/latest_checkpointed_iteration.txt"
        tracker.parent.mkdir(parents=True, exist_ok=True)
        tracker.write_text(str(upload.iteration))
        self.volume.commit()
        self._write_json(
            upload.manifest_path,
            {
                "schema_version": 1,
                "run_id": upload.run_id,
                "completed_at_ns": time.time_ns(),
                "attempt_id": upload.attempt_id,
                "iteration": upload.iteration,
                "version": upload.iteration + 1,
                "host_ranks": list(upload.host_ranks),
                "checkpoint_root": f"{upload.root}/checkpoints",
                "hf_directory": f"{upload.root}/{upload.hf_directory}"
                if upload.hf_directory
                else None,
                "files": files,
            },
        )
        self.volume.commit()
