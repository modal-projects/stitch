"""The distributed publication protocol — how a trainer's ranks converge on one
published version.

This is the modal- and framework-agnostic core each integration's publish hook
delegates to (the cookbook's is ``cookbook.common.hooks.commit_and_wake``).
``TrainerComms`` abstracts how ranks rendezvous, the Store abstracts where bytes
live, and the Pool wake stays best-effort. A single-process trainer gets the
default comms for free; a distributed one subclasses them over its runtime
(torch.distributed, MPI, ...) and hands them in with its Store.

The protocol composes ``publish.py``'s single-writer helpers rather than
replacing them: rank 0 runs the same ``publish_version`` / ``claim_run`` a
one-host trainer calls directly.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

from stitch.pools.base import Pool
from stitch.publish import claim_run, publish_version, wake_pool
from stitch.stores.base import Store
from stitch.stores.s3 import S3Store
from stitch.types import WEIGHT_PREFIX, VersionRef, decide_pointer_move

logger = logging.getLogger(__name__)


class TrainerComms:
    """The trainer's communicator — how distributed ranks rendezvous during a publish.

    Each method's default is the single-process case: one writer, singleton
    gathers. A distributed trainer subclasses all three over its runtime. Every
    method must be callable on every rank with the same published_dir — the
    protocol gathers results across ranks so all of them fail or succeed together.
    """

    def rank(self) -> int | None:
        """This process's global rank, or None off the distributed path (single
        process: treated as rank 0)."""
        return None

    def all_gather_object(self, value: Any) -> list[Any]:
        """Gather one small control-plane value from every rank."""
        return [value]

    def is_host_leader(self) -> bool:
        """Whether this rank performs the once-per-host-or-mount store side effects
        (its host's shard commit/upload), for runtimes where several ranks share one
        host's files."""
        return True


class Publisher:
    """One run's publication actor: claim the run's boot checkpoint, then publish
    each framework-written weight update across the trainer's ranks."""

    def __init__(
        self,
        store: Store,
        pool: Pool | None = None,
        *,
        run_id: str,
        comms: TrainerComms | None = None,
    ) -> None:
        self._store = store
        self._pool = pool
        self._run_id = run_id
        self._comms = comms if comms is not None else TrainerComms()

    def claim(self, *, boot_version: int = 0) -> None:
        """Start the run at the checkpoint every replica already serves (rank 0's
        job; the single-writer ``claim_run`` does the move)."""
        if self._comms.rank() not in (None, 0):
            return
        claim_run(self._store, self._pool, self._run_id, boot_version=boot_version)

    def publish(self, published_dir: str) -> None:
        """Publish one framework-written disk update as the run's next served version.

        A ``weight_vNNNNNN`` directory is a version publish; any other directory is
        the framework's run directory — a durability boundary for a mounted store
        (every rank commits) and a no-op for an upload store. The version publish
        splits by backend: with a shared mount every host leader commits its mount
        and rank 0 publishes from the refreshed view; with per-host uploads each
        leader uploads its node-local files and rank 0 commits ``latest`` only after
        the gathered receipts verify, so a replica never follows ``latest`` to
        incomplete bytes."""
        name = Path(published_dir).name
        if not name.startswith(WEIGHT_PREFIX):
            if not isinstance(self._store, S3Store):
                self._store.commit()
            return

        target = VersionRef.parse(f"{self._run_id}/{name}")
        already_published, expected = self._pointer_snapshot(target)
        if already_published:
            logger.warning(
                "%s is already published; leaving it immutable", published_dir
            )
            return
        decide_pointer_move(expected, target)  # every rank rejects a rewind identically

        if isinstance(self._store, S3Store):
            self._publish_uploaded(target, published_dir, expected)
        else:
            self._publish_mounted(published_dir)

    def _pointer_snapshot(self, target: VersionRef) -> tuple[bool, VersionRef | None]:
        """Rank 0 reads ``latest`` and every rank agrees on the gathered result.

        Returns whether ``target`` was already published and the pointer rank 0 saw
        — the expected predecessor of rank 0's later conditional advance."""
        checked = self._comms.rank() in (None, 0)
        current = None
        already_published = False
        error = None
        if checked:
            try:
                current = self._store.read_pointer()
                already_published = (
                    current is not None
                    and current.run_id == target.run_id
                    and current.version >= target.version
                )
            except Exception:  # noqa: BLE001
                error = f"rank 0:\n{traceback.format_exc()}"
        states = self._comms.all_gather_object(
            (checked, already_published, current, error)
        )
        errors = [state[3] for state in states if state[3] is not None]
        if errors:
            raise RuntimeError(
                "checkpoint publication state check failed:\n" + "\n".join(errors)
            )
        rank_zero = [state for state in states if state[0]]
        if len(rank_zero) != 1:
            raise RuntimeError(
                "checkpoint publication state check did not identify exactly one rank 0"
            )
        _, already, expected, _ = rank_zero[0]
        return already, expected

    def _publish_mounted(self, published_dir: str) -> None:
        """Commit every host's mount, then publish from rank 0's refreshed view."""
        commit_error = None
        if self._comms.is_host_leader():
            try:
                self._store.commit()
            except Exception:  # noqa: BLE001
                commit_error = f"rank {self._comms.rank()}:\n{traceback.format_exc()}"
        self._raise_gathered_failures("checkpoint commit", commit_error)

        publish_error = None
        if self._comms.rank() in (None, 0):
            try:
                self._store.refresh()
                publish_version(
                    self._store, self._pool, published_dir, run_id=self._run_id
                )
            except Exception:  # noqa: BLE001
                publish_error = f"rank 0:\n{traceback.format_exc()}"
        self._raise_gathered_failures("checkpoint publication", publish_error)

    def _publish_uploaded(
        self,
        target: VersionRef,
        published_dir: str,
        expected: VersionRef | None,
    ) -> None:
        """Upload once per host, verify the gathered receipts, and commit ``latest`` last."""
        receipt = None
        upload_error = None
        if self._comms.is_host_leader():
            try:
                receipt = self._store.upload_version_files(target, published_dir)
            except Exception:  # noqa: BLE001
                upload_error = f"rank {self._comms.rank()}:\n{traceback.format_exc()}"

        results = self._comms.all_gather_object((receipt, upload_error))
        upload_errors = [error for _, error in results if error is not None]
        if upload_errors:
            raise RuntimeError("checkpoint upload failed:\n" + "\n".join(upload_errors))
        receipts = [receipt for receipt, _ in results if receipt is not None]
        if not receipts:
            raise RuntimeError("checkpoint upload produced no host receipts")

        publish_error = None
        if self._comms.rank() in (None, 0):
            try:
                manifest = self._store.verify_version(target, receipts)
                self._store.compare_and_advance_pointer(expected, target)
                wake_pool(self._pool, target)
                logger.info(
                    "published %s: kind=%s files=%d hosts=%d",
                    target.identity,
                    manifest.kind.value,
                    len(manifest.files),
                    len(receipts),
                )
            except Exception:  # noqa: BLE001
                publish_error = f"rank 0:\n{traceback.format_exc()}"
        self._raise_gathered_failures("checkpoint publication", publish_error)

    def _raise_gathered_failures(self, phase: str, local_error: str | None) -> None:
        errors = [
            error
            for error in self._comms.all_gather_object(local_error)
            if error is not None
        ]
        if errors:
            raise RuntimeError(f"{phase} failed:\n" + "\n".join(errors))
