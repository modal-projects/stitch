"""Shared trainer hooks for publishing HF weight updates and routing rollout requests.

Trainer integrations point their lifecycle callbacks at this module. Each hook reads the
run coordinates from the trainer's argument namespace and composes the Stitch core with a
``ModalVolumeStore`` and ``ModalFlashPool``.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from stitch.pools.modal_flash import ModalFlashPool
from stitch.publish import claim_run, constrain_request, publish_version
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.types import PointerRewind, VersionRef

from . import process

logger = logging.getLogger(__name__)


def sample_affinity_key(sample: Any) -> str | None:
    """Return one stable routing key for a rollout trajectory or GRPO group."""
    group_index = getattr(sample, "group_index", None)
    if group_index is not None:
        return f"group-{group_index}"
    for name in ("routing_key", "session_id"):
        value = getattr(sample, name, None)
        if value is not None:
            return str(value)
    return None


# ── publish ────────────────────────────────────────────────────────────────────
def commit_and_wake(args: Any, published_dir: str, rollout_engines: Any = None) -> None:
    """Bridge the framework's disk-delta publish to the stitch store. The framework fires this
    at each durability boundary: a version dir (``weight_vNNNNNN``, holding the HF index) and —
    at baseline/pointer commit — the run dir. Every rank commits its mounted writes; after all
    commits succeed, rank 0 refreshes and validates the complete checkpoint before advancing
    the pointer. Keying on the dir name (not on reading an index) keeps run-dir calls a clean
    no-op, not a missing-file crash."""
    del rollout_engines
    store = _store(args)
    if not Path(published_dir).name.startswith("weight_v"):
        # Baseline cleanup is performed by rank 0 only, so a normal mounted
        # Volume commit is authoritative here.
        store.commit()
        return

    publication_state: tuple[bool, str | None] = (False, None)
    if process.dist_rank() in (None, 0):
        try:
            target = VersionRef.parse(f"{_run_id(args)}/{Path(published_dir).name}")
            current = store.read_pointer()
            publication_state = (
                current is not None
                and current.run_id == target.run_id
                and current.version >= target.version,
                None,
            )
        except Exception:  # noqa: BLE001
            publication_state = (False, f"rank 0:\n{traceback.format_exc()}")
    publication_states = process.dist_all_gather_object(publication_state)
    errors = [error for _, error in publication_states if error is not None]
    if errors:
        raise RuntimeError(
            "checkpoint publication state check failed:\n" + "\n".join(errors)
        )
    if any(already_published for already_published, _ in publication_states):
        logger.warning("%s is already published; leaving it immutable", published_dir)
        return

    commit_error = None
    try:
        store.commit()
    except Exception:  # noqa: BLE001
        commit_error = f"rank {process.dist_rank()}:\n{traceback.format_exc()}"
    _raise_distributed_failures("checkpoint commit", commit_error)

    publish_error = None
    if process.dist_rank() in (None, 0):
        try:
            store.refresh()
            publish_version(store, _pool(args), published_dir, run_id=_run_id(args))
        except PointerRewind:
            # A same-run republish (e.g. a retried step) — drop it rather than serve stale.
            logger.warning(
                "publish of %s would rewind latest; dropping",
                published_dir,
                exc_info=True,
            )
        except Exception:  # noqa: BLE001
            publish_error = f"rank 0:\n{traceback.format_exc()}"
    _raise_distributed_failures("checkpoint publication", publish_error)


def _raise_distributed_failures(phase: str, local_error: str | None) -> None:
    errors = [
        error
        for error in process.dist_all_gather_object(local_error)
        if error is not None
    ]
    if errors:
        raise RuntimeError(f"{phase} failed:\n" + "\n".join(errors))


def claim_pool(args: Any) -> None:
    """Launch hook (rank 0): reset every replica to base before the first publish, so a
    cold or finished-run-warm pool starts this run clean."""
    if process.dist_rank() not in (None, 0):
        return
    claim_run(_store(args), _pool(args), _run_id(args))


# ── staleness-gated rollout requests ────────────────────────────────────────────
async def gated_rollout_request_hook(
    args: Any, sample: Any, request: dict[str, Any]
) -> None:
    """Pin each request to a bounded-staleness version, so a too-stale replica returns a
    retryable 409 (nudging it to sync) instead of the trainer spending rollout compute on
    weights beyond its lag bound."""
    payload, headers = request["payload"], dict(request.get("headers") or {})
    mode = str(getattr(args, "rollout_request_weight_version_mode", "min"))
    affinity = str(
        getattr(args, "rollout_session_affinity_header", "x-session-affinity")
    )

    latest = exact = None
    lag = 0
    if mode != "none":
        floor = await _latest.get(args)
        lag = int(getattr(args, "rollout_request_weight_version_lag", 0))
        if mode == "exact":
            exact = max(0, floor - lag)
        else:
            latest = floor
    constrain_request(
        payload,
        headers,
        latest=latest,
        lag=lag,
        exact=exact,
        session_id=sample_affinity_key(sample),
        affinity_header=affinity,
    )
    request["headers"] = headers
    request["max_retries"] = int(
        getattr(args, "rollout_request_retry_attempts", request.get("max_retries", 60))
    )
    request["retry_sleep"] = float(
        getattr(args, "rollout_request_retry_sleep", request.get("retry_sleep", 1.0))
    )


class _CachedPointer:
    """TTL-cached durable ``latest`` version for request admission.

    Publication runs on trainer ranks while request hooks may run in separate processes,
    so their Volume mounts can have independent visibility. Co-located request processes
    share a file-locked refresh lease to reload at most once per TTL.
    """

    def __init__(self) -> None:
        self._version = 0
        self._at = -1e9
        self._store: ModalVolumeStore | None = None
        self._lock = asyncio.Lock()

    async def get(self, args: Any, ttl: float = 2.0) -> int:
        store = self._store
        root = Path(_transport_root(args))
        run_id = _run_id(args)
        volume = getattr(args, "experiment_volume_name", None)
        if (
            store is None
            or store.root != root
            or store.run_id != run_id
            or store.volume_name != (volume or None)
        ):
            store = self._store = _store(args)
            self._version = 0
            self._at = -1e9
        now = time.monotonic()
        if now - self._at < ttl:
            return self._version
        async with self._lock:
            if time.monotonic() - self._at < ttl:
                return self._version
            try:
                await asyncio.to_thread(_refresh_mount, store, ttl)
                pointer = store.read_pointer()
                self._version = pointer.version if pointer else 0
            except Exception:  # noqa: BLE001
                logger.warning(
                    "gate: could not read latest; using cached %s",
                    self._version,
                    exc_info=True,
                )
            finally:
                self._at = time.monotonic()
        return self._version


def _refresh_mount(store: ModalVolumeStore, ttl: float) -> None:
    """Refresh a mounted Volume at most once per container and TTL.

    A trainer may run many request-hook processes in one container. They share the mount
    and this local ``flock`` lease, so one process reloads while the others reuse the
    resulting view. Separate containers get their own lease and refresh independently.
    """
    if not store.volume_name:
        store.refresh()
        return

    key = hashlib.sha256(
        f"{store.volume_name}\0{store.root}\0{store.run_id}".encode()
    ).hexdigest()[:16]
    lease_dir = Path("/tmp/stitch-volume-refresh")
    lease_dir.mkdir(parents=True, exist_ok=True)
    with (lease_dir / f"{key}.lock").open("a+", encoding="utf-8") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        lease.seek(0)
        try:
            refreshed_at = float(lease.read() or "-inf")
        except ValueError:
            refreshed_at = float("-inf")
        if time.monotonic() - refreshed_at < ttl:
            return
        try:
            store.refresh()
        finally:
            lease.seek(0)
            lease.truncate()
            lease.write(str(time.monotonic()))
            lease.flush()


_latest = _CachedPointer()


# ── args → run coordinates ───────────────────────────────────────────────────────
def _store(args: Any) -> ModalVolumeStore:
    volume = getattr(args, "experiment_volume_name", None)
    return ModalVolumeStore(
        _transport_root(args),
        volume_name=volume or None,
        run_id=_run_id(args),
    )


def _pool(args: Any) -> ModalFlashPool:
    app = getattr(args, "rollout_modal_flash_app_name", None)
    if not app:
        raise ValueError("rollout_modal_flash_app_name is required")
    cls = getattr(args, "rollout_modal_flash_server_cls_name", "Server")
    return ModalFlashPool(app, cls)


def _transport_root(args: Any) -> str:
    # The framework owns <run>/updates; Stitch owns <run>/latest.
    write_dir = getattr(args, "update_weight_disk_dir", None)
    if not write_dir:
        raise ValueError("update_weight_disk_dir is required")
    write_dir = Path(write_dir)
    if write_dir.name != "updates":
        raise ValueError(
            f"update_weight_disk_dir must end in /updates: path={str(write_dir)!r}"
        )
    return str(write_dir.parent)


def _run_id(args: Any) -> str:
    run_id = getattr(args, "run_id", None)
    if not run_id:
        raise ValueError(
            "run_id is required in the trainer hook arguments — it is the run's fence token"
        )
    return str(run_id)
