"""Shared framework-hook logic (miles and slime both write the same HF-delta layout to
a Modal Volume and wake a Modal Flash pool, so the logic is one place).

Each run config points its hook paths straight at this module (e.g.
``custom_delta_pre_push_path = "cookbook.common.hooks.commit_and_wake"``). Each hook reads
the run's coordinates off the trainer's ``args`` namespace (the framework ``setattr``s its
``custom_config_path`` onto it) and calls the stitch core against a ModalVolumeStore +
ModalFlashPool.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from stitch.pools.modal_flash import ModalFlashPool
from stitch.publish import claim_run, constrain_request, publish_version
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.types import PointerRewind

from . import process

logger = logging.getLogger(__name__)


# ── publish ────────────────────────────────────────────────────────────────────
def commit_and_wake(args: Any, published_dir: str, rollout_engines: Any = None) -> None:
    """Bridge the framework's disk-delta publish to the stitch store. The framework fires this
    at each durability boundary: a version dir (``weight_vNNNNNN``, holding the HF index) and —
    at baseline/pointer commit — the run dir. Every rank flushes its writes; rank 0 publishes
    only when handed a version dir. Keying on the dir name (not on reading an index) keeps the
    run-dir calls a clean no-op, not a missing-file crash."""
    del rollout_engines
    store = _store(args)
    store.commit()
    if process.dist_rank() not in (None, 0) or not Path(published_dir).name.startswith("weight_v"):
        return
    try:
        publish_version(store, _pool(args), published_dir, run_id=_run_id(args))
    except PointerRewind:
        # A same-run republish (e.g. a retried step) — drop it rather than serve stale.
        logger.warning("publish of %s would rewind latest; dropping", published_dir, exc_info=True)


def claim_pool(args: Any) -> None:
    """Launch hook (rank 0): reset every replica to base before the first publish, so a
    cold or finished-run-warm pool starts this run clean."""
    if process.dist_rank() not in (None, 0):
        return
    claim_run(_store(args), _pool(args), _run_id(args))


# ── staleness-gated rollout requests ────────────────────────────────────────────
async def gated_rollout_request_hook(args: Any, sample: Any, request: dict[str, Any]) -> None:
    """Pin each request to a bounded-staleness version, so a too-stale replica returns a
    retryable 409 (nudging it to sync) instead of the trainer spending rollout compute on
    weights beyond its lag bound."""
    payload, headers = request["payload"], dict(request.get("headers") or {})
    mode = str(getattr(args, "rollout_request_weight_version_mode", "min"))
    affinity = str(getattr(args, "rollout_session_affinity_header", "x-session-affinity"))
    session_id = getattr(sample, "session_id", None)

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
        payload, headers, latest=latest, lag=lag, exact=exact,
        session_id=session_id, affinity_header=affinity,
    )
    request["headers"] = headers
    request["max_retries"] = int(getattr(args, "rollout_request_retry_attempts", request.get("max_retries", 60)))
    request["retry_sleep"] = float(getattr(args, "rollout_request_retry_sleep", request.get("retry_sleep", 1.0)))


class _CachedPointer:
    """TTL-cached ``latest`` version. The per-request hook gets no rollout id, so the
    staleness floor comes from the published pointer (already advanced by the publish
    hook), cached with a Volume reload so it isn't reloaded once per request."""

    def __init__(self) -> None:
        self._version = 0
        self._at = -1e9
        self._store: ModalVolumeStore | None = None

    async def get(self, args: Any, ttl: float = 2.0) -> int:
        store = self._store
        if store is None:
            store = self._store = _store(args)
        now = time.monotonic()
        if now - self._at >= ttl:
            self._at = now
            try:
                await asyncio.to_thread(store.refresh)  # reload is blocking; keep the loop free
                pointer = store.read_pointer()
                self._version = pointer.version if pointer else 0
            except Exception:  # noqa: BLE001
                logger.warning("gate: could not read latest; using cached %s", self._version, exc_info=True)
        return self._version


_latest = _CachedPointer()


# ── args → run coordinates ───────────────────────────────────────────────────────
def _store(args: Any) -> ModalVolumeStore:
    volume = getattr(args, "update_weight_delta_volume_name", None) or os.environ.get("DELTA_VOLUME_NAME")
    return ModalVolumeStore(_transport_root(args), volume_name=volume or None)


def _pool(args: Any) -> ModalFlashPool:
    app = getattr(args, "rollout_modal_flash_app_name", None) or os.environ.get("DELTA_APP_NAME")
    cls = getattr(args, "rollout_modal_flash_server_cls_name", None) or os.environ.get("DELTA_SERVER_CLS_NAME", "Server")
    return ModalFlashPool(app, cls)


def _transport_root(args: Any) -> str:
    # The trainer writes version dirs under <root>/<run_id>; the Store is rooted at <root>.
    write_dir = getattr(args, "update_weight_disk_dir", None) or os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin")
    return str(Path(write_dir).parent)


def _run_id(args: Any) -> str:
    run_id = getattr(args, "run_id", None)
    if not run_id:
        raise ValueError("run_id is required (pass it via custom_config_path) — it is the run's fence token")
    return str(run_id)


