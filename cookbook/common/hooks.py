"""Shared trainer hooks for publishing HF weight updates and routing rollout requests.

Trainer integrations point their lifecycle callbacks at this module. Each hook reads the
run coordinates from the trainer's argument namespace and wires the configured Store and
``ModalFlashPool`` into :class:`stitch.publisher.Publisher`, the modal-agnostic core
that owns the distributed publish protocol.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from stitch.pools.base import Pool
from stitch.pools.modal_flash import ModalFlashFleet, ModalFlashPool
from stitch.publish import constrain_request
from stitch.publisher import Publisher, TrainerComms
from stitch.stores.base import Store

from . import process, storage

logger = logging.getLogger(__name__)


class _TorchComms(TrainerComms):
    """The trainer's torch.distributed comms, via ``common.process`` helpers.

    Off the distributed path (no initialized process group) each helper degrades
    to the single-process default, so a single-host dev run needs no wiring.
    """

    def rank(self) -> int | None:
        return process.dist_rank()

    def all_gather_object(self, value: Any) -> list[Any]:
        return process.dist_all_gather_object(value)

    def is_host_leader(self) -> bool:
        return process.dist_is_container_leader()


# ── publish ────────────────────────────────────────────────────────────────────
def commit_and_wake(args: Any, published_dir: str, rollout_engines: Any = None) -> None:
    """Publish one framework-written disk update and wake rollout replicas.

    Volume trainers commit each host's shared mount before rank 0 publishes.
    S3 trainers instead upload from node-local disk once per host, gather small
    receipts, and let rank 0 verify the complete S3 version before advancing
    ``latest``. The framework also invokes this hook for its run directory;
    that is a Volume durability boundary and an S3 no-op.
    """
    del rollout_engines
    _publisher(args).publish(published_dir)


def claim_pool(args: Any, *, boot_version: int = 0) -> None:
    """Launch hook (rank 0): identify the checkpoint already served by every replica
    before the first publish."""
    _publisher(args).claim(boot_version=boot_version)


def _publisher(args: Any) -> Publisher:
    return Publisher(
        _store(args),
        _pool(args),
        run_id=_run_id(args),
        comms=_TorchComms(),
    )


# ── staleness-gated rollout requests ────────────────────────────────────────────
async def gated_rollout_request_hook(
    hook_args: dict[str, Any], _context: Any, request: dict[str, Any]
) -> None:
    """Pin each request to a bounded-staleness version, so a too-stale replica returns a
    retryable 409 (nudging it to sync) instead of the trainer spending rollout compute on
    weights beyond its lag bound."""
    args = SimpleNamespace(**hook_args)
    payload, headers = request["payload"], dict(request.get("headers") or {})
    mode = str(getattr(args, "rollout_request_weight_version_mode", "min"))

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
    )
    request["headers"] = headers
    request["max_attempts"] = int(
        getattr(
            args,
            "rollout_request_max_attempts",
            request.get("max_attempts", 1),
        )
    )
    request["retry_interval"] = float(
        getattr(
            args,
            "rollout_request_retry_interval",
            request.get("retry_interval", 1.0),
        )
    )


class _CachedPointer:
    """TTL-cached ``latest`` version from the trainer's configured store.

    The request gate reads the trainer host's mounted view, which the rank-zero
    publisher updates directly. Reloading that mount can fail while framework
    processes hold files open; cross-host refresh belongs to rollout-replica
    reconciliation, and S3 has no mounted snapshot to refresh.
    """

    def __init__(self) -> None:
        self._version = 0
        self._at = -1e9
        self._store: Store | None = None
        self._store_key: tuple[str | None, ...] | None = None

    async def get(self, args: Any, ttl: float = 2.0) -> int:
        store = self._store
        store_key = _store_key(args)
        if store is None or self._store_key != store_key:
            store = self._store = _store(args)
            self._store_key = store_key
            self._version = 0
            self._at = -1e9
        now = time.monotonic()
        if now - self._at >= ttl:
            try:
                pointer = store.read_pointer()
                self._version = pointer.version if pointer else 0
            except Exception:  # noqa: BLE001
                logger.warning(
                    "gate: could not read latest; using cached %s",
                    self._version,
                    exc_info=True,
                )
            self._at = time.monotonic()
        return self._version


_latest = _CachedPointer()


# ── args → run coordinates ───────────────────────────────────────────────────────
def _store(args: Any) -> Store:
    return storage.create_store(
        str(getattr(args, "stitch_store_backend", storage.MODAL_VOLUME)),
        local_root=_transport_root(args),
        run_id=_run_id(args),
        volume_name=getattr(args, "experiment_volume_name", None) or None,
        s3_root=getattr(args, "stitch_s3_root", None) or None,
        s3_endpoint_url=getattr(args, "stitch_s3_endpoint_url", None) or None,
    )


def _store_key(args: Any) -> tuple[str | None, ...]:
    return (
        str(getattr(args, "stitch_store_backend", storage.MODAL_VOLUME)),
        _transport_root(args),
        _run_id(args),
        getattr(args, "experiment_volume_name", None) or None,
        getattr(args, "stitch_s3_root", None) or None,
        getattr(args, "stitch_s3_endpoint_url", None) or None,
    )


def _pool(args: Any) -> Pool:
    app = getattr(args, "rollout_modal_flash_app_name", None)
    if not app:
        raise ValueError("rollout_modal_flash_app_name is required")
    classes = getattr(args, "rollout_modal_flash_server_cls_names", None)
    if classes:
        classes = list(classes)
        if len(classes) == 1:
            return ModalFlashPool(app, classes[0])
        router = getattr(args, "rollout_modal_flash_router_function", None)
        if not router:
            raise ValueError(
                "rollout_modal_flash_router_function is required for multiple pools"
            )
        return ModalFlashFleet(app, classes, gateway_function=router)
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
