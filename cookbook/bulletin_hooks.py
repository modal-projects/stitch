"""Shared Modal Volume bulletin-board hooks for publish + rollout gating.

Two hooks that any trainer (slime, miles, ...) plugs into via its
``custom_delta_pre_push_path`` and ``custom_rollout_request_hook_path``:

- :func:`commit_and_wake` — advance the ``latest`` pointer, commit the Volume,
  and best-effort wake the Flash rollout pool.
- :func:`gated_rollout_request_hook` — pin each rollout request to a bounded-
  staleness weight version so unusable (too-stale) rollouts are never generated.

Both hooks read their config off the trainer's ``args`` namespace (the trainer's
``--custom-config-path`` setattr's every key onto ``args``). The only
trainer-specific axis is the env-var fallback for the Flash app / class name;
callers pass those as ``app_name_env`` / ``cls_name_env``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import parse_weight_identity
from stitch.providers.modal import commit_volume, discover_flash_targets, volume_reloader, wake_targets


logger = logging.getLogger(__name__)


# ── Publish hook ──────────────────────────────────────────────────────────────


def commit_and_wake(
    args: Any,
    version_dir: str,
    rollout_engines: list[Any],
    *,
    app_name_env: str,
    cls_name_env: str,
) -> None:
    """Trainer ``custom_delta_pre_push_path`` hook (publish-only, bulletin board).

    The trainer has written ``weight_v{N}/`` to the Modal Volume. Advance the
    committed ``latest`` pointer, commit the Volume so the rollout pool's
    ``reload`` sees the new version, then best-effort wake the Flash pool. The
    sidecars self-sync (wake RPC, periodic poll, startup), so a missed wake only
    costs latency.

    ``app_name_env`` / ``cls_name_env`` are the env-var names the trainer uses
    for the Flash app and server class (e.g. ``"SLIME_DELTA_APP_NAME"``).
    """
    del rollout_engines
    version = parse_weight_identity(Path(version_dir).name)
    rank = distributed_rank()

    # Rank 0 owns the `latest` pointer and writes it before committing, so the
    # committed bulletin is self-consistent for the poll/startup path. The pointer
    # lives at the transport root (the Volume mount) and is self-identifying —
    # `<run_id>/weight_vN` — while the trainer wrote the version dir under the run
    # partition (update_weight_disk_dir = <root>/<run_id>), so a new run is a
    # forward move of the pointer, never a colliding rewind.
    if version is not None and rank in (None, 0):
        FilesystemBulletinBoard(_transport_root(args), layout="slime").write_latest(
            _run_id(args), version
        )
    commit_volume(_volume_name(args))

    if version is None or rank not in (None, 0):
        return
    # Waking warm containers is a best-effort latency optimization: a transient
    # Modal control-plane error must not kill the training step — `latest` is
    # already committed and sidecars self-sync on their next poll.
    try:
        app_name = getattr(args, "rollout_modal_flash_app_name", None) or os.environ[app_name_env]
        cls_name = getattr(args, "rollout_modal_flash_server_cls_name", None) or os.getenv(
            cls_name_env, "Server"
        )
        wake_targets(discover_flash_targets(app_name=app_name, cls_name=cls_name), version)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Best-effort rollout wake failed for version %s; sidecars will self-sync",
            version,
            exc_info=True,
        )


# ── Staleness-gated rollout requests ──────────────────────────────────────────


class CachedLatestPointer:
    """TTL-cached ``(run_id, version)`` from the bulletin board's ``latest`` pointer.

    The per-request hook gets no rollout_id, so the staleness floor is derived
    out-of-band from the published ``latest`` pointer (the publish hook already
    advanced + committed it). TTL-cached with a Volume reload so a (possibly
    cross-node) rollout actor sees rank-0's committed pointer without a Volume
    reload per request.
    """

    def __init__(self) -> None:
        self.version: int = 0
        self.run_id: str | None = None
        self._refreshed_at: float = -1e9
        self._board: FilesystemBulletinBoard | None = None

    async def get(self, args: Any, ttl: float = 2.0) -> int:
        now = time.monotonic()
        if self._board is None:
            self._board = _gate_board(args)
        if now - self._refreshed_at >= ttl:
            self._refreshed_at = now
            try:
                await self._board.refresh()
                run_id, version = self._board.read_latest()
                # Staleness floor is per-run (version restarts at 1 each run); on a
                # run change adopt the new run's pointer immediately so the floor
                # isn't pinned to a finished run's higher version number.
                self.run_id = run_id
                self.version = int(version)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "gate: could not read latest published version; using cached %s",
                    self.version,
                    exc_info=True,
                )
        return self.version


_latest_cache = CachedLatestPointer()


async def gated_rollout_request_hook(args: Any, sample: Any, request: dict[str, Any]) -> None:
    """Trainer ``custom_rollout_request_hook_path``: gate each rollout on
    ``weight_version - k`` so unusable (too-stale) rollouts are never generated.

    A request pinned to ``min_required_version = latest - lag`` is admitted only
    by a replica within ``lag`` versions of the newest weights; a lagging replica
    returns a retryable 409 (which also nudges it to sync forward), so the trainer
    never spends rollout compute on weights staler than its bound. ``min`` mode
    (not ``exact``) lets the request cross in_place commits without being quiesced.
    """
    mode = str(getattr(args, "rollout_request_weight_version_mode", "min"))
    if mode != "none":
        latest = await _latest_cache.get(args)
        lag = int(getattr(args, "rollout_request_weight_version_lag", 0))
        target = max(0, latest - lag)
        key = "exact_version" if mode == "exact" else "min_required_version"
        request["payload"]["weight_version"] = {key: target}

    request["max_retries"] = int(getattr(args, "rollout_request_retry_attempts", request.get("max_retries", 60)))
    request["retry_sleep"] = float(getattr(args, "rollout_request_retry_sleep", request.get("retry_sleep", 1.0)))

    session_id = getattr(sample, "session_id", None)
    if session_id:
        header = str(getattr(args, "rollout_session_affinity_header", "x-session-affinity"))
        headers = dict(request.get("headers") or {})
        headers.setdefault(header, session_id)
        request["headers"] = headers


# ── Shared helpers ────────────────────────────────────────────────────────────


def distributed_rank() -> int | None:
    """Return the torch distributed rank, or None if not initialized."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        return None
    return None


def _volume_name(args: Any) -> str:
    return str(getattr(args, "update_weight_delta_volume_name", None) or os.environ["DELTA_VOLUME_NAME"])


def bulletin_root(args: Any) -> str:
    """Where the trainer writes version dirs: ``<transport_root>/<run_id>``."""
    return str(
        getattr(args, "update_weight_disk_dir", None)
        or os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin")
    )


def _transport_root(args: Any) -> str:
    """The Volume mount root that holds the canonical ``latest`` pointer and that
    the sidecar boards are rooted at — the parent of the per-run write dir."""
    return str(Path(bulletin_root(args)).parent)


def _run_id(args: Any) -> str:
    """The run partition (chain identity). Passed explicitly via custom_config,
    falling back to the basename of the per-run write dir."""
    return str(getattr(args, "run_id", None) or Path(bulletin_root(args)).name)


def _gate_board(args: Any) -> FilesystemBulletinBoard:
    vol = getattr(args, "update_weight_delta_volume_name", None) or os.environ.get("DELTA_VOLUME_NAME")
    refresh = volume_reloader(vol) if vol else None
    return FilesystemBulletinBoard(_transport_root(args), refresh=refresh, layout="slime")
