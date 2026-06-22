"""Modal publish hook for the slime_disagg bulletin-board example.

The canonical path: slime publish-only writes ``weight_v{N}/`` + a ``latest``
pointer straight to the Modal Volume bulletin board (rename works on a Volume),
and the elastic Flash pool of SGLang servers + stitch sidecars self-syncs from
it. This hook only layers on the Modal-specific concerns — Volume durability and
a best-effort rollout-pool wake. None of the standalone hot-load shim machinery
(front door, HTTP ``/hot_load``, auth headers, S3-copy) belongs here.
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


def commit_and_wake(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """SLIME ``custom_delta_pre_push_path`` hook (publish-only, bulletin board).

    slime has written ``weight_v{N}/`` to the Modal Volume. Advance the committed
    ``latest`` pointer, commit the Volume so the rollout pool's ``reload`` sees
    the new version, then best-effort wake the Flash pool. The sidecars self-sync
    (wake RPC, periodic poll, startup), so a missed wake only costs latency.
    """
    del rollout_engines
    version = parse_weight_identity(Path(version_dir).name)
    rank = _distributed_rank()

    # Rank 0 owns the `latest` pointer and writes it before committing, so the
    # committed bulletin is self-consistent for the poll/startup path (slime's own
    # latest write is post-hook and uncommitted). Every rank commits its shards.
    # The pointer lives at the transport root (the Volume mount) and is
    # self-identifying — `<run_id>/weight_vN` — while slime wrote the version dir
    # under the run partition (update_weight_disk_dir = <root>/<run_id>), so a new
    # run is a forward move of the pointer, never a colliding rewind.
    if version is not None and rank in (None, 0):
        FilesystemBulletinBoard(_transport_root(args), layout="slime").write_latest(
            _run_id(args), version
        )
    commit_volume(_volume_name(args))

    if version is None or rank not in (None, 0):
        # version is None on the baseline call (disk-dir root); only rank 0 wakes.
        return
    # Waking warm containers is a best-effort latency optimization: a transient
    # Modal control-plane error must not kill the training step — `latest` is
    # already committed and sidecars self-sync on their next poll.
    try:
        app_name = getattr(args, "rollout_modal_flash_app_name", None) or os.environ["SLIME_DELTA_APP_NAME"]
        cls_name = getattr(args, "rollout_modal_flash_server_cls_name", None) or os.getenv(
            "SLIME_DELTA_SERVER_CLS_NAME", "Server"
        )
        wake_targets(discover_flash_targets(app_name=app_name, cls_name=cls_name), version)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Best-effort rollout wake failed for version %s; sidecars will self-sync",
            version,
            exc_info=True,
        )


def _distributed_rank() -> int | None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        return None
    return None


def _volume_name(args: Any) -> str:
    return str(getattr(args, "update_weight_delta_volume_name", None) or os.environ["DELTA_VOLUME_NAME"])


def _bulletin_root(args: Any) -> str:
    """Where slime writes version dirs: ``<transport_root>/<run_id>``."""
    return str(
        getattr(args, "update_weight_disk_dir", None)
        or os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin")
    )


def _transport_root(args: Any) -> str:
    """The Volume mount root that holds the canonical ``latest`` pointer and that
    the sidecar boards are rooted at — the parent of the per-run write dir."""
    return str(Path(_bulletin_root(args)).parent)


def _run_id(args: Any) -> str:
    """The run partition (chain identity). Passed explicitly via custom_config,
    falling back to the basename of the per-run write dir."""
    return str(getattr(args, "run_id", None) or Path(_bulletin_root(args)).name)


# ── M3: staleness-gated rollout requests ──────────────────────────────────────

# Cache of the latest published version for the gate. The per-request hook gets no
# rollout_id, so the staleness floor is derived out-of-band from the published
# `latest` pointer (the publish hook already advanced + committed it). TTL-cached
# with a Volume reload so a (possibly cross-node) rollout actor sees rank-0's
# committed pointer without a Volume reload per request.
_latest_cache: dict[str, Any] = {"version": 0, "run_id": None, "at": -1e9, "board": None}


def _gate_board(args: Any) -> FilesystemBulletinBoard:
    vol = getattr(args, "update_weight_delta_volume_name", None) or os.environ.get("DELTA_VOLUME_NAME")
    refresh = volume_reloader(vol) if vol else None
    # The pointer lives at the transport root (the mount), not the per-run write dir.
    return FilesystemBulletinBoard(_transport_root(args), refresh=refresh, layout="slime")


async def _latest_published(args: Any, ttl: float = 2.0) -> int:
    now = time.monotonic()
    if _latest_cache["board"] is None:
        _latest_cache["board"] = _gate_board(args)
    if now - _latest_cache["at"] >= ttl:
        _latest_cache["at"] = now  # claim the refresh slot before awaiting -> no reload herd
        try:
            await _latest_cache["board"].refresh()
            run_id, version = _latest_cache["board"].read_latest()
            # Staleness floor is per-run (version restarts at 1 each run); on a run
            # change adopt the new run's pointer immediately so the floor isn't
            # pinned to a finished run's higher version number.
            _latest_cache["run_id"] = run_id
            _latest_cache["version"] = int(version)
        except Exception:  # noqa: BLE001
            logger.warning(
                "gate: could not read latest published version; using cached %s",
                _latest_cache["version"],
                exc_info=True,
            )
    return _latest_cache["version"]


async def gated_rollout_request_hook(args: Any, sample: Any, request: dict[str, Any]) -> None:
    """SLIME ``custom_rollout_request_hook_path`` (M3): gate each rollout on
    ``weight_version - k`` so unusable (too-stale) rollouts are never generated.

    The per-request hook receives no rollout_id, so the staleness floor is derived
    out-of-band from the published ``latest`` pointer (see ``_latest_published``).
    A request pinned to ``min_required_version = latest - lag`` is admitted only by
    a replica within ``lag`` versions of the newest weights; a lagging replica
    returns a retryable 409 (which also nudges it to sync forward), so the trainer
    never spends rollout compute on weights staler than its bound. ``min`` mode
    (not ``exact``) lets the request cross in_place commits without being quiesced.
    """
    mode = str(getattr(args, "rollout_request_weight_version_mode", "min"))
    if mode != "none":
        latest = await _latest_published(args)
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
