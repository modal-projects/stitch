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
from pathlib import Path
from typing import Any

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import parse_weight_identity
from stitch.providers.modal import commit_volume, discover_flash_targets, wake_targets


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

    # Rank 0 owns the monotonic `latest` pointer and writes it before committing,
    # so the committed bulletin is self-consistent for the poll/startup path
    # (slime's own latest write is post-hook and uncommitted). Every rank commits
    # its node's shards.
    if version is not None and rank in (None, 0):
        FilesystemBulletinBoard(_bulletin_root(args), layout="slime").write_latest(version)
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
    return str(
        getattr(args, "update_weight_disk_dir", None)
        or os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin")
    )
