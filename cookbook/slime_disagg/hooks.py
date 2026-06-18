"""Modal publish hooks for the slime_disagg bulletin-board example.

These layer the Modal-specific concerns — Volume durability and best-effort
rollout-pool wake — on top of stitch's provider-agnostic bulletin-board publish,
so the core trainer adapter (``stitch.trainers.slime``) stays free of any
provider import.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from stitch.providers.modal import commit_volume, discover_flash_targets, wake_targets
from stitch.trainers import slime as slime_trainer


logger = logging.getLogger(__name__)


def commit_delta_volume(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """SLIME ``custom_delta_pre_push_path`` hook: commit the Modal Volume so the
    just-written delta files are durable before the manifest is published."""
    del rollout_engines
    commit_volume(_volume_name(args))
    logger.info("Committed delta Volume for %s", version_dir)


def publish_delta_version(
    args: Any,
    version_dir: str,
    files: list[str],
    weight_version: str | int,
    rollout_engines: list[Any],
) -> list[Any]:
    """SLIME ``custom_delta_publish_path`` hook: publish to the bulletin board
    (core), commit the Volume, then best-effort wake the rollout pool."""
    result = slime_trainer.publish_delta_version(
        args, version_dir, files, weight_version, rollout_engines
    )
    commit_volume(_volume_name(args))
    # Waking warm containers is a best-effort latency optimization: sidecars
    # self-sync when a version-pinned request rejects, so a transient Modal
    # control-plane error here must not kill the training step — latest.json
    # already points at the new version.
    try:
        app_name = getattr(args, "rollout_modal_flash_app_name", None) or os.environ["SLIME_DELTA_APP_NAME"]
        cls_name = getattr(args, "rollout_modal_flash_server_cls_name", None) or os.getenv(
            "SLIME_DELTA_SERVER_CLS_NAME", "Server"
        )
        wake_targets(discover_flash_targets(app_name=app_name, cls_name=cls_name), int(weight_version))
    except Exception:  # noqa: BLE001
        logger.warning(
            "Best-effort rollout wake failed for version %s; sidecars will self-sync on demand",
            weight_version,
            exc_info=True,
        )
    return result


def _volume_name(args: Any) -> str:
    return str(getattr(args, "update_weight_delta_volume_name", None) or os.environ["DELTA_VOLUME_NAME"])
