"""Slime trainer adapter for disaggregated rollout weight sync."""

from __future__ import annotations

import logging
import os
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import Artifact, VersionManifest, read_latest
from stitch.providers.modal import commit_volume, discover_flash_targets, wake_targets


logger = logging.getLogger(__name__)


def commit_delta_volume(args: Any, version_dir: str, rollout_engines: list[Any]) -> None:
    """Slime ``custom_delta_pre_push_path`` hook for Modal Volume durability."""
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
    """Slime ``custom_delta_publish_path`` hook for publish-only disk deltas."""
    del rollout_engines
    version = int(weight_version)
    root = Path(_bulletin_root(args))
    version_path = Path(version_dir)
    sorted_files = sorted(files)

    manifest = VersionManifest(
        version=version,
        base_version=version - 1,
        backend="sparse_delta",
        load_format="delta",
        transition_files=sorted_files,
        artifacts=[Artifact(kind="transition", path=path) for path in sorted_files],
        created_at=time.time(),
        run_id=getattr(args, "run_id", None),
        base_model=getattr(args, "hf_checkpoint", None),
        metadata={"trainer": "slime", "transport": "disk"},
    )
    FilesystemBulletinBoard(root).publish_manifest(manifest, version_path=version_path)
    commit_volume(_volume_name(args))
    logger.info("Published sparse delta version %s with %d file(s)", version, len(sorted_files))

    # Waking warm containers is a best-effort latency optimization: sidecars
    # self-sync when a version-pinned request rejects. A transient Modal
    # control-plane error here must not kill the training step, especially
    # since latest.json already points at the new version.
    try:
        app_name = getattr(args, "rollout_modal_flash_app_name", None) or os.environ["SLIME_DELTA_APP_NAME"]
        cls_name = getattr(args, "rollout_modal_flash_server_cls_name", None) or os.getenv(
            "SLIME_DELTA_SERVER_CLS_NAME", "Server"
        )
        wake_targets(discover_flash_targets(app_name=app_name, cls_name=cls_name), version)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Best-effort rollout wake failed for version %s; sidecars will self-sync on demand",
            version,
            exc_info=True,
        )
    return []


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
):
    """Run Slime's default SGLang rollout with a publish-version policy."""
    from slime.rollout import sglang_rollout as upstream_rollout
    from slime.rollout.sglang_rollout import rollout_weight_version_context

    assert args.rollout_global_dataset
    target_version = rollout_target_weight_version(args, rollout_id, evaluation=evaluation)
    logger.info(
        "Disaggregated %s rollout_id=%s target_weight_version=%s",
        "eval" if evaluation else "train",
        rollout_id,
        target_version,
    )

    with rollout_weight_version_context(args, target_version):
        if evaluation:
            output, _ = upstream_rollout.run(upstream_rollout.eval_rollout(args, rollout_id))
            return output

        output, aborted_samples = upstream_rollout.run(
            upstream_rollout.generate_rollout_async(args, rollout_id, data_source.get_samples)
        )
        if aborted_samples:
            data_source.add_samples(aborted_samples)
        return output


def rollout_target_weight_version(args: Namespace, rollout_id: int, evaluation: bool = False) -> int:
    if not evaluation:
        return int(rollout_id)

    root = getattr(args, "update_weight_delta_root", None)
    if root is None:
        delta_dir = getattr(args, "update_weight_delta_dir", None)
        if delta_dir is None:
            return int(rollout_id)
        root = Path(delta_dir).parent
    return read_latest(root)


def _volume_name(args: Any) -> str:
    return str(getattr(args, "update_weight_delta_volume_name", None) or os.environ["DELTA_VOLUME_NAME"])


def _bulletin_root(args: Any) -> str:
    return str(getattr(args, "update_weight_delta_root", None) or Path(args.update_weight_delta_dir).parent)
