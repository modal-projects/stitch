"""Slime trainer adapter for disaggregated rollout weight sync."""

from __future__ import annotations

import logging
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import Artifact, VersionManifest


logger = logging.getLogger(__name__)


def publish_delta_version(
    args: Any,
    version_dir: str,
    files: list[str],
    weight_version: str | int,
    rollout_engines: list[Any],
) -> list[Any]:
    """Slime ``custom_delta_publish_path`` hook: write the version manifest and
    advance ``latest.json`` on the bulletin board.

    Provider-agnostic — no provider import. Durability (e.g. committing a Modal
    Volume) and best-effort rollout-pool wake are layered on by the consuming
    example's hook (see ``cookbook/slime_disagg/hooks.py``).
    """
    del rollout_engines
    version = int(weight_version)
    root = Path(_bulletin_root(args))
    version_path = Path(version_dir)

    # Disk-delta slime writes a canonical model.safetensors.index.json carrying
    # the delta encoding/compression/checksum; lift it instead of hardcoding the
    # format. Fall back to the pre-index layout when the engine didn't write one.
    if (version_path / "model.safetensors.index.json").exists():
        manifest = VersionManifest.from_slime_index(
            version_path,
            run_id=getattr(args, "run_id", None),
            base_model=getattr(args, "hf_checkpoint", None),
        )
    else:
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
    logger.info("Published delta version %s with %d file(s)", manifest.version, len(manifest.transition_files))
    return []


def rollout_request_weight_version_hook(args: Namespace, sample: Any, request: dict[str, Any]) -> None:
    """Slime ``custom_rollout_request_hook_path`` hook: set retry budget and
    session affinity on one rollout request.

    The pool serves whatever version each replica has hot-loaded; a lagging
    replica returns a retryable 409, so the retry budget is what keeps requests
    flowing across a weight update.
    """
    # Generous retries on every request so a lagging/scaling replica (a 409
    # weight-version reject or a transient error) is retried, not failed.
    request["max_retries"] = int(getattr(args, "rollout_request_retry_attempts", request.get("max_retries", 60)))
    request["retry_sleep"] = float(getattr(args, "rollout_request_retry_sleep", request.get("retry_sleep", 1.0)))
    if getattr(sample, "session_id", None):
        # Provider-neutral by default. Modal's Flash gateway routes session
        # affinity on the Modal-Session-ID header, so the Modal configs set this
        # to that name and affinity is honored at the gateway (one hop) rather
        # than re-routed inside the rollout container.
        affinity_header = str(
            getattr(args, "rollout_session_affinity_header", "x-session-affinity")
        )
        headers = dict(request.get("headers") or {})
        headers.setdefault(affinity_header, sample.session_id)
        request["headers"] = headers


def _bulletin_root(args: Any) -> str:
    root = (
        getattr(args, "update_weight_disk_dir", None)
        or getattr(args, "update_weight_delta_root", None)
    )
    if root:
        return str(root)
    return str(Path(args.update_weight_delta_dir).parent)
