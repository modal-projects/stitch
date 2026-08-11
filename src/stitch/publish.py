"""Trainer-side helpers: publish a version, claim a run, constrain a rollout request.

These are what a training framework wires into its publish and request hooks. They
compose the Store and Pool ports, so they work with any backend — no Modal here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from stitch.pools.base import Pool
from stitch.stores.base import Store
from stitch.types import (
    VersionConstraint,
    VersionManifest,
    VersionRef,
    decide_pointer_move,
)

logger = logging.getLogger(__name__)


def publish_version(
    store: Store,
    pool: Pool | None,
    version_dir: str,
    *,
    run_id: str,
) -> VersionRef:
    """Publish one version from a framework-written directory (full or delta): derive the
    manifest from its HF index, write it durably, advance ``latest`` (rejecting a rewind),
    then wake the pool. Files land before the pointer moves, so a replica never sees a
    pointer to incomplete bytes."""
    manifest = VersionManifest.from_hf_index(version_dir, run_id=run_id)
    expected_dir = Path(manifest.ref.identity).name
    if Path(version_dir).name != expected_dir:
        raise ValueError(
            f"checkpoint index identifies {expected_dir!r}, but was published from "
            f"{Path(version_dir).name!r}"
        )
    missing = [
        name for name in manifest.files if not (Path(version_dir) / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"incomplete source version {version_dir}: missing " + ", ".join(missing)
        )
    expected = store.read_pointer()
    decide_pointer_move(expected, manifest.ref)  # rewind guard
    store.publish(manifest, version_dir)
    store.compare_and_advance_pointer(expected, manifest.ref)
    wake_pool(pool, manifest.ref)
    logger.info(
        "published %s: kind=%s files=%d",
        manifest.ref.identity,
        manifest.kind.value,
        len(manifest.files),
    )
    return manifest.ref


def claim_run(
    store: Store,
    pool: Pool | None,
    run_id: str,
    *,
    boot_version: int = 0,
) -> None:
    """Start a run at its boot checkpoint before its first publish: write the pointer and
    wake the pool, so replicas and the trainer agree on the first served version. A reused
    ``run_id`` above the boot checkpoint is a rewind, while retrying the same claim is
    idempotent."""
    if boot_version < 0:
        raise ValueError("boot_version must be non-negative")
    boot = VersionRef(run_id, boot_version)
    current = store.read_pointer()
    if current == boot:
        # Re-write the same pointer so a retry also retries its durability
        # boundary after an interrupted/ambiguous backend write.
        store.claim(boot)
        wake_pool(pool, boot)
        logger.info("run %s already claimed at v%d", run_id, boot_version)
        return
    decide_pointer_move(current, boot)  # rewind guard (a reused run_id above boot)
    store.claim(boot)
    wake_pool(pool, boot)
    logger.info("claimed run %s at v%d", run_id, boot_version)


def wake_pool(pool: Pool | None, ref: VersionRef) -> None:
    """Best-effort pool wake: the pointer is already durable, so a transient control-plane
    error just costs latency (replicas self-sync on their next poll/startup)."""
    if pool is None:
        return
    try:
        pool.wake(pool.discover_replicas(), ref)
    except Exception:  # noqa: BLE001
        logger.warning(
            "pool wake failed for %s; replicas will self-sync",
            ref.identity,
            exc_info=True,
        )


def constrain_request(
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    latest: int | None = None,
    lag: int = 0,
    exact: int | None = None,
    session_id: Any = None,
    affinity_header: str | None = None,
) -> None:
    """Set the version constraint (on ``payload``) and session affinity (on ``headers``)
    for one outgoing rollout request. ``exact`` pins a single version; otherwise a
    bounded-lag request floors the version at ``latest - lag``. Mutates in place."""
    if exact is not None:
        constraint = VersionConstraint(exact_version=int(exact))
    elif latest is not None:
        constraint = VersionConstraint(min_version=max(0, int(latest) - int(lag)))
    else:
        constraint = VersionConstraint()
    payload["weight_version"] = constraint.to_payload()
    if affinity_header and session_id is not None:
        headers[affinity_header] = str(session_id)
