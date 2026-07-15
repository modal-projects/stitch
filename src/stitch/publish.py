"""Trainer-side helpers: publish a version, claim a run, constrain a rollout request.

These are what a training framework wires into its publish and request hooks. They
compose the Store and Pool ports, so they work with any backend — no Modal here.
"""

from __future__ import annotations

import logging
from typing import Any

from stitch.pools.base import Pool
from stitch.stores.base import Store
from stitch.versions import VersionConstraint, VersionManifest, VersionRef, decide_pointer_move

logger = logging.getLogger(__name__)


def publish_version(
    store: Store,
    pool: Pool | None,
    version_dir: str,
    *,
    run_id: str,
    base_model: str | None = None,
) -> VersionRef:
    """Publish one version from a framework-written directory (full or delta): derive the
    manifest from its HF index, write it durably, advance ``latest`` (rejecting a rewind),
    then wake the pool. Files land before the pointer moves, so a replica never sees a
    pointer to incomplete bytes."""
    manifest = VersionManifest.from_hf_index(version_dir, run_id=run_id, base_model=base_model)
    decide_pointer_move(store.read_pointer(), manifest.ref)  # raises PointerRewind on a rewind
    store.publish(manifest, version_dir)
    store.advance_pointer(manifest.ref)
    _wake(pool, manifest.ref)
    return manifest.ref


def claim_run(store: Store, pool: Pool | None, run_id: str) -> None:
    """Start a run at base before its first publish: write the base pointer and wake the
    pool, so every replica (cold or warm on a finished run) resets to base up front. A
    reused ``run_id`` (the run's per-launch fence token) is a rewind — rejected here so a
    restart can't leave the pool pinned to a dead incarnation's high-water mark."""
    decide_pointer_move(store.read_pointer(), VersionRef(run_id, 0))  # raises PointerRewind on a reused run_id
    store.claim(run_id)
    _wake(pool, VersionRef(run_id, 0))


def _wake(pool: Pool | None, ref: VersionRef) -> None:
    """Best-effort pool wake: the pointer is already durable, so a transient control-plane
    error just costs latency (replicas self-sync on their next poll/startup)."""
    if pool is None:
        return
    try:
        pool.wake(pool.discover_replicas(), ref)
    except Exception:  # noqa: BLE001
        logger.warning("pool wake failed for %s; replicas will self-sync", ref.identity, exc_info=True)


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
