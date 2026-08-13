"""Trainer-side helpers: publish a version, claim a run, constrain a rollout request.

These are what a training framework wires into its publish and request hooks. They
compose the Store and Pool ports, so they work with any backend — no Modal here.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
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


def restore_run(
    store: Store,
    pool: Pool,
    target: VersionRef,
    checkpoint_dir: str,
    *,
    timeout: float = 2 * 60 * 60,
) -> None:
    """Restore every live replica to a complete checkpoint within the same run.

    The trainer is stopped while this runs. Move the durable pointer first so a
    replacement replica boots at the restored version, then repeatedly discover
    the elastic pool until every visible replica has committed ``target``.
    """
    current = store.read_pointer()
    if current is None or current.run_id != target.run_id:
        actual = current.identity if current is not None else "<unset>"
        raise ValueError(f"cannot restore {target.identity!r} from latest {actual!r}")
    if target.version > current.version:
        raise ValueError(
            f"restore target v{target.version} is newer than latest v{current.version}"
        )
    if current != target:
        store.compare_and_advance_pointer(current, target)

    import httpx

    deadline = time.monotonic() + timeout
    stable = 0
    stable_replicas: frozenset[str] | None = None
    while time.monotonic() < deadline:
        replicas = pool.discover_replicas()
        if not replicas:
            stable = 0
            stable_replicas = None
            time.sleep(2.0)
            continue

        def restore_replica(url: str) -> bool:
            try:
                response = httpx.post(
                    f"{url.rstrip('/')}/restore",
                    json={
                        "target": target.identity,
                        "checkpoint_dir": checkpoint_dir,
                    },
                    timeout=max(0.1, deadline - time.monotonic()),
                )
                response.raise_for_status()
                return response.json().get("applied") == target.identity
            except Exception:  # noqa: BLE001
                logger.warning(
                    "failed to restore replica %s to %s",
                    url,
                    target.identity,
                    exc_info=True,
                )
                return False

        with ThreadPoolExecutor(max_workers=min(16, len(replicas))) as executor:
            restored = list(executor.map(restore_replica, replicas))
        if all(restored):
            visible = frozenset(replicas)
            stable = stable + 1 if visible == stable_replicas else 1
            stable_replicas = visible
            if stable == 2:
                logger.info(
                    "restored %d rollout replicas to %s",
                    len(replicas),
                    target.identity,
                )
                return
        else:
            stable = 0
            stable_replicas = None
        time.sleep(2.0)
    raise TimeoutError(f"rollout pool did not restore to {target.identity}")


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
