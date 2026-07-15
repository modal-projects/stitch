"""Replay a recorded delta chain through the real publish path at a controlled cadence.

A pool certification doesn't need a trainer: past runs' chains persist on the delta
Volume as ``<root>/<source_run>/weight_vNNNNNN/`` dirs, and ``publish_version()`` only
needs those dirs. Replaying re-publishes them under a fresh ``run_id`` against a live
pool — real delta densities, sizes, and chain structure, zero trainer GPUs.

Caveats: the pool must serve the same base model the chain was recorded against, and
each publish *copies* the version dir under the new run prefix (that's the real path:
files land before the pointer moves), so the volume grows per replay and nothing GCs.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path


def replay(
    *,
    root: str,
    source_run: str,
    app_name: str | None = None,
    cls_name: str = "Server",
    pool: object = None,  # any stitch Pool; overrides app_name (local shakeout / tests)
    volume_name: str | None = None,
    cadence_s: float = 30.0,
    limit: int | None = None,
    run_id: str | None = None,
) -> str:
    """Claim the pool under a fresh run, then publish the recorded chain v1..vN with
    ``cadence_s`` between publishes. Returns the run_id used. With neither ``pool``
    nor ``app_name`` there are no wakes — replicas converge via their backstop."""
    from stitch.publish import claim_run, publish_version
    from stitch.stores.modal_volume import ModalVolumeStore

    store = ModalVolumeStore(root, volume_name=volume_name)
    if pool is None and app_name is not None:
        from stitch.pools.modal_flash import ModalFlashPool

        pool = ModalFlashPool(app_name, cls_name)
    run_id = run_id or f"replay-{uuid.uuid4().hex[:8]}"

    src = Path(root) / source_run
    dirs = sorted(d for d in src.iterdir() if d.is_dir() and d.name.startswith("weight_v"))
    dirs = [d for d in dirs if int(d.name.removeprefix("weight_v")) > 0]  # v0 is the pool's own base
    if limit is not None:
        dirs = dirs[:limit]
    if not dirs:
        raise SystemExit(f"no weight_v* dirs under {src}")

    print(f"replaying {len(dirs)} versions from {source_run!r} as run {run_id!r} at {cadence_s}s cadence")
    claim_run(store, pool, run_id)
    for d in dirs:
        t = time.time()
        ref = publish_version(store, pool, str(d), run_id=run_id)
        print(f"published {ref.identity} in {time.time() - t:.1f}s")
        time.sleep(cadence_s)
    return run_id


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="delta bulletin root (the mounted Volume path)")
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--app", required=True, help="target pool app name")
    ap.add_argument("--cls", default="Server")
    ap.add_argument("--volume-name", default=None)
    ap.add_argument("--cadence-s", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    replay(
        root=args.root, source_run=args.source_run, app_name=args.app, cls_name=args.cls,
        volume_name=args.volume_name, cadence_s=args.cadence_s, limit=args.limit,
    )
