"""Probe: inspect (and, with --confirm, wipe) the standalone provider's S3
transport state.

The run-id design partitions each run's delta chain under ``<run_id>/weight_v{N}/``
and a single self-identifying ``latest`` pointer (``<run_id>/weight_vNNNNNN``).
Pre-run-id deployments left a flat ``weight_v*`` chain + a bare ``latest``
(``"NNNNNN"``); that layout is incompatible and must be wiped once before the
new code runs (a cold-starting replica would otherwise read the stale bare
pointer and choke on the orphaned flat chain until the first publish).

    m run -m modal_probes.inspect_s3_transport::inspect
    m run -m modal_probes.inspect_s3_transport::wipe --confirm
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal

BUCKET = "modal-stitch-s3-transport"
PREFIX = "standalone-rollouts/moonlight"
OIDC_ROLE = "arn:aws:iam::459781239556:role/modal-buckets/stitch-s3-transport-role"


def _mount(read_only: bool) -> modal.CloudBucketMount:
    return modal.CloudBucketMount(
        bucket_name=BUCKET,
        key_prefix=f"{PREFIX}/",
        oidc_auth_role_arn=OIDC_ROLE,
        read_only=read_only,
    )


image = modal.Image.debian_slim()
app = modal.App("inspect-s3-transport")


def _report(root: Path) -> None:
    latest = root / "latest"
    print(f"latest pointer = {latest.read_text().strip() if latest.exists() else 'MISSING'}")
    print(f"top-level entries: {sorted(p.name for p in root.iterdir())}")
    # Run partitions: any top-level dir holding its own weight_v* chain.
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("weight_v")):
        wv = sorted(x.name for x in d.glob("weight_v*"))
        if wv:
            print(f"[run dir] {d.name}: versions={wv}")
    flat = sorted(root.glob("weight_v*"))
    if flat:
        print(f"!! legacy flat (pre-run-id) version dirs present: {[d.name for d in flat]}")
    for d in flat:
        idx = d / "model.safetensors.index.json"
        meta = json.loads(idx.read_text()).get("metadata", {}) if idx.exists() else {}
        print(f"   {d.name}.index.metadata = {meta}")


@app.function(image=image, volumes={"/mnt/t": _mount(read_only=True)}, timeout=10 * 60, region="us")
def inspect() -> None:
    _report(Path("/mnt/t"))


@app.function(image=image, volumes={"/mnt/t": _mount(read_only=False)}, timeout=10 * 60, region="us")
def wipe(confirm: bool = False) -> None:
    """Delete everything under the prefix (pointer + all run/flat version dirs).

    The state is migration-incompatible (and, for the diagnosed app, corrupt), so
    this is a clean reset: replicas then cold-start at (None, 0) on base, and the
    first training run writes a fresh ``<run_id>/`` partition + pointer.
    """
    root = Path("/mnt/t")
    if not confirm:
        print("Refusing to wipe without --confirm. Current state:")
        _report(root)
        return
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    print("Wiped. Post-wipe state:")
    _report(root)
