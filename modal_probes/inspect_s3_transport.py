"""Probe: dump the standalone provider's S3 transport state to confirm the
cross-run weight-version mismatch (durable `latest` + overwritten weight_v* chain).

    m run -m modal_probes.inspect_s3_transport::inspect
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

BUCKET = "modal-stitch-s3-transport"
PREFIX = "standalone-rollouts/moonlight"
OIDC_ROLE = "arn:aws:iam::459781239556:role/modal-buckets/stitch-s3-transport-role"

mount = modal.CloudBucketMount(
    bucket_name=BUCKET,
    key_prefix=f"{PREFIX}/",
    oidc_auth_role_arn=OIDC_ROLE,
    read_only=True,
)
image = modal.Image.debian_slim()
app = modal.App("inspect-s3-transport")


@app.function(image=image, volumes={"/mnt/t": mount}, timeout=10 * 60, region="us")
def inspect() -> None:
    root = Path("/mnt/t")
    latest = root / "latest"
    cur_epoch = root / "current_epoch"
    print(f"current_epoch = {cur_epoch.read_text().strip() if cur_epoch.exists() else 'MISSING'}")
    print(f"flat latest   = {latest.read_text().strip() if latest.exists() else 'MISSING'}")
    print(f"top-level entries: {sorted(p.name for p in root.iterdir())}")
    # Epoch dirs: any top-level dir that itself contains a `latest` or weight_v*.
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("weight_v")):
        ep_latest = d / "latest"
        wv = sorted(x.name for x in d.glob("weight_v*"))
        print(f"[epoch dir] {d.name}: latest={ep_latest.read_text().strip() if ep_latest.exists() else 'MISSING'}, versions={wv}")
    for d in sorted(root.glob("weight_v*")):
        files = sorted(d.rglob("*"))
        mtimes = [f.stat().st_mtime for f in files if f.is_file()]
        span = f"mtime {min(mtimes):.0f}..{max(mtimes):.0f}" if mtimes else "no files"
        meta = {}
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            meta = json.loads(idx.read_text()).get("metadata", {})
        print(f"{d.name}: {sum(f.is_file() for f in files)} files, {span}")
        print(f"   index.metadata = {meta}")
