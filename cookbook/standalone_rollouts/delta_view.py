"""Host-local ``weight_vN`` view of the customer's identity-named upload dirs.

The customer uploads each checkpoint to ``<transport>/<opaque_identity>/``, but
slime's disk-delta decoder walks ``<delta_root>/weight_v{N:06d}/``. The
front-door ledger maps identity -> version; this builds a host-local directory
per version — one symlink per uploaded file, plus ``latest`` -> the transport
pointer — so the unmodified decoder operates against a normal weight_vN slime
layout while the bytes are read straight from the mount.

The customer's upload is never modified. The disk-delta ``metadata`` block the
decoder needs is written next to the upload as a derived ``stitch.index.json``
(:func:`merge_index_metadata`), and the view presents that file under the
standard HF index name. mountpoint-s3 cannot host a symlink, but a host-local
symlink *into* the mount resolves transparently on read, so the view lives on
the container's ephemeral disk. It is rebuilt from the ledger on every board
refresh; each version dir is built once, since a signalled upload is immutable.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from cookbook.standalone_rollouts.ledger import IdentityLedger
from stitch.protocol import atomic_write_text, weight_identity


LATEST_FILE = "latest"
HF_INDEX_FILE = "model.safetensors.index.json"
DERIVED_INDEX_FILE = "stitch.index.json"


def merge_index_metadata(index_path: Path, metadata: dict[str, str]) -> None:
    """Derive the decoder's index from the customer's uploaded HF index: merge
    the disk-delta ``metadata`` block and write the result next to the upload
    as ``stitch.index.json``, leaving the customer's bytes (and any digest or
    signature over them) untouched.

    Raises ``FileNotFoundError`` when the upload has not landed and
    ``ValueError`` when the uploaded index is not a JSON object; the front door
    maps both to customer-actionable 4xx responses, never a 500.
    """
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"uploaded {index_path.name} is not valid JSON: {exc}") from exc
    if not isinstance(index, dict):
        raise ValueError(f"uploaded {index_path.name} must be a JSON object")
    index.setdefault("metadata", {}).update(metadata)
    atomic_write_text(index_path.with_name(DERIVED_INDEX_FILE), json.dumps(index))


def _ensure_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and os.readlink(link) == str(target):
        return
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def _build_version_dir(vdir: Path, identity_dir: Path) -> None:
    """Materialize one weight_vN view dir: a symlink per uploaded file, with the
    derived index presented under the HF name the decoder reads. Built into a
    tmp dir renamed into place, so a crash never leaves a half-built dir."""
    tmp = vdir.with_name(vdir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    derived = identity_dir / DERIVED_INDEX_FILE
    for f in identity_dir.iterdir():
        if f.name == DERIVED_INDEX_FILE:
            continue
        if f.name == HF_INDEX_FILE and derived.exists():
            (tmp / HF_INDEX_FILE).symlink_to(derived)
        else:
            (tmp / f.name).symlink_to(f)
    tmp.rename(vdir)


def rebuild_delta_view(view_root: str | Path, transport_root: str | Path, ledger: IdentityLedger) -> None:
    """(Re)build the host-local weight_vN view under ``view_root`` from ``ledger``.

    Idempotent, and O(newly signalled versions) per call: an existing version
    dir is left alone. Links ``latest`` to the transport pointer.
    """
    view = Path(view_root)
    transport = Path(transport_root)
    view.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(view / LATEST_FILE, transport / LATEST_FILE)
    for version, identity in ledger.items_by_version():
        vdir = view / weight_identity(version)
        if vdir.is_symlink():
            vdir.unlink()  # an earlier layout linked the identity dir whole
        if not vdir.exists():
            _build_version_dir(vdir, transport / identity)
