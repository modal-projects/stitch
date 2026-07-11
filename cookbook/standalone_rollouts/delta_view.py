"""Host-local ``weight_vN`` view of the customer's identity-named upload dirs.

The customer uploads each checkpoint to ``<transport>/<opaque_identity>/`` (spec
§1 couples the dir name to the identity), but slime's disk-delta decoder walks
``<delta_root>/weight_v{N:06d}/``. The front-door ledger maps identity -> version;
this builds a host-local directory of symlinks (``weight_vN`` -> the transport's
identity dir, plus ``latest`` -> the transport pointer) so the unmodified decoder
and bulletin board operate against a normal weight_vN slime layout.

mountpoint-s3 cannot host a symlink (ENOSYS, same family as rename), but a
host-local symlink *into* the mount resolves through transparently on read, so
the view lives on the container's ephemeral disk and the delta bytes are still
read straight from S3. The view is rebuilt from the ledger on every board
refresh (i.e. before every sync), so newly-signalled versions appear.
"""

from __future__ import annotations

import os
from pathlib import Path

from cookbook.standalone_rollouts.ledger import IdentityLedger
from stitch.protocol import weight_identity


LATEST_FILE = "latest"


def _ensure_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() and os.readlink(link) == str(target):
        return
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def rebuild_delta_view(view_root: str | Path, transport_root: str | Path, ledger: IdentityLedger) -> None:
    """(Re)build the host-local weight_vN view under ``view_root`` from ``ledger``.

    Idempotent: existing correct symlinks are left alone. Links ``latest`` to the
    transport pointer and each minted version to its identity dir on the transport.
    """
    view = Path(view_root)
    transport = Path(transport_root)
    view.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(view / LATEST_FILE, transport / LATEST_FILE)
    for version, identity in ledger.items_by_version():
        _ensure_symlink(view / weight_identity(version), transport / identity)
