"""Per-container rollout sidecar for the standalone hot-load provider.

A pool member of the log-as-truth design: a :class:`WeightSyncManager` that
reconciles its local SGLang server to the shared ``latest`` pointer. The trainer
uploads ``weight_v{N}/`` version dirs to object storage (mounted here as the
transport root) in the flat slime/customer layout, and the front door advances
``latest`` on ``POST /hot_load``. Each replica pulls from that durable monotonic
log — on startup, on a best-effort wake, and on a periodic reconcile — applies
the disk delta host-side, and reloads the engine.

There is no desired-mailbox and no self-reported replica state: a scaled-up
container catches up by reading ``latest`` with no push, and the front door
derives pool readiness by enumerating the live containers and querying
``/server_info``. Authentication and the customer hot-load API live in the front
door (``modal_serve.py``), not here.
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from collections.abc import Callable

from cookbook.standalone_rollouts.delta_view import rebuild_delta_view
from cookbook.standalone_rollouts.ledger import IdentityLedger, load_ledger_dict
from stitch.bulletin import FilesystemBulletinBoard
from stitch.engines.sglang import SGLangDiskDeltaAdapter
from stitch.servers.sglang import create_app as create_sglang_app
from stitch.sync import CommitMode, WeightSyncManager


logger = logging.getLogger(__name__)
VERSIONED_ROUTES = frozenset({"generate", "v1/chat/completions", "v1/completions"})


def build_manager(
    *,
    upstream_url: str,
    transport_root: str,
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    commit_mode: CommitMode = "in_place",
    run_id: str | None = None,
    debug_requests: bool = False,
    delta_view_dir: str | None = None,
) -> WeightSyncManager:
    # The transport root is the object-store mount holding the version dirs and
    # the raw `latest` pointer. Deltas are read straight from the mount.
    #
    # The customer uploads to opaque-identity dirs (`<transport>/<identity>/`),
    # not weight_vN, so when delta_view_dir is set the board is rooted at a
    # host-local weight_vN symlink view of the transport (rebuilt from the
    # front-door ledger on every refresh), letting the unmodified decoder walk
    # weight_vN while the bytes still come from the identity dirs on the mount.
    # Without it (the internal slime harness / tests, which write weight_vN dirs
    # directly) the board reads the transport as-is.
    if delta_view_dir:
        board = FilesystemBulletinBoard(
            delta_view_dir,
            layout="slime",
            refresh=_delta_view_refresh(delta_view_dir, transport_root),
        )
    else:
        board = FilesystemBulletinBoard(transport_root, layout="slime")
    engine = SGLangDiskDeltaAdapter(
        upstream_url=upstream_url,
        local_checkpoint_dir=local_checkpoint_dir,
        base_checkpoint_dir=base_checkpoint_dir,
    )
    return WeightSyncManager(
        board=board,
        engine=engine,
        run_id=run_id,
        commit_mode=commit_mode,
        debug_requests=debug_requests,
    )


def _delta_view_refresh(view_dir: str, transport_root: str) -> Callable[[], None]:
    """Board refresh callback: rebuild the host-local weight_vN view from the
    front door's identity ledger on the transport, so the view tracks
    newly-signalled versions before each sync reads/apply."""
    def _refresh() -> None:
        ledger = IdentityLedger.from_dict(load_ledger_dict(transport_root))
        rebuild_delta_view(view_dir, transport_root, ledger)

    return _refresh


def create_app(manager: WeightSyncManager, *, upstream_url: str, poll_interval: float = 5.0):
    # include_sync_routes=True keeps /rpc_sync_from_bulletin_board so the front
    # door can wake replicas the moment it advances `latest`; the periodic
    # reconcile is the fallback that converges anything that missed the wake.
    return create_sglang_app(
        manager,
        upstream_url=upstream_url,
        versioned_routes=VERSIONED_ROUTES,
        include_sync_routes=True,
        background_sync_interval=poll_interval,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument(
        "--transport-root",
        default=os.environ.get("STITCH_SHIM_TRANSPORT_ROOT"),
        help="Object-store mount holding weight_v{N}/ dirs and the raw `latest` pointer.",
    )
    parser.add_argument(
        "--local-checkpoint-dir",
        default=os.environ.get("STITCH_LOCAL_CHECKPOINT_DIR", "/local-checkpoint"),
        help="Writable host-local full HF checkpoint patched in place by each delta.",
    )
    parser.add_argument(
        "--base-checkpoint-dir",
        default=os.environ.get("STITCH_BASE_CHECKPOINT_DIR"),
        help="Base HF checkpoint the local copy is seeded from (deltas build on it).",
    )
    parser.add_argument(
        "--commit-mode",
        choices=("quiesce", "in_place"),
        default=os.environ.get("STITCH_SHIM_COMMIT_MODE", "in_place"),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("STITCH_SHIM_POLL_INTERVAL", "5.0")),
        help="Seconds between background reconciles against the `latest` pointer.",
    )
    parser.add_argument(
        "--delta-view-dir",
        default=os.environ.get("STITCH_DELTA_VIEW_DIR"),
        help="Host-local dir for the weight_vN symlink view of the transport's "
        "opaque-identity dirs. Set for the customer layout; omit when version "
        "dirs are already named weight_vN.",
    )
    parser.add_argument("--run-id", default=os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex)
    parser.add_argument(
        "--debug-requests",
        action="store_true",
        default=os.environ.get("STITCH_SHIM_DEBUG_REQUESTS", "").lower() in {"1", "true", "yes"},
    )
    args = parser.parse_args()
    if not args.transport_root:
        raise SystemExit("--transport-root/STITCH_SHIM_TRANSPORT_ROOT is required")
    if not args.base_checkpoint_dir:
        raise SystemExit(
            "--base-checkpoint-dir/STITCH_BASE_CHECKPOINT_DIR is required: deltas are"
            " applied host-side on top of a copy of this base HF checkpoint."
        )

    logging.basicConfig(level=logging.INFO)
    import uvicorn

    manager = build_manager(
        upstream_url=args.upstream_url,
        transport_root=args.transport_root,
        local_checkpoint_dir=args.local_checkpoint_dir,
        base_checkpoint_dir=args.base_checkpoint_dir,
        commit_mode=args.commit_mode,
        run_id=args.run_id,
        debug_requests=args.debug_requests,
        delta_view_dir=args.delta_view_dir,
    )
    uvicorn.run(
        create_app(manager, upstream_url=args.upstream_url, poll_interval=args.poll_interval),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
