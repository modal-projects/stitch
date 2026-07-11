"""Shared SGLang weight-sync sidecar spine for the disagg cookbook trainers.

The reusable versioned-proxy library lives in ``stitch.servers.sglang``; this
module wires the concrete bulletin-board + Modal Volume + SGLang realization
once, and the per-trainer ``cookbook/<trainer>/sidecar.py`` modules are thin
entrypoints.

The delta apply lives in the ENGINE (sglang ``/pull_weights`` +
``weight_sync/local_checkpoint``): the sidecar imports no trainer package at
all — slime and miles publish the same wire format and the engine decodes it.
What lives here: arg parsing, the bulletin board + Volume reloader, and the
manager/proxy assembly.
"""

from __future__ import annotations

import argparse
import logging
import os

from stitch.bulletin import FilesystemBulletinBoard
from stitch.engines.sglang import SGLangDiskDeltaAdapter
from stitch.servers.sglang import create_app
from stitch.sync import CommitMode, WeightSyncManager


logger = logging.getLogger(__name__)


def build_manager(
    *,
    upstream_url: str,
    bulletin_root: str,
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    volume_name: str = "",
    run_id: str | None = None,
    commit_mode: CommitMode = "in_place",
    debug_requests: bool = False,
) -> WeightSyncManager:
    """Build the reconciling :class:`WeightSyncManager` for one rollout replica.

    The delta decode/apply and base materialization live in the engine behind
    ``/pull_weights`` (seeded from the engine's own model path, checksum-verified,
    reseed-on-corruption), so no trainer decoder is wired here.
    """
    refresh = None
    if volume_name:
        from stitch.providers.modal import volume_reloader

        refresh = volume_reloader(volume_name)
    # Publish-only writes the flat slime-native layout (weight_v{N}/ +
    # model.safetensors.index.json + a raw `latest` pointer) to the transport;
    # miles writes the same layout, so layout="slime" reads either unchanged.
    board = FilesystemBulletinBoard(bulletin_root, refresh=refresh, layout="slime")
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


def run_sidecar() -> None:
    """Parse args and run the sidecar uvicorn server.

    The per-trainer entrypoints call this; ``run_sidecar`` owns every knob
    (host/port/transport/commit-mode/timeout/...).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument(
        "--bulletin-root",
        default=os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin"),
    )
    parser.add_argument("--volume-name", default=os.environ.get("DELTA_VOLUME_NAME", ""))
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
    parser.add_argument("--run-id", default=os.environ.get("DISAGG_RUN_ID"))
    parser.add_argument(
        "--commit-mode",
        choices=("quiesce", "in_place"),
        default=os.environ.get("SIDECAR_COMMIT_MODE", "in_place"),
        help=(
            "in_place (default): pause/apply/continue without flushing; in-flight "
            "requests keep decoding on stale KV and version isolation comes from "
            "extra_key stamping. quiesce: wait out active requests and flush "
            "before applying (safe on any build)."
        ),
    )
    parser.add_argument(
        "--debug-requests",
        action="store_true",
        default=os.environ.get("SIDECAR_DEBUG_REQUESTS", "").lower() in {"1", "true", "yes"},
        help="Log every versioned sidecar proxy request at INFO level.",
    )
    parser.add_argument(
        "--upstream-timeout",
        type=float,
        default=float(os.environ.get("SIDECAR_UPSTREAM_TIMEOUT", "3600")),
        help=(
            "Seconds to wait for an upstream SGLang response before failing the "
            "request (5xx) instead of holding it open forever. Must exceed the "
            "slowest legitimate generation."
        ),
    )
    args = parser.parse_args()
    if not args.base_checkpoint_dir:
        raise SystemExit(
            "--base-checkpoint-dir/STITCH_BASE_CHECKPOINT_DIR is required: the engine"
            " serves this base HF checkpoint and seeds its local copy from it."
        )

    logging.basicConfig(level=logging.INFO)
    import uvicorn

    manager = build_manager(
        upstream_url=args.upstream_url,
        bulletin_root=args.bulletin_root,
        local_checkpoint_dir=args.local_checkpoint_dir,
        base_checkpoint_dir=args.base_checkpoint_dir,
        volume_name=args.volume_name,
        run_id=args.run_id,
        commit_mode=args.commit_mode,
        debug_requests=args.debug_requests,
    )
    uvicorn.run(
        create_app(
            manager,
            upstream_url=args.upstream_url,
            upstream_timeout=args.upstream_timeout,
        ),
        host=args.host,
        port=args.port,
        log_level="info",
    )
