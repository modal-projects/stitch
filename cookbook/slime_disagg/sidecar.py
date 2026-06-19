"""SGLang weight-sync sidecar launcher for the slime_disagg example.

The reusable versioned-proxy library lives in ``stitch.servers.sglang``; this
module wires the concrete bulletin-board + Modal Volume + SGLang-disk-delta
realization and runs it as a process (``helpers.start_sglang_sidecar`` launches
``python3 -m cookbook.slime_disagg.sidecar``).
"""

from __future__ import annotations

import argparse
import logging
import os

from stitch.bulletin import FilesystemBulletinBoard
from stitch.engines.sglang import SGLangDiskDeltaAdapter
from stitch.servers.sglang import create_app
from stitch.sync import CommitMode, WeightSyncManager


def build_manager(
    *,
    upstream_url: str,
    bulletin_root: str,
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    volume_name: str = "",
    run_id: str | None = None,
    commit_mode: CommitMode = "quiesce",
    debug_requests: bool = False,
) -> WeightSyncManager:
    refresh = None
    if volume_name:
        from stitch.providers.modal import volume_reloader

        refresh = volume_reloader(volume_name)
    board = FilesystemBulletinBoard(bulletin_root, refresh=refresh)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument(
        "--bulletin-root",
        default=os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin"),
    )
    parser.add_argument(
        "--volume-name", default=os.environ.get("DELTA_VOLUME_NAME", "")
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
    parser.add_argument("--run-id", default=os.environ.get("DISAGG_RUN_ID"))
    parser.add_argument(
        "--commit-mode",
        choices=("quiesce", "in_place"),
        default=os.environ.get("SIDECAR_COMMIT_MODE", "quiesce"),
        help=(
            "quiesce: wait out active requests and flush before applying. "
            "in_place: pause/apply/continue without flushing; in-flight "
            "requests keep decoding on stale KV and version isolation comes "
            "from extra_key stamping. in_place requires an engine build with "
            "the overlap-drain fix."
        ),
    )
    parser.add_argument(
        "--debug-requests",
        action="store_true",
        default=os.environ.get("SIDECAR_DEBUG_REQUESTS", "").lower()
        in {"1", "true", "yes"},
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
            "--base-checkpoint-dir/STITCH_BASE_CHECKPOINT_DIR is required: deltas are"
            " applied host-side on top of a copy of this base HF checkpoint."
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


if __name__ == "__main__":
    main()
