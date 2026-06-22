"""SGLang weight-sync sidecar launcher for the miles_disagg example.

The reusable versioned-proxy library lives in ``stitch.servers.sglang``; this
module wires the concrete bulletin-board + Modal Volume + SGLang-disk-delta
realization and runs it as a process (``helpers.start_sglang_sidecar`` launches
``python3 -m cookbook.miles_disagg.sidecar``).

The ONLY functional difference from cookbook/slime_disagg/sidecar.py is the
host-side delta decoder: this injects ``miles.utils.disk_delta`` into the
adapter instead of letting it default to ``slime.utils.disk_delta``. The two are
byte-identical (same XOR/overwrite + zstd + xxh3/blake3/adler32 wire format), but
the rollout pool image ships miles (``--no-deps``), not slime, so the miles
functions must be injected explicitly.
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


def _parallel_init_local_checkpoint(workers: int = 32):
    """Return an ``init_local_checkpoint(local, base)`` that copies the base
    shards concurrently. miles' default copies them one at a time with
    ``shutil.copy2`` — fine for a 16 B Moonlight base, far too slow for K2.6's
    ~591 GB. We reuse miles' apply-lock + applied-version bookkeeping so
    ``apply_deltas`` stays compatible, and fall back to the miles default if those
    internals move under us."""
    import concurrent.futures

    from miles.utils import disk_delta as dd

    def _init(local_ckpt_dir: str, base_dir: str) -> None:
        try:
            apply_lock = dd._apply_lock
            read_version = dd._read_applied_version
            write_version = dd._write_applied_version
            drop_page_cache = dd.drop_page_cache
        except AttributeError:  # miles internals changed — use its own copier
            dd.init_local_checkpoint(local_ckpt_dir, base_dir)
            return

        with apply_lock(local_ckpt_dir):
            if read_version(local_ckpt_dir) is not None:
                return
            logger.info("Materializing base %s -> %s (%d copy workers)", base_dir, local_ckpt_dir, workers)
            os.makedirs(local_ckpt_dir, exist_ok=True)
            files = [e for e in os.scandir(base_dir) if e.is_file()]

            def _copy(entry: os.DirEntry) -> None:
                import shutil

                shutil.copy2(entry.path, os.path.join(local_ckpt_dir, entry.name))
                drop_page_cache(entry.path)  # don't evict the local copy we keep resident

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, max(1, len(files)))) as ex:
                for _ in ex.map(_copy, files):  # re-raises the first copy failure
                    pass
            write_version(local_ckpt_dir, "000000")

    return _init


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
    # The miles host-side delta decoder. miles is installed --no-deps in the
    # serving image for exactly this module (Megatron is absent — the pool never
    # trains). Pinned to the same miles ref as the trainer so encoder == decoder.
    from miles.utils.disk_delta import apply_deltas

    refresh = None
    if volume_name:
        from stitch.providers.modal import volume_reloader

        refresh = volume_reloader(volume_name)
    # miles publish-only writes the flat layout (weight_v{N}/ +
    # model.safetensors.index.json + a raw `latest` pointer) to the Volume — the
    # same layout slime uses, so layout="slime" reads it unchanged.
    board = FilesystemBulletinBoard(bulletin_root, refresh=refresh, layout="slime")
    engine = SGLangDiskDeltaAdapter(
        upstream_url=upstream_url,
        local_checkpoint_dir=local_checkpoint_dir,
        base_checkpoint_dir=base_checkpoint_dir,
        apply_deltas=apply_deltas,
        init_local_checkpoint=_parallel_init_local_checkpoint(),
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
