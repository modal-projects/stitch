"""Shared SGLang weight-sync sidecar spine for the disagg cookbook trainers.

The reusable versioned-proxy library lives in ``stitch.servers.sglang``; this
module wires the concrete bulletin-board + Modal Volume + SGLang-disk-delta
realization once, and the per-trainer ``cookbook/<trainer>/sidecar.py`` modules
are thin adapters that select the host-side delta decoder.

The ONLY axis that varies between trainers is the host-side ``disk_delta``
module the rollout pool ships ``--no-deps`` (``slime.utils.disk_delta`` vs
``miles.utils.disk_delta``). Their XOR/overwrite + zstd + xxh3/blake3 wire
formats are byte-identical, but the pool image installs exactly one of them, so
the adapter names which to import. Everything else — arg parsing, the bulletin
board + Volume reloader, the base-materialization strategy — lives here.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import logging
import os
import shutil
from typing import Callable

from stitch.bulletin import FilesystemBulletinBoard
from stitch.engines.sglang import SGLangDiskDeltaAdapter
from stitch.servers.sglang import create_app
from stitch.sync import WeightSyncManager


logger = logging.getLogger(__name__)


def parallel_init_local_checkpoint(disk_delta_module: str, workers: int = 32) -> Callable[[str, str], None]:
    """Return an ``init_local_checkpoint(local, base)`` that copies the base
    shards concurrently and skips the copy when the materialized base is current.

    ``disk_delta``'s default seeds the local checkpoint one shard at a time with
    ``shutil.copy2`` — fine for a 16 B Moonlight base, far too slow for K2.6's
    ~591 GB. We reuse the decoder's apply-lock + applied-version bookkeeping so
    ``apply_deltas`` stays compatible, and fall back to the decoder's own copier
    if those internals move under us. ``disk_delta_module`` is the trainer's
    decoder (``slime.utils.disk_delta`` / ``miles.utils.disk_delta``); both expose
    the same internals (miles is forked from slime), so one implementation serves
    both. Imported lazily so the module imports without slime/miles installed."""
    import importlib

    def _base_fingerprint(base_dir: str) -> str:
        """Identity of the base's bytes: (filename, size, mtime_ns) over every shard. Re-prep
        rewrites the shards (new mtime/size), so this changes even when the tensor index/shapes
        don't — which is exactly the case that bit K2.6 (NVFP4 values changed, names didn't)."""
        h = hashlib.sha256()
        for e in sorted(os.scandir(base_dir), key=lambda e: e.name):
            if e.is_file():
                st = e.stat()
                h.update(f"{e.name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
        return h.hexdigest()

    def _init(local_ckpt_dir: str, base_dir: str) -> None:
        dd = importlib.import_module(disk_delta_module)
        try:
            apply_lock = dd._apply_lock
            read_version = dd._read_applied_version
            write_version = dd._write_applied_version
            drop_page_cache = dd.drop_page_cache
        except AttributeError:  # decoder internals changed — use its own copier
            dd.init_local_checkpoint(local_ckpt_dir, base_dir)
            return

        fp_path = os.path.join(local_ckpt_dir, ".base_fingerprint")
        base_fp = _base_fingerprint(base_dir)
        with apply_lock(local_ckpt_dir):
            if read_version(local_ckpt_dir) is not None:
                try:
                    cur = open(fp_path).read().strip()
                except FileNotFoundError:
                    cur = None
                if cur == base_fp:
                    return  # current base already materialized
                # STALE: /local-checkpoint holds a different (older) base — e.g. after a re-prep.
                # Reusing it makes the host-side delta (diffed against the NEW base) XOR onto the
                # WRONG bytes -> base_local XOR delta != export -> checksum mismatch on every tensor.
                logger.warning(
                    "stale /local-checkpoint (base changed: %s != %s) — wiping + re-materializing",
                    cur, base_fp,
                )
                shutil.rmtree(local_ckpt_dir, ignore_errors=True)
            logger.info("Materializing base %s -> %s (%d copy workers)", base_dir, local_ckpt_dir, workers)
            os.makedirs(local_ckpt_dir, exist_ok=True)
            files = [e for e in os.scandir(base_dir) if e.is_file()]

            def _copy(entry: os.DirEntry) -> None:
                shutil.copy2(entry.path, os.path.join(local_ckpt_dir, entry.name))
                drop_page_cache(entry.path)  # don't evict the local copy we keep resident

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, max(1, len(files)))) as ex:
                for _ in ex.map(_copy, files):  # re-raises the first copy failure
                    pass
            write_version(local_ckpt_dir, "000000")
            with open(fp_path, "w") as f:
                f.write(base_fp)

    return _init


def build_manager(
    *,
    upstream_url: str,
    bulletin_root: str,
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    disk_delta_module: str,
    inject_apply_deltas: bool,
    base_copy_workers: int = 32,
    volume_name: str = "",
    run_id: str | None = None,
    debug_requests: bool = False,
) -> WeightSyncManager:
    """Build the reconciling :class:`WeightSyncManager` for one rollout replica.

    ``disk_delta_module`` is the trainer's host-side decoder. ``inject_apply_deltas``
    passes that module's ``apply_deltas`` to the engine explicitly — needed when the
    decoder is not slime's default (miles), and harmless when it is. The base copy is
    always the shared concurrent/stale-aware materializer.
    """
    import importlib

    refresh = None
    if volume_name:
        from stitch.providers.modal import volume_reloader

        refresh = volume_reloader(volume_name)
    # Publish-only writes the flat slime-native layout (weight_v{N}/ +
    # model.safetensors.index.json + a raw `latest` pointer) to the transport;
    # miles writes the same layout, so layout="slime" reads either unchanged.
    board = FilesystemBulletinBoard(bulletin_root, refresh=refresh, layout="slime")
    apply_deltas = None
    if inject_apply_deltas:
        # The decoder is installed --no-deps in the serving image for exactly this
        # module (Megatron is absent — the pool never trains). Pinned to the same
        # trainer ref as the encoder so encoder == decoder.
        apply_deltas = importlib.import_module(disk_delta_module).apply_deltas
    engine = SGLangDiskDeltaAdapter(
        upstream_url=upstream_url,
        local_checkpoint_dir=local_checkpoint_dir,
        base_checkpoint_dir=base_checkpoint_dir,
        apply_deltas=apply_deltas,
        init_local_checkpoint=parallel_init_local_checkpoint(disk_delta_module, base_copy_workers),
    )
    return WeightSyncManager(
        board=board,
        engine=engine,
        run_id=run_id,
        debug_requests=debug_requests,
    )


def run_sidecar(*, disk_delta_module: str, inject_apply_deltas: bool) -> None:
    """Parse args and run the sidecar uvicorn server.

    The per-trainer adapter calls this with its decoder module; ``run_sidecar``
    owns every other knob (host/port/transport/timeout/...).
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
        "--base-copy-workers",
        type=int,
        default=int(os.environ.get("SIDECAR_BASE_COPY_WORKERS", "32")),
        help="Threads used to materialize the base checkpoint (concurrent shard copy).",
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
        disk_delta_module=disk_delta_module,
        inject_apply_deltas=inject_apply_deltas,
        base_copy_workers=args.base_copy_workers,
        volume_name=args.volume_name,
        run_id=args.run_id,
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
