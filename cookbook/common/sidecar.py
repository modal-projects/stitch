"""The shared Server-side entrypoint: build the Store + Engine and run the versioned
rollout proxy in front of the local sglang. One entrypoint for every recipe.

The Server container launches this as a subprocess via common/process.py's
``start_sidecar``, which passes every setting from the experiment config as an explicit
flag. The config is the single source of truth; the few defaults below are only for
running the sidecar standalone in dev.
"""

from __future__ import annotations

import argparse
import logging
import sys

from stitch.engines.sglang import SGLangEngine
from stitch.service import serve
from stitch.stores.modal_volume import ModalVolumeStore


def _configure_logging() -> None:
    """Emit INFO logs to stdout (uvicorn configures only its own loggers)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    _configure_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--upstream", default="http://127.0.0.1:8001")
    p.add_argument("--bulletin-root", required=True)
    p.add_argument("--local-checkpoint-dir", required=True)
    p.add_argument("--volume-name", default="")
    p.add_argument("--commit-mode", choices=["in_place", "quiesce"], default="in_place")
    p.add_argument("--weight-update-mode", choices=["disk", "host_runtime"], default="disk")
    p.add_argument("--flush-cache-on-commit", action="store_true")
    p.add_argument("--run-id", default=None)
    p.add_argument("--debug-requests", action="store_true")
    p.add_argument("--reconcile-interval", type=float, default=5.0)  # 0 disables the periodic re-check
    args = p.parse_args()

    store = ModalVolumeStore(args.bulletin_root, volume_name=args.volume_name or None)
    engine = SGLangEngine(
        args.upstream,
        args.local_checkpoint_dir,
        weight_update_mode=args.weight_update_mode,
    )
    serve(
        store, engine,
        run_id=args.run_id, commit_mode=args.commit_mode,
        flush_cache_on_commit=args.flush_cache_on_commit,
        host=args.host, port=args.port, debug_requests=args.debug_requests,
        reconcile_interval=args.reconcile_interval,
    )


if __name__ == "__main__":
    main()
