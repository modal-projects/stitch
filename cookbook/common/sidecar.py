"""The shared Server-side entrypoint: build the Store + Engine and run the versioned
rollout proxy in front of the local sglang. One entrypoint for every recipe.

The Server container launches this as a subprocess via common/process.py's
``start_sidecar``, which passes every setting from the experiment config as an explicit
flag. The config is the single source of truth; the few defaults below are only for
running the sidecar standalone in dev.
"""

from __future__ import annotations

import argparse

from stitch.engines.sglang import SGLangEngine
from stitch.service import serve
from stitch.stores.modal_volume import ModalVolumeStore


def main() -> None:
    p = argparse.ArgumentParser()
    # Fixed container layout — same on every replica.
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--upstream", default="http://127.0.0.1:8001")
    # Config-owned: no universal default, so a missing one fails loudly rather than
    # silently serving the wrong store / checkpoint.
    p.add_argument("--bulletin-root", required=True)         # store root; exp.DELTA_BULLETIN_ROOT
    p.add_argument("--local-checkpoint-dir", required=True)  # engine's local ckpt; exp.LOCAL_CHECKPOINT_PATH
    # Genuine defaults: "" volume-name = local dir (no Modal volume), and in_place is the
    # default commit policy.
    p.add_argument("--volume-name", default="")
    p.add_argument("--commit-mode", choices=["quiesce", "in_place"], default="in_place")
    p.add_argument("--flush-cache-on-commit", action="store_true")  # evict sglang's prefix/KV cache on reload
    p.add_argument("--run-id", default=None)
    p.add_argument("--debug-requests", action="store_true")
    # Background convergence backstop: re-check latest every N seconds so a replica that
    # missed its wake (cold start racing the last publish, or a lost best-effort wake) still
    # catches up. 0 disables it (wake + 409-self-heal only).
    p.add_argument("--reconcile-interval", type=float, default=5.0)
    args = p.parse_args()

    store = ModalVolumeStore(args.bulletin_root, volume_name=args.volume_name or None)
    engine = SGLangEngine(args.upstream, args.local_checkpoint_dir)
    serve(
        store, engine,
        run_id=args.run_id, commit_mode=args.commit_mode,
        flush_cache_on_commit=args.flush_cache_on_commit,
        host=args.host, port=args.port, debug_requests=args.debug_requests,
        reconcile_interval=args.reconcile_interval,
    )


if __name__ == "__main__":
    main()
