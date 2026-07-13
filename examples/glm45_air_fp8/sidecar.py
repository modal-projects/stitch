"""Sidecar entrypoint for the Server replica: build the Store + Engine and run the
versioned rollout proxy in front of the local sglang.

The Server container launches this as a subprocess:
  python -m examples.glm45_air_fp8.sidecar --port 8000 --upstream http://127.0.0.1:8001 ...
Defaults come from the serving container's env, so the flags are optional.
"""

from __future__ import annotations

import argparse
import os

from stitch.engines.sglang import SGLangEngine
from stitch.service import serve
from stitch.stores.modal_volume import ModalVolumeStore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--upstream", default="http://127.0.0.1:8001")
    p.add_argument("--bulletin-root", default=os.environ.get("DELTA_BULLETIN_ROOT", "/delta-bulletin"))
    p.add_argument("--volume-name", default=os.environ.get("DELTA_VOLUME_NAME", ""))
    p.add_argument("--local-checkpoint-dir", default=os.environ.get("STITCH_LOCAL_CHECKPOINT_DIR", "/local-checkpoint"))
    p.add_argument("--commit-mode", choices=["quiesce", "in_place"], default=os.environ.get("SIDECAR_COMMIT_MODE", "quiesce"))
    p.add_argument("--run-id", default=os.environ.get("DISAGG_RUN_ID") or None)
    p.add_argument("--debug-requests", action="store_true")
    args = p.parse_args()

    store = ModalVolumeStore(args.bulletin_root, volume_name=args.volume_name or None)
    engine = SGLangEngine(args.upstream, args.local_checkpoint_dir)
    serve(
        store, engine,
        run_id=args.run_id, commit_mode=args.commit_mode,
        host=args.host, port=args.port, debug_requests=args.debug_requests,
    )


if __name__ == "__main__":
    main()
