"""Launch an isolated standalone rollout pool.

A run's pool is a deployed Flash app an external trainer or harness reaches by name, so the
run id has to be in the app name before the pool exists. This mints the id, deploys
``app.py``'s pool under it, waits for the replica floor, and prints the endpoints the
publisher and rollout clients need. A fresh launch is isolated even from an
identical-config launch; pass ``--run-id`` to recreate an existing pool's deployment.

    EXPERIMENT_CONFIG=glm5_2_fp8 uv run --extra modal python -m cookbook.standalone.launch

A plain script, not a ``modal run`` entrypoint: ``App.deploy()`` only persists outside a
``modal run`` session. Minting the id here, before importing the pool module, lets the
pool's app name resolve from ``RUN_ID`` at import.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import uuid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        help="reuse an existing run identity instead of minting one",
    )
    args = parser.parse_args()
    os.environ["RUN_ID"] = (
        args.run_id or os.environ.get("RUN_ID") or uuid.uuid4().hex[:8]
    )

    from stitch.pools.modal_flash_lb_temp import ModalFlashLBPool
    from stitch.service import await_pool_ready

    run = importlib.import_module("cookbook.standalone.app")
    print(f"Deploying pool {run.APP_NAME}", flush=True)
    run.app.deploy()
    pool = ModalFlashLBPool(run.APP_NAME, "Server")
    if not await_pool_ready(pool, replica_floor=run.modal_cfg.rollout_min_containers):
        print(f"Pool {run.APP_NAME} did not reach its replica floor", flush=True)
        sys.exit(1)
    print(f"Pool ready: run_id={run.RUN_ID}")
    print(f"  publications: {run.RUN_DIR}/updates on {run.exp.EXPERIMENT_VOLUME_NAME}")
    print(f"  pool gateway: {pool.gateway_url()}")
    print(f"  rollout traffic: {run.Router.get_url()}")


if __name__ == "__main__":
    main()
