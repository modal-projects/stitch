"""Launch a standalone pool and claim its initial policy version.

The run identity must be set before importing the deployment module. This is a
plain script because App.deploy() persists only outside a modal run session.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import uuid

CLAIM_FUNCTION_NAME = "claim_boot_pointer"


def main() -> None:
    if "EXPERIMENT_CONFIG" not in os.environ:
        raise SystemExit("EXPERIMENT_CONFIG is required")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        help="reuse an existing run identity instead of minting one",
    )
    args = parser.parse_args()
    os.environ["RUN_ID"] = (
        args.run_id or os.environ.get("RUN_ID") or uuid.uuid4().hex[:8]
    )

    from stitch.pools.modal_flash import ModalFlashPool
    from stitch.service import await_pool_ready

    run = importlib.import_module("cookbook.standalone.app")
    print(f"Deploying pool {run.APP_NAME}", flush=True)
    run.app.deploy()

    import modal

    # The default checkpoint store is only accessible where its Volume is
    # mounted, so claim through the deployed app before waiting for replicas.
    modal.Function.from_name(run.APP_NAME, CLAIM_FUNCTION_NAME).remote()
    pool = ModalFlashPool(run.APP_NAME, "Server")
    if not await_pool_ready(pool, replica_floor=run.modal_cfg.rollout_min_containers):
        print(f"Pool {run.APP_NAME} did not reach its replica floor", flush=True)
        sys.exit(1)
    print(f"Pool ready: run_id={run.RUN_ID}")
    print(f"  publications: {run.RUN_DIR}/updates on {run.exp.EXPERIMENT_VOLUME_NAME}")
    print(f"  pool gateway: {pool.gateway_url()}")


if __name__ == "__main__":
    main()
