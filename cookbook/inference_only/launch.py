"""One-command launch: mint a unique run id, claim its boot pointer, and stand up
that run's inference pool.

A run's pool is a deployed Flash app clients reach by name, so the run id has to
be in the app name before deploy. This mints the id, deploys ``app.py``'s pool
under it, and waits for it to be ready. There is no trainer.

The boot-pointer claim writes through the checkpoint store, which for the default
Modal Volume backend is only reachable where the run volume is mounted — so the
claim runs as a one-shot function in the deployed app (``claim_boot_pointer`` in
``app.py``), invoked remotely right after deploy. Replicas only refresh + read
the pointer and wait for this claim, keeping a single writer.

    EXPERIMENT_CONFIG=glm5_2_fp8 uv run --extra modal python -m cookbook.inference_only.launch

A plain script, not a ``modal run`` entrypoint: ``App.deploy()`` only persists
outside a ``modal run`` session. Minting the id here, before importing the pool
module, also lets the pool's app name resolve from ``RUN_ID`` at import.
"""

from __future__ import annotations

import os
import uuid

# Name of the one-shot claim function deployed with the pool app.
CLAIM_FUNCTION_NAME = "claim_boot_pointer"


def main() -> None:
    if "EXPERIMENT_CONFIG" not in os.environ:
        raise SystemExit("EXPERIMENT_CONFIG is required")
    if not os.environ.get("RUN_ID"):
        os.environ["RUN_ID"] = uuid.uuid4().hex[:8]
    from cookbook.common import launch

    run = _load_run()
    run_id = os.environ["RUN_ID"]

    # Deploy first so the claim function exists, then claim remotely — the claim
    # lands seconds after deploy, well inside the replicas' pointer-wait window.
    launch.deploy_pool(run, after_deploy=lambda: _claim_boot_pointer_remote(run))
    print(
        f"run {run_id} up on {run.APP_NAME}; "
        f"stop it with: modal app stop {run.APP_NAME}"
    )


def _load_run():
    from cookbook.inference_only import app as run

    return run


def _claim_boot_pointer_remote(run) -> None:
    """Invoke the deployed app's one-shot boot-pointer claim (single writer)."""
    import modal

    claim = modal.Function.from_name(run.APP_NAME, CLAIM_FUNCTION_NAME)
    claim.remote()


if __name__ == "__main__":
    main()
