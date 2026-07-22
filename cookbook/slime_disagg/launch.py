"""One-command launch: mint a unique run id, stand up that run's pool, start training.

A run's pool is a deployed Flash app the trainer reaches by name, so the run id has to be in the
app name before either half runs — which is why deploy and launch can't be one CLI call. This
closes that: mint the id, deploy ``app.py``'s pool under it, wait for it to be ready, then spawn
the trainer. Each launch is its own run — two launches of an identical config are two isolated
runs (distinct ids), like commits.

    EXPERIMENT_CONFIG=kimi_k2_6_int4 uv run --extra modal python -m cookbook.slime_disagg.launch

A plain script, not a ``modal run`` entrypoint: ``App.deploy()`` only persists outside a ``modal
run`` session (inside one the deployed app is torn down with the session). Minting the id here,
before importing the pool module, also lets the pool's app name resolve from ``RUN_ID`` at import.
"""

from __future__ import annotations

import os
import uuid


def main() -> None:
    os.environ["RUN_ID"] = uuid.uuid4().hex[:8]
    from cookbook.slime_disagg import app as run

    run.app.deploy()
    _await_pool_ready(run)
    run.spawn_train()
    print(f"run {os.environ['RUN_ID']} up on {run.APP_NAME}; stop it with: modal app stop {run.APP_NAME}")


def _await_pool_ready(run, timeout: float = 20 * 60) -> None:
    """Block until the pool answers /health, so the trainer's first rollout hits a ready pool rather
    than a 5xx storm while the engines are still loading. Spawn anyway on timeout (the trainer retries)."""
    import time

    import httpx

    gateway = run.ModalFlashPool(run.APP_NAME, "Server").gateway_url()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{gateway}/health", timeout=10).status_code == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(30)
    print(f"WARNING: {run.APP_NAME} pool not ready after {timeout:.0f}s; spawning trainer anyway")


if __name__ == "__main__":
    main()
