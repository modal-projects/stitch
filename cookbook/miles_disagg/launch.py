"""One-command launch: mint a unique run id, stand up that run's pool, start training.

A run's pool is a deployed Flash app the trainer reaches by name, so the run id has to be in the
app name before either half runs — which is why deploy and launch can't be one CLI call. This
closes that: mint the id, deploy ``app.py``'s pool under it, spawn the trainer. Each launch is its
own run — two launches of an identical config are two isolated runs (distinct ids), like commits.

    EXPERIMENT_CONFIG=glm45_air_fp8 uv run --extra modal python -m cookbook.miles_disagg.launch

A plain script, not a ``modal run`` entrypoint: ``App.deploy()`` only persists outside a ``modal
run`` session (inside one the deployed app is torn down with the session). Minting the id here,
before importing the pool module, also lets the pool's app name resolve from ``RUN_ID`` at import.
"""

from __future__ import annotations

import os
import uuid


def main() -> None:
    os.environ["RUN_ID"] = uuid.uuid4().hex[:8]
    from cookbook.miles_disagg import app as run

    run.app.deploy()
    run.spawn_train()
    print(f"run {os.environ['RUN_ID']} up on {run.APP_NAME}; stop it with: modal app stop {run.APP_NAME}")


if __name__ == "__main__":
    main()
