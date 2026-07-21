"""One-command launch: mint a unique run id, stand up that run's pool, start training.

The pool is a deployed Flash app the trainer reaches by name, so the app name has to carry the
run id before either half runs — which is why deploy and launch can't just be one CLI call. This
entrypoint closes that: it mints the id, deploys ``app.py``'s pool app under it, and spawns the
trainer against it. Each launch is its own run — two launches of an identical config are two
isolated runs (distinct ids), like two commits.

    EXPERIMENT_CONFIG=glm45_air_fp8 uv run --extra modal modal run -m cookbook.miles_disagg.launch

Lives in its own app (not the run-scoped pool app) so it can mint the id before importing the
pool module, whose app name is fixed at import from ``RUN``.
"""

from __future__ import annotations

import os
import uuid

import modal

app = modal.App("miles-disagg-launch")


@app.local_entrypoint()
def main() -> None:
    os.environ["RUN_ID"] = uuid.uuid4().hex[:8]
    from cookbook.miles_disagg import app as run

    run.app.deploy()
    run.spawn_train()
    print(f"run {os.environ['RUN_ID']} up on {run.APP_NAME}; stop it with: modal app stop {run.APP_NAME}")
