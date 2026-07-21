#!/usr/bin/env bash
# Launch one isolated run: auto-mint a RUN signature that scopes its own pool app + delta root,
# then deploy the pool and spawn the trainer against it. Set RUN yourself to target an existing run.
set -euo pipefail
: "${EXPERIMENT_CONFIG:?set EXPERIMENT_CONFIG}"
export RUN="${RUN:-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')}"
echo "run signature: ${EXPERIMENT_CONFIG} / ${RUN}"

uv run --extra modal modal deploy -m cookbook.miles_disagg.app
uv run --extra modal modal run -m cookbook.miles_disagg.app::launch_train
