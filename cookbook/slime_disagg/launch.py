"""Launch a Slime run's trainer on its deployed pool.

Mint the run id, wait for the pool's floor, then spawn the trainer. The pool's
lifecycle belongs to explicit deploys of ``app.py`` under the same run id; a
missing pool fails fast with that instruction. Each launch is its own run — two
launches of an identical config are two isolated runs (distinct ids), like
commits.

    EXPERIMENT_CONFIG=kimi_k2_6_int4 uv run --extra modal python -m cookbook.slime_disagg.launch

Minting the id here, before importing the pool module, lets the pool's app name
resolve from ``RUN_ID`` at import.
"""

from __future__ import annotations

import os
import uuid


def main() -> None:
    os.environ["RUN_ID"] = os.environ.get("RUN_ID") or uuid.uuid4().hex[:8]
    from cookbook.common import launch
    from cookbook.slime_disagg import app as run

    launch.spawn_on_pool(run)
    print(
        f"run {os.environ['RUN_ID']} up on {run.APP_NAME}; "
        f"stop it with: modal app stop {run.APP_NAME}"
    )


if __name__ == "__main__":
    main()
