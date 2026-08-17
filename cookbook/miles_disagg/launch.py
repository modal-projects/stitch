"""Launch a Miles/Stitch run's trainer on its deployed pool.

Mint or accept the run id, wait for the pool's floor, then spawn the trainer.
The pool's lifecycle belongs to explicit deploys — first bring it up (and later
change its infra) with the same run id:

    EXPERIMENT_CONFIG=glm45_air_fp8 RUN_ID=myrun01 uv run --extra modal modal deploy -m cookbook.miles_disagg.app
    EXPERIMENT_CONFIG=glm45_air_fp8 RUN_ID=myrun01 uv run --extra modal python -m cookbook.miles_disagg.launch

Recovery is Modal's job, not this script's: the trainer retries with identical
input and re-derives its resume state from the run volume (see
``cookbook.miles_disagg.resume``), so nothing here supervises a running trainer
and a live pool is never redeployed or replaced here. ``--resume-from RUN_ID``
reuses an existing run's identity — for a run past its retry budget or a manual
takeover; the new trainer call supersedes the old one and resumes like any
retry.

Minting the id here, before importing the pool module, lets the pool's app name
resolve from ``RUN_ID`` at import.
"""

from __future__ import annotations

import argparse
import importlib
import os
import uuid
from typing import Any

from cookbook.miles_disagg.resume import validate_resumable_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-from",
        metavar="RUN_ID",
        help="reuse RUN_ID; its trainer resumes from the newest complete checkpoint pair",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    experiment = os.environ["EXPERIMENT_CONFIG"]
    exp = importlib.import_module(f"cookbook.miles_disagg.configs.{experiment}")

    if args.resume_from is not None:
        validate_resumable_config(exp.miles)
        run_id = args.resume_from
    else:
        run_id = os.environ.get("RUN_ID") or uuid.uuid4().hex[:8]
        _warn_unless_resumable(exp.miles)
    os.environ["RUN_ID"] = run_id

    from cookbook.common import launch

    run = importlib.import_module("cookbook.miles_disagg.app")
    call = launch.spawn_on_pool(run)
    print(
        f"run {run_id} up on {run.APP_NAME}; trainer call {call.object_id} "
        f"retries and resumes on its own — stop the run with: modal app stop {run.APP_NAME}"
    )


def _warn_unless_resumable(cfg: Any) -> None:
    try:
        validate_resumable_config(cfg)
    except ValueError as exc:
        print(
            f"WARNING: this config is not resumable ({exc}); "
            "a trainer retry restarts from scratch",
            flush=True,
        )


if __name__ == "__main__":
    main()
