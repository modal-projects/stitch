"""Launch a Miles/Stitch run whose trainer retries and resumes on its own.

A fresh launch mints a run id, deploys the run's pool, waits for its floor, and
spawns the trainer. Recovery is Modal's job, not this script's: the trainer
retries with identical input and re-derives its resume state from the run
volume (``cookbook.miles_disagg.resume``), so nothing here supervises a running
trainer and a live pool is never redeployed.

``--resume-from RUN_ID`` reuses a run's identity for a takeover or a run past
its retry budget: it cancels the recorded trainer call and spawns a successor
on the deployed pool. If the pool is gone, deploy it first under the same id:

    EXPERIMENT_CONFIG=<experiment> RUN_ID=<run> uv run --extra modal modal deploy -m cookbook.miles_disagg.app

A plain script, not a ``modal run`` entrypoint: ``App.deploy()`` only persists
outside a ``modal run`` session, and minting the id before importing the pool
module lets its app name resolve from ``RUN_ID`` at import.
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
import uuid
from typing import Any

from cookbook.miles_disagg.resume import read_trainer_call, validate_resumable_config

_CANCEL_TIMEOUT = 10 * 60


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

    from cookbook.common import launch

    if args.resume_from is not None:
        validate_resumable_config(exp.miles)
        os.environ["RUN_ID"] = args.resume_from
        run = importlib.import_module("cookbook.miles_disagg.app")
        _cancel_recorded_trainer_call(run)
        call = launch.spawn_on_pool(run)
    else:
        explicit = os.environ.get("RUN_ID")
        os.environ["RUN_ID"] = explicit or uuid.uuid4().hex[:8]
        _warn_unless_resumable(exp.miles)
        run = importlib.import_module("cookbook.miles_disagg.app")
        if explicit is not None and launch.pool_reachable(run):
            raise SystemExit(
                f"run {explicit!r} already has a live pool ({run.APP_NAME}); "
                f"resume it with --resume-from {explicit}, or stop it first: "
                f"modal app stop {run.APP_NAME}"
            )
        call = launch.deploy_pool_and_spawn(run)
    print(
        f"run {os.environ['RUN_ID']} up on {run.APP_NAME}; trainer call {call.object_id} "
        f"retries and resumes on its own — stop the run with: modal app stop {run.APP_NAME}"
    )


def _cancel_recorded_trainer_call(run: Any) -> None:
    """Cancel the run's live trainer call and wait for it to stop, so a takeover
    never briefly runs two trainers against one run."""
    import modal

    volume = modal.Volume.from_name(run.exp.EXPERIMENT_VOLUME_NAME, version=2)
    call_id = read_trainer_call(volume, os.environ["RUN_ID"])
    if call_id is None:
        return
    call = modal.FunctionCall.from_id(call_id)
    try:
        call.cancel(terminate_containers=True)
    except Exception as exc:  # noqa: BLE001 — a long-gone call is fine
        print(f"note: could not cancel trainer call {call_id}: {exc}", flush=True)
        return
    print(f"Cancelled trainer call {call_id}; waiting for it to stop", flush=True)
    # A cancelled call yields no output through .get(); only the call graph
    # shows its inputs settling.
    deadline = time.monotonic() + _CANCEL_TIMEOUT
    while time.monotonic() < deadline:
        if all(_input_settled(node) for node in call.get_call_graph()):
            return
        time.sleep(10)
    raise SystemExit(
        f"trainer call {call_id} did not stop within {_CANCEL_TIMEOUT}s; "
        "retry, or cancel it in the Modal dashboard first"
    )


def _input_settled(node: Any) -> bool:
    from modal.types import InputStatus

    return node.status != InputStatus.PENDING and all(
        _input_settled(child) for child in getattr(node, "children", [])
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
