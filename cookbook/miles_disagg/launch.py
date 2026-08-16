"""Launch an isolated Miles/Stitch run, optionally resuming its saved checkpoint.

A run's pool is a deployed Flash app the trainer reaches by name, so the run id has to be in the
app name before either half runs — which is why deploy and launch can't be one CLI call. This
closes that: mint the id, deploy ``app.py``'s pool under it, wait for it to be ready, then spawn
the trainer. A resume keeps that identity and recreates the rollout deployment
from its latest complete trainer/rollout checkpoint.

    EXPERIMENT_CONFIG=glm45_air_fp8 uv run --extra modal python -m cookbook.miles_disagg.launch

A plain script, not a ``modal run`` entrypoint: ``App.deploy()`` only persists outside a ``modal
run`` session (inside one the deployed app is torn down with the session). Minting the id here,
before importing the pool module, also lets the pool's app name resolve from ``RUN_ID`` at import.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import uuid
from collections.abc import MutableMapping
from typing import Any

import modal

from cookbook.miles_disagg.resume import (
    RESUME_POINT_ENV,
    ResumePoint,
    ResumePointNotFound,
    resolve_resume_point,
    restore_resume_point,
    validate_auto_resume_config,
    validate_resume_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-from",
        metavar="RUN_ID",
        help="resume RUN_ID from its latest complete Megatron/HF checkpoint pair",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="wait for training and recover unexpected termination from its latest checkpoint",
    )
    parser.add_argument(
        "--run-attempt",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.run_attempt:
        _run_attempt(supervise=True)
        return

    experiment = os.environ["EXPERIMENT_CONFIG"]
    exp = importlib.import_module(f"cookbook.miles_disagg.configs.{experiment}")
    if args.auto_resume:
        _run_auto_resume(exp, args.resume_from)
        return

    if args.resume_from is not None:
        validate_resume_config(exp.miles)
        resume_point = _resume_point_for_run(exp, args.resume_from)
        run_id = args.resume_from
    else:
        resume_point = None
        run_id = uuid.uuid4().hex[:8]
    _set_attempt_env(
        os.environ,
        run_id=run_id,
        resume_point=resume_point,
    )
    _run_attempt(supervise=False)


def _run_auto_resume(exp: Any, resume_from: str | None) -> None:
    validate_auto_resume_config(exp.miles)
    run_id = resume_from or uuid.uuid4().hex[:8]
    resume_point = (
        _resume_point_for_run(exp, resume_from) if resume_from is not None else None
    )
    while True:
        result = _run_supervised_attempt(
            run_id=run_id,
            resume_point=resume_point,
        )
        if result.returncode == 0:
            return

        resume_point = _resume_point_after_failure(exp, run_id, resume_point)
        print(
            f"Trainer stopped unexpectedly; resuming run {run_id} from "
            f"checkpoint v{resume_point.version}",
            flush=True,
        )


def _resume_point_for_run(exp: Any, run_id: str) -> ResumePoint:
    from cookbook.common import storage

    if storage.StoreDeployment.from_environment().backend != storage.MODAL_VOLUME:
        raise ValueError("Miles resume currently requires the Modal Volume store")
    volume = modal.Volume.from_name(exp.EXPERIMENT_VOLUME_NAME, version=2)
    resume_point = resolve_resume_point(
        volume,
        source_run_id=run_id,
        save_hf=exp.miles.save_hf,
    )
    print(
        f"Resolved resume checkpoint: run_id={resume_point.source_run_id}, "
        f"version={resume_point.version}, "
        f"trainer={resume_point.trainer_checkpoint}, "
        f"rollout={resume_point.rollout_checkpoint}",
        flush=True,
    )
    return resume_point


def _resume_point_after_failure(
    exp: Any, failed_run_id: str, previous: ResumePoint | None
) -> ResumePoint:
    try:
        return _resume_point_for_run(exp, failed_run_id)
    except ResumePointNotFound:
        if previous is None:
            raise
        print(
            f"Run {failed_run_id} produced no newer complete checkpoint; "
            f"retrying checkpoint v{previous.version}",
            flush=True,
        )
        return previous


def _set_attempt_env(
    env: MutableMapping[str, str],
    *,
    run_id: str,
    resume_point: ResumePoint | None,
) -> None:
    env["RUN_ID"] = run_id
    if resume_point is None:
        env.pop(RESUME_POINT_ENV, None)
        print(f"Launching fresh run {run_id}", flush=True)
        return
    if resume_point.source_run_id != run_id:
        raise ValueError(
            f"resume checkpoint belongs to run {resume_point.source_run_id!r}, "
            f"not {run_id!r}"
        )
    env[RESUME_POINT_ENV] = resume_point.to_json()
    print(
        f"Resuming run {run_id} from checkpoint v{resume_point.version}; "
        f"Stitch boot={run_id}/weight_v{resume_point.version:06d}",
        flush=True,
    )


def _run_supervised_attempt(
    *, run_id: str, resume_point: ResumePoint | None
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    _set_attempt_env(env, run_id=run_id, resume_point=resume_point)

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "cookbook.miles_disagg.launch",
            "--run-attempt",
        ],
        env=env,
        check=False,
    )


def _run_attempt(*, supervise: bool) -> None:
    from cookbook.common import launch
    from stitch.pools.modal_flash_lb_temp import ModalFlashLBPool
    from stitch.service import await_pool_ready

    run = importlib.import_module("cookbook.miles_disagg.app")
    resume_point = (
        ResumePoint.from_json(payload)
        if (payload := os.environ.get(RESUME_POINT_ENV))
        else None
    )
    if resume_point is None:
        trainer_call = launch.deploy_pool_and_spawn(run)
    else:
        # Replacement servers block before engine startup until this pointer is
        # restored, so none can reconcile against the abandoned suffix.
        run.app.deploy(strategy="recreate")
        volume = modal.Volume.from_name(run.exp.EXPERIMENT_VOLUME_NAME, version=2)
        target = restore_resume_point(volume, resume_point)
        print(f"Restored {target.identity}; waiting for the recreated pool", flush=True)
        await_pool_ready(
            ModalFlashLBPool(run.APP_NAME, "Server"),
            replica_floor=run.modal_cfg.rollout_min_containers,
        )
        trainer_call = run.spawn_train()

    if not supervise:
        print(
            f"run {os.environ['RUN_ID']} up on {run.APP_NAME}; "
            f"stop it with: modal app stop {run.APP_NAME}"
        )
        return
    print(
        f"run {os.environ['RUN_ID']} up on {run.APP_NAME}; "
        "the auto-resume launcher is supervising its trainer"
    )
    trainer_call.get()


if __name__ == "__main__":
    main()
