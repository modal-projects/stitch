"""Run offline eval points, each in its own app: deploy it, wait for the pool's floor,
run the driver to completion, and stop the app. A point whose metrics are complete is
skipped; an interrupted one runs again from scratch. Eval apps never touch training:
they have their own names and mount training volumes read-only.

    MODAL_FUNCTION_RUNTIME=runc MODAL_ENVIRONMENT=stitch-dev uv run --extra modal \\
      python -m cookbook.miles_disagg.eval_launch --spec swebench_pro_hetero \\
      --experiment qwen3_6_35b_a3b_hetero_score_centering --run r03 \\
      --versions 50,100 --views bf16,fp8,nvfp4 --engines 32

Version 0 is the base model, evaluated once under the spec's base recipe and shared by
every arm. ``--engines`` sizes each point's pool; more engines finish a point sooner.
Each point runs in a child process, so its app module imports with that point's
environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from cookbook.miles_disagg import evaluation

_POINT_FLAG = "--point-from-environment"
_WAIT_SECONDS = 600


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="module in eval_configs")
    parser.add_argument("--experiment", help="the evaluated recipe (an arm)")
    parser.add_argument("--run", help="the evaluated run id")
    parser.add_argument(
        "--versions", required=True, help="published versions, e.g. 0,50,100"
    )
    parser.add_argument("--views", default="bf16,fp8,nvfp4")
    parser.add_argument("--parallel", type=int, default=3, help="points at once")
    parser.add_argument(
        "--engines", type=int, help="engines per point's pool (default: the spec's)"
    )
    parser.add_argument(
        "--smoke",
        metavar="TASKSxSAMPLES",
        help="run a fixed subset, e.g. 50x1, into the point's smoke directory",
    )
    return parser


def main() -> None:
    if os.environ.get("MODAL_FUNCTION_RUNTIME") != "runc":
        raise SystemExit(
            "set MODAL_FUNCTION_RUNTIME=runc: eval apps must not run on gVisor"
        )
    if not os.environ.get("MODAL_ENVIRONMENT"):
        raise SystemExit("MODAL_ENVIRONMENT must name the environment for the eval")
    if sys.argv[1:] == [_POINT_FLAG]:
        raise SystemExit(_run_point())
    args = _parser().parse_args()
    spec = importlib.import_module(f"cookbook.miles_disagg.eval_configs.{args.spec}")
    points = _points(spec, args)
    if args.smoke is not None:
        _parse_smoke(args.smoke)
    with ThreadPoolExecutor(max(1, args.parallel)) as executor:
        codes = list(
            executor.map(
                lambda point: _spawn(args.spec, point, args.smoke, args.engines), points
            )
        )
    failed = [point for point, code in zip(points, codes, strict=True) if code != 0]
    for point in failed:
        print(f"FAILED: {point}", flush=True)
    raise SystemExit(1 if failed else 0)


def _points(spec: Any, args: argparse.Namespace) -> list[evaluation.EvalPoint]:
    points = []
    for version in sorted({int(value) for value in args.versions.split(",")}):
        for view in args.views.split(","):
            if view not in spec.POOLS:
                raise SystemExit(f"{args.spec} has no pool for view {view!r}")
            if version == 0:
                points.append(evaluation.EvalPoint(spec.BASE_EXPERIMENT, None, 0, view))
            elif args.experiment and args.run:
                points.append(
                    evaluation.EvalPoint(args.experiment, args.run, version, view)
                )
            else:
                raise SystemExit("versions above 0 need --experiment and --run")
    return points


def _spawn(
    spec_name: str,
    point: evaluation.EvalPoint,
    smoke: str | None,
    engines: int | None,
) -> int:
    env = {
        **os.environ,
        "EVAL_SMOKE": smoke or "",
        "EVAL_ENGINES": str(engines or ""),
        "EXPERIMENT_CONFIG": point.experiment,
        "EVAL_CONFIG": spec_name,
        "EVAL_RUN": point.run_id or "",
        "EVAL_VERSION": str(point.version),
        "EVAL_VIEW": point.view,
    }
    command = [sys.executable, "-m", "cookbook.miles_disagg.eval_launch", _POINT_FLAG]
    return subprocess.run(command, env=env, check=False).returncode


def _run_point() -> int:
    import modal
    from modal.exception import NotFoundError

    from stitch.pools.modal_flash import ModalFlashPool
    from stitch.service import await_pool_ready

    point_app = importlib.import_module("cookbook.miles_disagg.eval_app")
    point, name = point_app.POINT, point_app.APP_NAME
    label = f"[{name}]"
    smoke = _parse_smoke(os.environ.get("EVAL_SMOKE") or None)
    point_path = PurePosixPath(point_app.spec.NAME) / point.relative_dir
    if smoke is not None:
        point_path /= evaluation.smoke_dir(smoke)
    metrics = _read_json(point_app.eval_volume, point_path / "metrics.json")
    if metrics is not None and metrics.get("complete"):
        print(f"{label} complete; skipping", flush=True)
        return 0
    _require_checkpoint(point_app)
    pool = ModalFlashPool(name, point_app.POOL.name)
    try:
        pool.gateway_url()
    except (NotFoundError, RuntimeError):
        pass
    else:
        print(f"{label} already deployed; stop it first: modal app stop {name}")
        return 1
    print(
        f"{label} deploying; stop it any time with: modal app stop {name}", flush=True
    )
    try:
        point_app.app.deploy()
        await_pool_ready(pool, replica_floor=point_app.POOL.min_containers)
        _require_served_version(pool, point.version)
        call = modal.Function.from_name(name, "drive").spawn(
            _manifest(point_app), list(smoke) if smoke else None
        )
        print(f"{label} driver call {call.object_id}", flush=True)
        while True:
            try:
                summary = call.get(timeout=_WAIT_SECONDS)
                break
            except TimeoutError:
                print(f"{label} still running", flush=True)
        print(f"{label} done: {json.dumps(summary)}", flush=True)
        return 0 if summary.get("complete") else 1
    finally:
        subprocess.run(["modal", "app", "stop", "--yes", name], check=False)


def _require_served_version(pool: Any, version: int) -> None:
    """Every ready replica must report the point's version. The pool boots that
    checkpoint and nothing publishes to its store, so any other version means it is
    not serving what the point evaluates."""
    import asyncio

    from stitch.service import readiness

    state = asyncio.run(readiness(pool))
    served = {
        replica.applied.version if replica.applied is not None else None
        for replica in state.replicas
        if replica.ready
    }
    if served != {version}:
        raise SystemExit(f"pool serves versions {served}, expected only {version}")


def _require_checkpoint(point_app: Any) -> None:
    """Fail before deploying when the point's checkpoint is not complete."""
    import modal

    from cookbook.common.constants import CHECKPOINTS_PATH

    point, exp = point_app.POINT, point_app.exp
    path = evaluation.checkpoint_path(
        exp, point, source_run_root=evaluation.SOURCE_RUN_PATH
    )
    if point.is_base:
        volume = modal.Volume.from_name("miles-checkpoints", version=2)
        marker = PurePosixPath(path.relative_to(CHECKPOINTS_PATH)) / "config.json"
    else:
        volume = modal.Volume.from_name(exp.EXPERIMENT_VOLUME_NAME, version=2)
        marker = (
            PurePosixPath(path.relative_to(evaluation.SOURCE_RUN_PATH)) / ".complete"
        )
    names = {Path(entry.path).name for entry in volume.listdir(str(marker.parent))}
    if marker.name not in names:
        raise SystemExit(f"no complete checkpoint for {point}: {marker} is missing")


def _manifest(point_app: Any) -> dict[str, Any]:
    from cookbook.miles_disagg import trainer_image

    point, pool, exp = point_app.POINT, point_app.POOL, point_app.exp
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return {
        "spec": point_app.spec.NAME,
        "experiment": point.experiment,
        "run_id": point.run_id,
        "version": point.version,
        "view": point.view,
        "app": point_app.APP_NAME,
        "checkpoint": str(
            evaluation.checkpoint_path(
                exp, point, source_run_root=evaluation.SOURCE_RUN_PATH
            )
        ),
        "pool": {
            "name": pool.name,
            "gpu": pool.gpu,
            "gpus_per_engine": pool.gpus_per_engine,
            "engines": pool.min_containers,
            "target_inputs": pool.target_inputs,
            "sglang_args": pool.sglang_args,
            "environment": pool.environment,
        },
        "stitch_commit": head,
        "stitch_dirty": bool(dirty),
        "miles_commit": getattr(exp, "MILES_REPO_REF", trainer_image.MILES_REPO_REF),
        "launched_at": datetime.now(UTC).isoformat(),
    }


def _parse_smoke(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        tasks, samples = (int(part) for part in value.lower().split("x"))
    except ValueError:
        raise SystemExit(
            f"--smoke takes TASKSxSAMPLES, e.g. 50x1; got {value!r}"
        ) from None
    if tasks < 1 or samples < 1:
        raise SystemExit(f"--smoke needs positive counts; got {value!r}")
    return tasks, samples


def _read_json(volume: Any, path: PurePosixPath) -> dict[str, Any] | None:
    try:
        return json.loads(b"".join(volume.read_file(str(path))))
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    main()
