"""Run one eval point through Miles' own eval, against the point's pool.

``run_eval_datasets`` is where Miles' eval-only mode (``train.py --num-rollout 0``)
ends up, and what its external-eval backends call. Calling it directly leaves out the
trainer and its Ray placement, which an eval against external engines has no use for.
Arguments come from the recipe's CLI args as the trainer builds them, and the session
servers are Miles' standalone ones, pointed at the pool. Runs in the trainer image.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cookbook.common import launch
from cookbook.common.constants import MODAL_SESSION_ID_HEADER
from cookbook.miles_disagg import evaluation
from cookbook.miles_disagg.config import YAML_CONFIG_FIELDS

logger = logging.getLogger(__name__)

_SESSION_SERVER_READY_SECONDS = 300.0


def miles_args(cfg: Any, *, pool_url: str) -> Any:
    """Miles' parsed arguments for an eval config, wired to the eval pool the way the
    trainer wires a run's arguments to its fleet."""
    from miles.utils.arguments import parse_args
    from miles.utils.external_utils.model_args_utils import load_model_args

    cfg.rollout_endpoint_url = pool_url
    cfg.rollout_session_affinity_header = MODAL_SESSION_ID_HEADER
    # An eval saves nothing and publishes nothing.
    cfg.save_interval = cfg.save = cfg.save_hf = None
    cfg.update_weight_disk_dir = tempfile.mkdtemp(prefix="eval-unused-updates-")
    launch.resolve_config(
        cfg,
        tempfile.mkdtemp(),
        checkpoint_fields=("hf_checkpoint", "load", "ref_load", "critic_load"),
        yaml_fields=YAML_CONFIG_FIELDS,
    )
    argv = cfg.cli_args()
    if cfg.megatron_model_type:
        argv = [*shlex.split(load_model_args(cfg.megatron_model_type)), *argv]
    saved = sys.argv
    sys.argv = ["eval", *argv]
    try:
        return parse_args()
    finally:
        sys.argv = saved


def start_session_servers(args: Any, *, backend_url: str) -> list[subprocess.Popen]:
    """Start the session servers Miles would, pointed at the eval pool, and publish
    them on ``args`` as Miles' tracer expects."""
    import httpx
    from miles.ray.specs.inference import compute_session_server_instance_id
    from miles.rollout.session.config import compute_session_server_config
    from miles.rollout.session.types import SessionServerInstance
    from miles.utils.workers.argv_utils import config_to_argv

    processes, instances = [], []
    base_port = args.session_server_port or 30000
    for index in range(args.session_server_workers):
        config = compute_session_server_config(
            args,
            host="127.0.0.1",
            port=base_port + index,
            instance_id=compute_session_server_instance_id(args, index),
            backend_url=backend_url,
        )
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "miles.rollout.session.server",
                    *config_to_argv(config),
                ]
            )
        )
        instances.append(
            SessionServerInstance(
                addr=f"127.0.0.1:{config.port}", instance_id=config.instance_id
            )
        )
    deadline = time.monotonic() + _SESSION_SERVER_READY_SECONDS
    for instance, process in zip(instances, processes, strict=True):
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"session server {instance.addr} exited at startup")
            try:
                if httpx.get(f"{instance.url}/health", timeout=5.0).is_success:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"session server {instance.addr} never became ready")
            time.sleep(1.0)
    args.session_server_instances = instances
    return processes


def run(
    *,
    exp: Any,
    spec: Any,
    point: evaluation.EvalPoint,
    pool: Any,
    pool_url: str,
    point_dir: Path,
    manifest: dict[str, Any],
    commit: Callable[[], None],
    smoke: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Evaluate ``point`` on the pool at ``pool_url``; returns the point's summary. A
    ``smoke`` of (tasks, samples) runs that subset into its own directory instead."""
    import ray
    from miles.ray.rollout.debug_data import save_debug_rollout_data
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
    from miles.rollout.inference_rollout.inference_rollout_eval import (
        run_eval_datasets,
    )
    from miles.utils.http_utils import init_http_client

    results_dir = point_dir / evaluation.smoke_dir(smoke) if smoke else point_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset, n_tasks = dict(spec.DATASET), spec.TASKS
    if smoke is not None:
        n_tasks, dataset["n_samples_per_eval_prompt"] = smoke
        dataset["path"] = str(
            _write_subset(Path(dataset["path"]), results_dir / "tasks.jsonl", n_tasks)
        )
    concurrency = pool.min_containers * pool.target_inputs
    cfg = evaluation.eval_miles_config(
        exp.miles,
        dataset=dataset,
        tasks_dir=spec.TASKS_DIR,
        sandbox_app=spec.SANDBOX_APP,
        concurrency=concurrency,
        dump_template=str(results_dir / "dump" / "{rollout_id}.pt"),
    )
    # The agent, its Ray workers, and the session servers read the recipe environment.
    os.environ.update(cfg.environment)
    args = miles_args(cfg, pool_url=pool_url)
    args.eval_infra_retries = spec.INFRA_RETRIES
    (results_dir / "manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "concurrency": concurrency,
                "dataset": dataset,
                "smoke": smoke,
            },
            indent=2,
        )
        + "\n"
    )
    commit()
    servers = start_session_servers(args, backend_url=pool_url)
    try:
        # One logical CPU per agent-controller actor, as the agent pool requests.
        processes = int(cfg.environment["MODAL_SWE_AGENT_PROCESSES"])
        ray.init(include_dashboard=False, num_cpus=max(os.cpu_count() or 1, processes))

        async def evaluate() -> dict[str, Any]:
            # Miles' rollout manager opens this shared client before any eval request.
            init_http_client(args)
            return await run_eval_datasets(GenerateState(args), {})

        data = asyncio.run(evaluate())
    finally:
        for server in servers:
            server.terminate()
        ray.shutdown()
    save_debug_rollout_data(args, data, rollout_id=0, evaluation=True)
    (result,) = data.values()
    n_samples = dataset["n_samples_per_eval_prompt"]
    records, failures = evaluation.results_from_samples(
        result["samples"], n_samples=n_samples
    )
    for name, rows in (("samples.jsonl", records), ("infra_failures.jsonl", failures)):
        (results_dir / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {
        **evaluation.summarize(records, n_samples=n_samples, n_tasks=n_tasks),
        "infra_failures": len(failures),
    }
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    commit()
    if summary["complete"] and smoke is None:
        log_to_wandb(exp, spec, point, summary)
    return summary


def _write_subset(source: Path, destination: Path, count: int) -> Path:
    """The eval dataset restricted to ``evaluation.smoke_tasks``' fixed subset."""
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    chosen = set(
        evaluation.smoke_tasks((row["metadata"]["instance_id"] for row in rows), count)
    )
    destination.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in rows
            if row["metadata"]["instance_id"] in chosen
        )
    )
    return destination


def log_to_wandb(
    exp: Any, spec: Any, point: evaluation.EvalPoint, summary: dict[str, Any]
) -> None:
    """Add the point to its W&B run, one per (recipe, run, view) in the spec's eval
    group, on an ``eval/version`` axis so points can arrive in any order. The eval
    volume holds the results; a W&B failure loses nothing."""
    try:
        import wandb

        owner = "base" if point.is_base else f"{exp.APP_NAME}-{point.run_id}"
        name = f"{owner}-{point.view}"
        run = wandb.init(
            project=exp.miles.wandb_project,
            group=f"{spec.NAME}-eval",
            name=name,
            id=hashlib.sha256(f"{spec.NAME}/{name}".encode()).hexdigest()[:16],
            resume="allow",
            config={"spec": spec.NAME, "view": point.view, "run_id": point.run_id},
        )
        run.define_metric("eval/version")
        run.define_metric("eval/*", step_metric="eval/version")
        run.log(
            {
                "eval/version": point.version,
                **{
                    f"eval/{metric}": value
                    for metric, value in summary.items()
                    if metric.startswith("pass@")
                },
            }
        )
        run.finish()
    except Exception:  # noqa: BLE001
        logger.exception("eval: W&B logging failed; results are on the eval volume")
