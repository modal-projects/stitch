"""Offline evaluation of saved checkpoints, kept out of training.

An eval point is one weight view of one checkpoint: a run's export at a published
version, or the base model at version 0. Everything about the harness (agent, sandbox
limits, context, functions, session server) is the evaluated recipe's own; an eval spec
adds only what an eval owns (dataset, sample count, sampler) and names the recipe pool
whose configuration serves each view. Nothing here imports Modal or Miles.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from cookbook.miles_disagg.config import MilesConfig

# Eval apps mount this volume where training apps mount their run volume (STITCH_PATH),
# and the evaluated run's volume, read-only, at SOURCE_RUN_PATH.
EVAL_VOLUME_NAME = "stitch-swebench-pro-eval"
SOURCE_RUN_PATH = Path("/source-run")

# Every eval dataset generates through this hook: the recipe's generate function, rerun
# when an episode aborts on infrastructure.
EVAL_GENERATE_FUNCTION = "cookbook.miles_disagg.eval_hooks.generate"

# Miles fields an eval may change; everything else must match the training recipe.
EVAL_FIELD_OVERRIDES = frozenset(
    {
        "eval_config",
        # The gate pins each request to the training run's newest weights, so it would
        # reject every request to a pool serving one fixed checkpoint.
        "custom_rollout_request_hook_path",
        "custom_rollout_request_hook_args",
        # Capacity, sized to the eval pool.
        "async_max_concurrent_samples",
        "session_server_workers",
        # Outputs: the eval dump, and W&B per point on a version axis instead of
        # Miles' per-job run.
        "save_debug_rollout_data",
        "use_wandb",
    }
)
EVAL_ENVIRONMENT_OVERRIDES = frozenset(
    {
        "MODAL_SWE_TASKS_DIR",
        "MODAL_SWE_SANDBOX_APP",
        "MODAL_SWE_AGENT_PROCESSES",
    }
)


@dataclass(frozen=True)
class EvalPoint:
    """``experiment`` is the training recipe whose harness the eval uses. Base points
    (version 0) have no run, and every arm built on the same base shares them."""

    experiment: str
    run_id: str | None
    version: int
    view: str

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError(f"version must be >= 0, got {self.version}")
        if (self.version == 0) != (self.run_id is None):
            raise ValueError(
                "version 0 is the base model, which has no run; others need one"
            )

    @property
    def is_base(self) -> bool:
        return self.version == 0

    @property
    def relative_dir(self) -> PurePosixPath:
        if self.is_base:
            return PurePosixPath("base") / self.view
        return (
            PurePosixPath(self.experiment)
            / str(self.run_id)
            / f"v{self.version:06d}"
            / self.view
        )

    @property
    def slug(self) -> str:
        if self.is_base:
            return f"base-{self.view}"
        return f"{self.run_id}-v{self.version}-{self.view}"


def task_set(spec: Any) -> str:
    """The spec's prepared task set: the directory of its dataset, which a new
    preparation of the benchmark gets anew (``swebench-pro-v2``)."""
    return PurePosixPath(spec.DATASET["path"]).parent.name


def results_path(spec: Any, point: EvalPoint) -> PurePosixPath:
    """A point's results on the eval volume, under its task set, so results from
    different preparations of a benchmark never mix."""
    return PurePosixPath(task_set(spec)) / point.relative_dir


def app_name(spec_name: str, recipe_app_name: str, point: EvalPoint) -> str:
    """The point's Modal app. It also names the pool's store on the eval volume, which
    nothing publishes to, so the pool keeps serving its boot checkpoint."""
    if point.is_base:
        name = f"stitch-eval-{spec_name}-{point.slug}"
    else:
        name = f"{recipe_app_name}-eval-{point.slug}"
    if len(name) > 64:
        raise ValueError(f"app name {name!r} exceeds Modal's 64 characters")
    return name


def checkpoint_path(exp: Any, point: EvalPoint, *, source_run_root: Path) -> Path:
    """The checkpoint a point serves: the recipe's static view for the base model, else
    the run's export of that view."""
    if point.is_base:
        return Path(exp.ROLLOUT_WEIGHT_VIEWS[point.view])
    # The export saved at rollout N serves published version N + 1.
    export = exp.miles.save_hf.format(rollout_id=point.version - 1)
    return source_run_root / str(point.run_id) / export / point.view


def checkpoint_dir(exp: Any, point: EvalPoint, *, source_run_root: Path) -> Path:
    """``checkpoint_path``, once it holds a whole checkpoint. An export counts only
    once its ``.complete`` marker is present, which the uploader writes after every
    other file is durable."""
    path = checkpoint_path(exp, point, source_run_root=source_run_root)
    marker = path / ("config.json" if point.is_base else ".complete")
    if not marker.is_file():
        raise FileNotFoundError(
            f"no complete checkpoint at {path}: {marker.name} missing"
        )
    return path


def eval_miles_config(
    miles_cfg: MilesConfig,
    *,
    dataset: Mapping[str, Any],
    tasks_dir: Path,
    sandbox_app: str,
    concurrency: int,
    dump_template: str,
) -> MilesConfig:
    """The training recipe's Miles config, changed only as ``EVAL_*_OVERRIDES`` allow."""
    cfg: Any = MilesConfig.from_payload(miles_cfg.to_payload())
    # Same sessions per session server and agent threads per process as training.
    cfg.session_server_workers = math.ceil(
        concurrency * cfg.session_server_workers / cfg.async_max_concurrent_samples
    )
    cfg.async_max_concurrent_samples = concurrency
    datasets = [{**dataset, "custom_generate_function_path": EVAL_GENERATE_FUNCTION}]
    document = json.dumps({"eval": {"datasets": datasets}})
    cfg.eval_config = "base64:" + base64.b64encode(document.encode()).decode()
    cfg.custom_rollout_request_hook_path = cfg.custom_rollout_request_hook_args = None
    cfg.save_debug_rollout_data = dump_template
    cfg.use_wandb = False
    threads = int(cfg.environment["MODAL_SWE_AGENT_THREADS_PER_PROCESS"])
    cfg.environment = {
        **cfg.environment,
        "MODAL_SWE_TASKS_DIR": str(tasks_dir),
        "MODAL_SWE_SANDBOX_APP": sandbox_app,
        "MODAL_SWE_AGENT_PROCESSES": str(math.ceil(concurrency / threads)),
    }
    return cfg


def config_drift(train: MilesConfig, evaluated: MilesConfig) -> list[str]:
    """Fields and environment keys where an eval config departs from its training
    recipe beyond what ``EVAL_*_OVERRIDES`` allow."""
    a, b = train.to_payload(), evaluated.to_payload()
    fields = set(a["fields"]) | set(b["fields"])
    drift = [
        name
        for name in sorted(fields - EVAL_FIELD_OVERRIDES)
        if a["fields"].get(name) != b["fields"].get(name)
    ]
    keys = set(a["environment"]) | set(b["environment"])
    drift += [
        f"environment.{key}"
        for key in sorted(keys - EVAL_ENVIRONMENT_OVERRIDES)
        if a["environment"].get(key) != b["environment"].get(key)
    ]
    for name in ("async_mode", "megatron_model_type"):
        if a[name] != b[name]:
            drift.append(name)
    return drift


def smoke_tasks(instance_ids: Iterable[str], count: int) -> list[str]:
    """A fixed subset of ``count`` tasks spread across the dataset by hash, so a smoke
    run covers its repositories rather than whichever one comes first."""
    ids = sorted(
        set(instance_ids), key=lambda value: hashlib.sha256(value.encode()).digest()
    )
    if not 0 < count <= len(ids):
        raise ValueError(f"smoke needs 1..{len(ids)} tasks, got {count}")
    return ids[:count]


def smoke_dir(smoke: tuple[int, int]) -> str:
    """Where a smoke run of (tasks, samples) writes, beside the point's real results."""
    tasks, samples = smoke
    return f"smoke-{tasks}x{samples}"


# ── Results ─────────────────────────────────────────────────────────────────────────


def results_from_samples(
    samples: Iterable[Any], *, n_samples: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-sample records from Miles' eval output: scored samples, and samples whose
    episode still aborted after its retries. ``eval_hooks.generate`` stamps each
    sample with its attempts and its eval index (task * n_samples + sample)."""
    records, failures = [], []
    for sample in samples:
        metadata = sample.metadata or {}
        status = getattr(sample.status, "value", sample.status)
        record = {
            "instance_id": metadata["instance_id"],
            "sample_index": metadata["eval_sample_index"] % n_samples,
            "attempts": metadata.get("eval_attempts"),
            "status": status,
        }
        if status == "aborted" or sample.reward is None:
            failures.append(record)
            continue
        records.append(
            {
                **record,
                "reward": float(sample.reward),
                "response_length": sample.response_length,
                "exit_status": metadata.get("exit_status"),
                "agent_metrics": metadata.get("agent_metrics"),
            }
        )
    return records, failures


def pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased pass@k estimate for one task with ``c`` of ``n`` samples correct
    (Chen et al., 2021)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n} c={c} k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(
    records: Iterable[Mapping[str, Any]], *, n_samples: int, n_tasks: int
) -> dict[str, Any]:
    """pass@1..n over tasks with all ``n_samples`` scored, with the standard error over
    tasks. Incomplete points report how far along they are and no pass@k."""
    by_task: dict[str, dict[int, float]] = defaultdict(dict)
    for record in records:
        by_task[record["instance_id"]][record["sample_index"]] = float(record["reward"])
    complete = {
        task: rewards for task, rewards in by_task.items() if len(rewards) == n_samples
    }
    summary: dict[str, Any] = {
        "tasks": n_tasks,
        "tasks_complete": len(complete),
        "samples_scored": sum(len(rewards) for rewards in by_task.values()),
        "complete": len(complete) == n_tasks,
    }
    if not summary["complete"]:
        return summary
    for k in range(1, n_samples + 1):
        values = [
            pass_at_k(n_samples, sum(r >= 1.0 for r in rewards.values()), k)
            for rewards in complete.values()
        ]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
        summary[f"pass@{k}"] = mean
        summary[f"pass@{k}_se"] = math.sqrt(variance / len(values))
    return summary
