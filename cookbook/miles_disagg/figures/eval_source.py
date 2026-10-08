"""Eval points from the eval Volume: every finished point of a recipe's run, checked and
scored the way the eval scores it.

A point is finished once its ``metrics.json`` reports ``complete``; the launcher never
reruns a finished point, so a cached finished point is never fetched again. Each point is
checked against its own directory before it is used (``check_point``), so a figure can
never show one experiment's samples under another's name.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from cookbook.miles_disagg import evaluation

HARD51_PATH = (
    Path(__file__).resolve().parents[1]
    / "eval_configs"
    / "swebench_pro_scale_v2_hard51.txt"
)
SUBSETS = ("full", "hard51")
FULL_VOCABULARY = (1.0, 1.0, -1)
# The eval's scoring rules, applied here to every point's samples so every point in a
# figure follows the same rules whatever code evaluated it:
# - a turn still generating at the turn time limit fails its episode (code dc573c6 on);
# - an episode that ends without the agent's submit command fails, whatever its diff
#   grades to (user, 2026-10-06), as the training recipes from fix/training-signal score it.
TURN_TIME_LIMIT_SECONDS = 300.0
SCORING_RULE = f"turn_time_limit_{int(TURN_TIME_LIMIT_SECONDS)}s+require_submission"


def over_turn_time_limit(
    row: Mapping[str, Any], limit: float = TURN_TIME_LIMIT_SECONDS
) -> bool:
    """Whether a sample breaks the turn time limit: it ended on the limit, a model
    request in its final attempt took the limit or longer (resent or not), or an earlier
    attempt aborted on a request deadline. Earlier attempts keep only their abort
    reason, so points from before code 55c2193, which recorded no reasons, can only be
    scored on their final attempt."""
    if row.get("exit_status") == "TurnTimeLimit":
        return True
    metrics = row.get("agent_metrics") or {}
    if any(
        d >= limit for d in metrics.get("client_model_request_durations_seconds") or ()
    ):
        return True
    return any(
        "deadline exceeded" in str(reason).lower() for reason in row.get("aborts") or ()
    )


def unsubmitted(row: Mapping[str, Any]) -> bool:
    """Whether a sample's episode ended without the agent's submit command."""
    return row.get("exit_status") != "Submitted"


def scored_as_failure(row: Mapping[str, Any]) -> bool:
    """Whether the scoring rules fail a sample whatever its graded reward."""
    return over_turn_time_limit(row) or unsubmitted(row)


@dataclass(frozen=True)
class EvalPoint:
    """One finished point: its identity, code, and per-sample rewards."""

    recipe: str | None  # None for the base model
    run_id: str | None
    version: int
    n_samples: int
    commit: str
    dirty: bool
    path: str
    rewards: tuple[tuple[str, int, float], ...] = field(repr=False)

    def scores(self, subset: str, hard: frozenset[str]) -> dict[str, float]:
        """pass@1..n with standard errors, on the full set or on the hard subset."""
        records = [
            {"instance_id": task, "sample_index": index, "reward": reward}
            for task, index, reward in self.rewards
            if subset == "full" or task in hard
        ]
        tasks = {record["instance_id"] for record in records}
        summary = evaluation.summarize(
            records, n_samples=self.n_samples, n_tasks=len(tasks)
        )
        if not summary["complete"]:
            raise ValueError(f"{self.path}: {subset} subset is incomplete")
        return {key: value for key, value in summary.items() if key.startswith("pass@")}


def hard_tasks(path: Path = HARD51_PATH) -> frozenset[str]:
    return frozenset(path.read_text().split())


def check_point(
    *,
    recipe: str | None,
    run_id: str | None,
    version: int,
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    n_tasks: int,
    excluded: Iterable[str],
) -> list[str]:
    """Everything wrong with a point, checked against the directory it was read from."""
    problems = []
    if recipe is None:
        if manifest.get("version") != 0 or manifest.get("run_id") is not None:
            problems.append("base point is not version 0 without a run")
    else:
        for name, want in (
            ("experiment", recipe),
            ("run_id", run_id),
            ("version", version),
            ("view", "bf16"),
        ):
            if manifest.get(name) != want:
                problems.append(
                    f"manifest {name} {manifest.get(name)!r}, directory {want!r}"
                )
        checkpoint = (
            f"/source-run/{run_id}/hf_checkpoints/weight_v{version - 1:06d}/bf16"
        )
        if manifest.get("checkpoint") != checkpoint:
            problems.append(
                f"checkpoint {manifest.get('checkpoint')!r}, expected {checkpoint!r}"
            )
    dataset = manifest.get("dataset", {})
    decoding = (dataset.get("temperature"), dataset.get("top_p"), dataset.get("top_k"))
    if decoding != FULL_VOCABULARY:
        problems.append(f"decoding {decoding} is not full-vocabulary sampling")
    n = dataset.get("n_samples_per_eval_prompt")
    per_task: dict[str, set[int]] = {}
    duplicates = 0
    for row in rows:
        indices = per_task.setdefault(row["instance_id"], set())
        duplicates += row["sample_index"] in indices
        indices.add(row["sample_index"])
    if duplicates:
        problems.append(f"{duplicates} duplicate (task, sample) rows")
    if len(per_task) != n_tasks:
        problems.append(f"{len(per_task)} tasks, expected {n_tasks}")
    if set(per_task) & set(excluded):
        problems.append("an excluded task was scored")
    if any(indices != set(range(n)) for indices in per_task.values()):
        problems.append(f"not every task has samples 0..{n - 1}")
    if not metrics.get("complete"):
        problems.append("metrics.json is not complete")
    return problems


class VolumeReader:
    """The eval Volume through Modal's API: list a directory, read a file."""

    def __init__(self, environment: str = "stitch-dev"):
        import modal

        self._volume = modal.Volume.from_name(
            evaluation.EVAL_VOLUME_NAME, environment_name=environment
        )

    # Modal's API fails a few percent of Volume calls with a transient InternalError
    # (2026-10-07: about one in 20 listings for an hour); a build makes ~100 of them.
    ATTEMPTS = 4
    BACKOFF_S = 2.0

    def _retrying(self, call: Callable[[], Any]) -> Any:
        from modal.exception import InternalError

        for attempt in range(1, self.ATTEMPTS + 1):
            try:
                return call()
            except InternalError:
                if attempt == self.ATTEMPTS:
                    raise
                time.sleep(self.BACKOFF_S * attempt)
        raise AssertionError("unreachable")

    def listdir(self, path: str) -> list[str]:
        from modal.exception import NotFoundError

        # Only a missing directory lists as empty. Any other error must fail the build:
        # read as empty, it would silently drop a finished point from the figures.
        try:
            return self._retrying(
                lambda: [
                    PurePosixPath(entry.path).name
                    for entry in self._volume.listdir(path)
                ]
            )
        except NotFoundError:
            return []

    def read(self, path: str) -> bytes:
        return self._retrying(lambda: b"".join(self._volume.read_file(path)))


class EvalSource:
    """Finished eval points on the Volume, cached under ``cache_dir``."""

    def __init__(
        self,
        spec: Any,
        cache_dir: Path,
        *,
        reader: Any | None = None,
        refresh: bool = False,
    ):
        self.spec = spec
        self.task_set = evaluation.task_set(spec)
        self.cache_dir = cache_dir / "eval" / self.task_set
        self.refresh = refresh
        self._reader = reader
        self.task_ids: frozenset[str] | None = None
        # Each point's metrics.json as read this build, to validate the cache with.
        self._metrics: dict[str, dict[str, Any]] = {}

    @property
    def reader(self) -> Any:
        if self._reader is None:
            self._reader = VolumeReader()
        return self._reader

    def _remote(
        self, recipe: str | None, run_id: str | None, version: int | None = None
    ) -> str:
        if recipe is None:
            return f"{self.task_set}/base/bf16"
        root = f"{self.task_set}/{recipe}/{run_id}"
        return root if version is None else f"{root}/v{version:06d}/bf16"

    def versions(self, recipe: str, run_id: str) -> list[int]:
        """The run's versions with a finished bf16 point, ascending."""
        found = []
        for name in self.reader.listdir(self._remote(recipe, run_id)):
            if not (name.startswith("v") and name[1:].isdigit()):
                continue
            version = int(name[1:])
            if self._finished(self._remote(recipe, run_id, version)):
                found.append(version)
        return sorted(found)

    def _finished(self, remote: str) -> bool:
        if "metrics.json" not in self.reader.listdir(remote):
            return False
        return bool(self._read_metrics(remote).get("complete"))

    def _read_metrics(self, remote: str) -> dict[str, Any]:
        if remote not in self._metrics:
            self._metrics[remote] = json.loads(
                self.reader.read(f"{remote}/metrics.json")
            )
        return self._metrics[remote]

    def point(self, recipe: str | None, run_id: str | None, version: int) -> EvalPoint:
        remote = self._remote(recipe, run_id, version)
        cached = (
            self.cache_dir / remote.removeprefix(f"{self.task_set}/") / "point.json"
        )
        # A point rerun at the same path (after its first result was archived) must not
        # be served from the old result's cache: use the cache only while its scores
        # still match the point's metrics.json.
        metrics = self._read_metrics(remote)
        scores = {k: v for k, v in metrics.items() if k.startswith("pass@")}
        payload = None
        if cached.is_file() and not self.refresh:
            payload = json.loads(cached.read_text())
            if payload["metrics"] != scores or payload.get("rule") != SCORING_RULE:
                payload = None
        if payload is None:
            manifest = json.loads(self.reader.read(f"{remote}/manifest.json"))
            rows = [
                json.loads(line)
                for line in self.reader.read(f"{remote}/samples.jsonl")
                .decode()
                .splitlines()
                if line
            ]
            problems = check_point(
                recipe=recipe,
                run_id=run_id,
                version=version,
                manifest=manifest,
                metrics=metrics,
                rows=rows,
                n_tasks=self.spec.TASKS,
                excluded=self.spec.EXCLUDED_TASKS,
            )
            if problems:
                raise ValueError(f"eval point {remote} failed its checks: {problems}")
            payload = {
                "recipe": recipe,
                "run_id": run_id,
                "version": version,
                "n_samples": manifest["dataset"]["n_samples_per_eval_prompt"],
                "commit": manifest.get("stitch_commit", ""),
                "dirty": bool(manifest.get("stitch_dirty")),
                "path": remote,
                "metrics": {k: v for k, v in metrics.items() if k.startswith("pass@")},
                "rule": SCORING_RULE,
                "over_turn_time_limit": sum(over_turn_time_limit(row) for row in rows),
                "unsubmitted": sum(unsubmitted(row) for row in rows),
                "rewards": [
                    [
                        row["instance_id"],
                        row["sample_index"],
                        0.0 if scored_as_failure(row) else float(row["reward"] or 0.0),
                    ]
                    for row in rows
                ],
            }
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(payload))
        point = EvalPoint(
            recipe=payload["recipe"],
            run_id=payload["run_id"],
            version=payload["version"],
            n_samples=payload["n_samples"],
            commit=payload["commit"],
            dirty=payload["dirty"],
            path=payload["path"],
            rewards=tuple(
                (task, index, reward) for task, index, reward in payload["rewards"]
            ),
        )
        # Every point must score the same tasks, or their curves are not comparable.
        tasks = frozenset(task for task, _, _ in point.rewards)
        if self.task_ids is None:
            self.task_ids = tasks
        elif tasks != self.task_ids:
            raise ValueError(f"eval point {remote} scores a different task set")
        return point

    def base(self) -> EvalPoint:
        return self.point(None, None, 0)

    def run_points(self, recipe: str, run_id: str) -> list[EvalPoint]:
        return [
            self.point(recipe, run_id, version)
            for version in self.versions(recipe, run_id)
        ]


def rows_for_table(
    points: Mapping[str, list[EvalPoint]],
    labels: Callable[[str], str],
    hard: frozenset[str],
) -> list[dict[str, Any]]:
    """One row per point and subset, with every pass@k and its standard error."""
    out = []
    for key, run_points in points.items():
        for point in run_points:
            for subset in SUBSETS:
                out.append(
                    {
                        "experiment": key,
                        "label": labels(key),
                        "recipe": point.recipe or "base",
                        "run_id": point.run_id or "",
                        "step": point.version,
                        "subset": subset,
                        "n_samples": point.n_samples,
                        "commit": point.commit[:7],
                        "path": point.path,
                        **point.scores(subset, hard),
                    }
                )
    return out
