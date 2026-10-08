"""Load ramp and the throughput-at-speed summary of the sampler benchmark.

``run_sweep`` ramps one variant's engines through a list of sessions per engine. Each
point grows the running replay (new sessions start mid-trajectory; old ones keep their
place), waits out a warm-up, measures one window, and appends a CSV row. The ramp stops
once the speed metric misses the loosest target, the engines saturate, or errors pass a
threshold.

The speed metric is per-turn latency at the 90th percentile by default: an agent turn
writes only ~77 tokens, so its decode rate is a noisy, prefill-dominated number, while
turn latency is what adds to an episode's wall time. Throughput-style metrics (decode or
end-to-end tok/s per request) remain available; for them higher is faster.

``summarize`` reads those rows. For every variant it finds the output throughput per GPU
at each target, interpolating between the two points whose speeds straddle the target. A configuration's figure is its best variant's; each is reported
relative to the B200 BF16 baseline, with the relative cost of a token at list prices.

    python -m cookbook.miles_disagg.bench.sweep run --url http://HOST:PORT \\
        --traces TRACES_DIR --out points.csv --config b200-bf16 --variant base
    python -m cookbook.miles_disagg.bench.sweep summarize points/*.csv --out summary.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cookbook.miles_disagg.bench import configs, replay

# Finer steps between 16 and 32 sessions, where the B200 BF16 KV cache fills and output
# per GPU falls from ~1,090 to ~280 tok/s (validation, 2026-10-07).
DEFAULT_SESSIONS = (8, 16, 20, 24, 28, 32, 48, 64, 96, 128, 192, 256)
# Per-request rates (higher is faster) and per-turn latencies in seconds (lower is faster).
RATE_METRICS = ("decode_tok_s_p50", "e2e_tok_s_p50")
LATENCY_METRICS = ("latency_s_p50", "latency_s_p90")
SPEED_METRICS = RATE_METRICS + LATENCY_METRICS
DEFAULT_SPEED_METRIC = "latency_s_p90"
# p90 turn latency bounds, seconds: B200 BF16 runs 3.8-5.5 s at 8-16 sessions per engine
# and 38 s once its KV cache is full (validation, 2026-10-07).
DEFAULT_TARGETS = (10.0, 20.0)
# A replica preempted mid-point answers 502/503 until the window ends: its replacement
# has a new address. Such a point is remeasured on the current replicas, this many times
# at most (B200 rows lost 4 replicas each within ten minutes, 2026-10-07).
ENGINE_LOST_ERRORS = ("http_502", "http_503")
MAX_ENGINE_RETRIES = 3
# How late the replay's event loop may wake (p90 over a window) before the point
# measures the client instead of the engines: one streaming driver loading four B200
# engines woke 0.46-0.72 s late and cut output per GPU by ~25% (2026-10-07).
DEFAULT_MAX_LOOP_LAG_S = 0.1

POINT_COLUMNS = (
    "run_id",
    "started_at",
    "config",
    "label",
    "variant",
    "gpu",
    "precision",
    "gpus_per_engine",
    "engines",
    "gpus",
    "price_per_gpu_hour",
    "sessions_per_engine",
    "sessions",
    "warmup_s",
    "window_s",
    "output_tok_s",
    "output_tok_s_per_gpu",
    "prompt_tok_s",
    "uncached_prompt_tok_s",
    "requests_s",
    "requests_s_per_gpu",
    "decode_tok_s_p10",
    "decode_tok_s_p50",
    "decode_tok_s_p90",
    "decode_samples",
    "e2e_tok_s_p50",
    "ttft_s_p50",
    "ttft_s_p90",
    "latency_s_p50",
    "latency_s_p90",
    "prompt_tokens_p50",
    "prompt_tokens_p90",
    "completion_tokens_p50",
    "requests",
    "errors",
    "error_kinds",
    "in_flight_mean",
    "loop_lag_s_p90",
    "loop_lag_s_max",
    "prompt_vs_recorded",
    "engine_retries",
    "server_gen_tok_s",
    "server_prompt_tok_s",
    "server_running_reqs",
    "server_queue_reqs",
    "server_token_usage",
    "server_cache_hit_rate",
    "server_cpu_s_per_request",
    "server_metrics",
    "stop_reason",
)
# Named columns read from the engines' Prometheus metrics, when exported.
SERVER_COLUMNS = {
    "server_gen_tok_s": "rate:sglang:generation_tokens_total",
    "server_prompt_tok_s": "rate:sglang:prompt_tokens_total",
    "server_running_reqs": "mean:sglang:num_running_reqs",
    "server_queue_reqs": "mean:sglang:num_queue_reqs",
    "server_token_usage": "mean:sglang:token_usage",
    "server_cache_hit_rate": "mean:sglang:cache_hit_rate",
}


@dataclass(frozen=True)
class SweepConfig:
    sessions_per_engine: tuple[int, ...] = DEFAULT_SESSIONS
    warmup_s: float = 120.0
    # Warm-up continues past warmup_s until every new session has finished its first
    # (long-prefill) request, up to this bound.
    max_warmup_s: float = 600.0
    window_s: float = 180.0
    targets: tuple[float, ...] = DEFAULT_TARGETS
    speed_metric: str = DEFAULT_SPEED_METRIC
    # A point that adds less than this fraction over the best before it does not count
    # as progress; the engines are saturated after this many such points in a row (with
    # 4-session steps, one point can miss the gain by noise alone).
    saturation_gain: float = 0.03
    saturation_patience: int = 2
    max_error_rate: float = 0.05
    max_loop_lag_s: float = DEFAULT_MAX_LOOP_LAG_S

    def __post_init__(self) -> None:
        if self.speed_metric not in SPEED_METRICS:
            raise ValueError(f"speed_metric must be one of {SPEED_METRICS}")
        if not self.targets or min(self.targets) <= 0:
            raise ValueError("targets must be positive (tok/s or seconds)")
        if self.saturation_patience < 1:
            raise ValueError("saturation_patience must be at least 1")
        if list(self.sessions_per_engine) != sorted(set(self.sessions_per_engine)):
            raise ValueError("sessions_per_engine must be strictly increasing")

    def sessions_up_to(self, max_batch: int) -> tuple[int, ...]:
        """The ramp an engine with a ``max_batch`` ceiling can serve without queueing."""
        return tuple(s for s in self.sessions_per_engine if s <= max_batch)


def pace(metric: str, value: float) -> float:
    """``value`` of ``metric`` oriented so that larger is faster."""
    return -value if metric in LATENCY_METRICS else value


def loosest_target(metric: str, targets: Sequence[float]) -> float:
    """The slowest speed any target allows: the longest latency or the lowest rate."""
    return max(targets) if metric in LATENCY_METRICS else min(targets)


def point_row(
    labels: dict[str, Any],
    summary: dict[str, Any],
    *,
    sessions_per_engine: int,
    engines: int,
    warmup_s: float,
) -> dict[str, Any]:
    """One CSV row: the variant's labels, the window's summary, and per-GPU output."""
    gpus = engines * int(labels["gpus_per_engine"])
    server = summary.get("server") or {}
    row = {column: None for column in POINT_COLUMNS}
    row.update({key: labels.get(key) for key in POINT_COLUMNS if key in labels})
    row.update({key: value for key, value in summary.items() if key in POINT_COLUMNS})
    row.update(
        engines=engines,
        gpus=gpus,
        sessions_per_engine=sessions_per_engine,
        sessions=summary.get("sessions", sessions_per_engine * engines),
        warmup_s=warmup_s,
        output_tok_s_per_gpu=summary["output_tok_s"] / gpus,
        requests_s_per_gpu=(
            summary["requests_s"] / gpus
            if summary.get("requests_s") is not None
            else None
        ),
        error_kinds=json.dumps(summary.get("error_kinds") or {}, sort_keys=True),
        server_cpu_s_per_request=_per_request(
            server.get("rate:sglang:process_cpu_seconds_total"),
            server.get("rate:sglang:e2e_request_latency_seconds_count"),
        ),
        server_metrics=json.dumps(server, sort_keys=True),
        **{column: server.get(name) for column, name in SERVER_COLUMNS.items()},
    )
    return row


# Every reason ``stop_reason`` gives for ending a ramp.
STOP_REASONS = ("client_bound", "below_slowest_target", "errors", "saturated")


def _per_request(rate: float | None, requests: float | None) -> float | None:
    """CPU seconds the engines' main (tokenizer) process spent per finished request.
    Live training engines spend about 0.011 s (B300 NVFP4, 5.4 turns/s, 2026-10-07)."""
    return rate / requests if rate is not None and requests else None


def engine_lost(summary: dict[str, Any], sweep: SweepConfig) -> bool:
    """Whether a window lost an engine: more than the tolerated share of its requests
    failed as an unreachable replica does."""
    kinds = summary.get("error_kinds") or {}
    lost = sum(kinds.get(kind, 0) for kind in ENGINE_LOST_ERRORS)
    attempts = (summary.get("requests") or 0) + (summary.get("errors") or 0)
    return bool(attempts) and lost / attempts > sweep.max_error_rate


def stop_reason(rows: Sequence[dict[str, Any]], sweep: SweepConfig) -> str | None:
    """Why the ramp should end after its last point, or None to continue."""
    last = rows[-1]
    lag = _number(last.get("loop_lag_s_p90"))
    if lag is not None and lag > sweep.max_loop_lag_s:
        return "client_bound"
    speed = _number(last.get(sweep.speed_metric))
    loosest = loosest_target(sweep.speed_metric, sweep.targets)
    if speed is None or pace(sweep.speed_metric, speed) < pace(
        sweep.speed_metric, loosest
    ):
        return "below_slowest_target"
    attempts = (_number(last.get("requests")) or 0) + (_number(last.get("errors")) or 0)
    if (
        attempts
        and (_number(last.get("errors")) or 0) / attempts > sweep.max_error_rate
    ):
        return "errors"
    output = [_number(row["output_tok_s_per_gpu"]) or 0.0 for row in rows]
    stalled = 0
    for i in range(len(output) - 1, 0, -1):
        if output[i] >= max(output[:i]) * (1 + sweep.saturation_gain):
            break
        stalled += 1
    if stalled >= sweep.saturation_patience:
        return "saturated"
    return None


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Add ``row`` by rewriting the whole file and swapping it in. Appending in place to
    a file on a Modal Volume committed after every point left stray fragments of earlier
    rows in the file (sweep2, 2026-10-07)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_csv([path]) if path.exists() else []
    staged = path.with_name(f".{path.name}.tmp")
    with staged.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for kept in [*rows, row]:
            writer.writerow({k: "" if v is None else v for k, v in kept.items()})
    os.replace(staged, path)


def read_csv(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Every point row in ``paths``. A line that is not one (a stray fragment, see
    ``append_csv``) is skipped: a point row names its configuration and ends in a stop
    reason the ramp writes."""
    rows = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                if re.fullmatch(r"[a-z0-9][a-z0-9-]*", row.get("config") or "") and (
                    row.get("stop_reason") or ""
                ) in ("", *STOP_REASONS):
                    rows.append(row)
    return rows


async def run_sweep(
    trajectories: Sequence[replay.Trajectory],
    targets: Sequence[replay.Target],
    *,
    labels: dict[str, Any],
    sweep: SweepConfig,
    replay_config: replay.ReplayConfig | None = None,
    prompts: replay.TokenPrompts | None = None,
    csv_path: Path | None = None,
    on_point: Callable[[dict[str, Any]], None] | None = None,
    refresh_targets: Callable[[], Awaitable[list[replay.Target]]] | None = None,
) -> list[dict[str, Any]]:
    """Ramp one variant's engines (one ``Target`` each) through ``sweep``'s sessions
    per engine; returns the rows, each also appended to ``csv_path``. With
    ``refresh_targets``, a point that lost an engine is remeasured on the replicas it
    returns, every session restarting as at the start of a point."""
    engines = len(targets)
    labels = {
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        **labels,
    }
    rows: list[dict[str, Any]] = []
    async with replay.Replay(
        trajectories, targets, replay_config, prompts=prompts
    ) as running:
        for sessions_per_engine in sweep.sessions_per_engine:
            retries = 0
            while True:
                await running.resize(sessions_per_engine * engines)
                started_at = datetime.now(UTC).isoformat()
                warmed = await running.warmup(sweep.warmup_s, sweep.max_warmup_s)
                summary = await running.measure(sweep.window_s)
                if (
                    refresh_targets is None
                    or retries >= MAX_ENGINE_RETRIES
                    or not engine_lost(summary, sweep)
                ):
                    break
                retries += 1
                await running.resize(0)
                await running.retarget(await refresh_targets())
            summary["engine_retries"] = retries
            row = point_row(
                {**labels, "started_at": started_at},
                summary,
                sessions_per_engine=sessions_per_engine,
                engines=engines,
                warmup_s=warmed,
            )
            rows.append(row)
            row["stop_reason"] = stop_reason(rows, sweep)
            if csv_path is not None:
                append_csv(csv_path, row)
            if on_point is not None:
                on_point(row)
            if row["stop_reason"]:
                break
    return rows


# ── Summary ──────────────────────────────────────────────────────────────────────────


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def throughput_at_speed(
    points: Iterable[dict[str, Any]],
    target: float,
    *,
    speed_key: str = DEFAULT_SPEED_METRIC,
    throughput_key: str = "output_tok_s_per_gpu",
) -> tuple[float | None, str]:
    """The most output per GPU any load sustains while ``speed_key`` meets ``target``
    (a latency at or under it, or a rate at or over it), and how it was found:

    - ``crossed``: the ramp slowed past the target; the value is the best measured
      point at or above it, or the linear interpolation (in speed) between the two
      points that straddle it, whichever is higher.
    - ``lower_bound``: every point ran faster than the target (the ramp ended first,
      usually because the engine saturated); the value is the best point's.
    - ``unreached``: even the lightest load ran slower than the target.

    A ``client_bound`` point measured the replay rather than the engines, and an
    ``errors`` point lost requests (or an engine): both are skipped.
    """
    curve = sorted(
        (
            (_number(p.get("sessions")) or 0.0, pace(speed_key, speed), throughput)
            for p in points
            if p.get("stop_reason") not in ("client_bound", "errors")
            and (speed := _number(p.get(speed_key))) is not None
            and (throughput := _number(p.get(throughput_key))) is not None
        ),
    )
    target = pace(speed_key, target)
    if not curve:
        return None, "unreached"
    candidates = [throughput for _, speed, throughput in curve if speed >= target]
    if not candidates:
        return None, "unreached"
    if len(candidates) == len(curve):
        return max(candidates), "lower_bound"
    for (_, fast, fast_out), (_, slow, slow_out) in zip(curve, curve[1:], strict=False):
        if fast >= target > slow:
            share = (fast - target) / (fast - slow)
            candidates.append(fast_out + share * (slow_out - fast_out))
    return max(candidates), "crossed"


def _with_requests_per_gpu(points: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows written before ``requests_s_per_gpu`` existed get it from their fleet totals."""
    out = []
    for point in points:
        if _number(point.get("requests_s_per_gpu")) is None:
            requests, gpus = (
                _number(point.get("requests_s")),
                _number(point.get("gpus")),
            )
            if requests is not None and gpus:
                point = {**point, "requests_s_per_gpu": requests / gpus}
        out.append(point)
    return out


def _latest_runs(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict]]:
    """Each (config, variant)'s rows from its most recent run."""
    runs: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for row in rows:
        key = (row["config"], row["variant"])
        runs.setdefault(key, {}).setdefault(row["run_id"], []).append(row)
    return {key: by_run[max(by_run)] for key, by_run in runs.items()}


def summarize(
    rows: Iterable[dict[str, Any]],
    *,
    targets: Sequence[float] = DEFAULT_TARGETS,
    speed_key: str = DEFAULT_SPEED_METRIC,
    baseline: str = configs.BASELINE,
) -> list[dict[str, Any]]:
    """One row per configuration measured: at each target, its best variant's output
    per GPU, relative to the baseline's, and the relative cost of a token at list
    prices, (price / throughput) over the baseline's. Each target also reports, for the
    variant chosen by output, the agent turns one GPU serves per hour and the cost of
    1,000 turns (``turns_per_gpu_hour@T``, ``usd_per_kturn@T``), and the same for the
    ``base`` variant (the configuration as the fleet ran it, ``*_base@T``)."""
    best: dict[str, dict[float, tuple[float | None, str, str | None]]] = {}
    turns: dict[tuple[str, str, float], float | None] = {}
    for (config_key, variant), points in sorted(_latest_runs(rows).items()):
        points = _with_requests_per_gpu(points)
        per_target = best.setdefault(config_key, {})
        for target in targets:
            value, kind = throughput_at_speed(points, target, speed_key=speed_key)
            turns[(config_key, variant, target)] = throughput_at_speed(
                points,
                target,
                speed_key=speed_key,
                throughput_key="requests_s_per_gpu",
            )[0]
            current = per_target.get(target)
            if current is None or (value or -1.0) > (current[0] or -1.0):
                per_target[target] = (
                    value,
                    kind,
                    variant if value is not None else None,
                )
    summary = []
    for config_key in [key for key in configs.CONFIGS if key in best] + sorted(
        set(best) - set(configs.CONFIGS)
    ):
        bench = configs.CONFIGS.get(config_key)
        price = bench.price_per_gpu_hour if bench else None
        row: dict[str, Any] = {
            "config": config_key,
            "label": bench.label if bench else config_key,
            "gpu": bench.family if bench else None,
            "precision": bench.precision if bench else None,
            "gpus_per_engine": bench.gpus_per_engine if bench else None,
            "price_per_gpu_hour": price,
        }
        for target in targets:
            value, kind, variant = best[config_key][target]
            base = best.get(baseline, {}).get(target, (None, "", None))[0]
            base_price = configs.CONFIGS[baseline].price_per_gpu_hour
            tag = f"{target:g}"
            row[f"tok_s_per_gpu@{tag}"] = value
            row[f"kind@{tag}"] = kind
            row[f"variant@{tag}"] = variant
            row[f"usd_per_mtok@{tag}"] = (
                price / (value * 3600) * 1e6 if value and price else None
            )
            row[f"rel_throughput@{tag}"] = value / base if value and base else None
            row[f"rel_cost_per_token@{tag}"] = (
                (price / value) / (base_price / base)
                if value and base and price
                else None
            )
            for suffix, name in (("", variant), ("_base", "base")):
                per_s = turns.get((config_key, name, target)) if name else None
                per_hour = per_s * 3600 if per_s else None
                row[f"turns_per_gpu_hour{suffix}@{tag}"] = per_hour
                row[f"usd_per_kturn{suffix}@{tag}"] = (
                    price / per_hour * 1000 if per_hour and price else None
                )
        summary.append(row)
    return summary


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})


# ── CLI ──────────────────────────────────────────────────────────────────────────────


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split(",") if part.strip())


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


def variant_labels(
    bench: configs.BenchConfig, variant: configs.Variant
) -> dict[str, Any]:
    return {
        "config": bench.key,
        "label": bench.label,
        "variant": variant.name,
        "gpu": bench.family,
        "precision": bench.precision,
        "gpus_per_engine": bench.gpus_per_engine,
        "price_per_gpu_hour": bench.price_per_gpu_hour,
    }


def sweep_config_from_args(args: argparse.Namespace) -> SweepConfig:
    if args.speed_metric in RATE_METRICS and not args.stream:
        raise SystemExit(f"--speed-metric {args.speed_metric} needs --stream")
    return SweepConfig(
        sessions_per_engine=_ints(args.sessions),
        warmup_s=args.warmup,
        max_warmup_s=max(args.max_warmup, args.warmup),
        window_s=args.window,
        targets=_floats(args.targets),
        speed_metric=args.speed_metric,
    )


def add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sessions", default=",".join(map(str, DEFAULT_SESSIONS)))
    parser.add_argument("--warmup", type=float, default=120.0)
    parser.add_argument("--max-warmup", type=float, default=600.0)
    parser.add_argument("--window", type=float, default=180.0)
    parser.add_argument(
        "--targets", default=",".join(f"{t:g}" for t in DEFAULT_TARGETS)
    )
    parser.add_argument(
        "--speed-metric", default=DEFAULT_SPEED_METRIC, choices=SPEED_METRICS
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream responses (times decode; far more client work per token)",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="ramp engines you already serve")
    run.add_argument("--url", action="append", required=True, help="one per engine")
    run.add_argument("--traces", required=True)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--config", required=True, choices=list(configs.CONFIGS))
    run.add_argument("--variant", default="base")
    run.add_argument("--model")
    add_sweep_arguments(run)

    summary = commands.add_parser("summarize", help="throughput at speed, per config")
    summary.add_argument("csv", nargs="+")
    summary.add_argument(
        "--targets", default=",".join(f"{t:g}" for t in DEFAULT_TARGETS)
    )
    summary.add_argument(
        "--speed-metric", default=DEFAULT_SPEED_METRIC, choices=SPEED_METRICS
    )
    summary.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        bench = configs.config(args.config)
        variant = bench.variant(args.variant)
        trajectories = replay.load_trajectories(args.traces)
        rows = asyncio.run(
            run_sweep(
                trajectories,
                [replay.Target(url.rstrip("/")) for url in args.url],
                labels=variant_labels(bench, variant),
                sweep=sweep_config_from_args(args),
                replay_config=replay.ReplayConfig(
                    model=args.model, seed=args.seed, stream=args.stream
                ),
                csv_path=args.out,
            )
        )
        for row in rows:
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "sessions",
                            "output_tok_s_per_gpu",
                            args.speed_metric,
                            "stop_reason",
                        )
                    }
                )
            )
        return
    rows = summarize(
        read_csv(args.csv), targets=_floats(args.targets), speed_key=args.speed_metric
    )
    if args.out:
        write_rows(args.out, rows)
    for row in rows:
        print(json.dumps(row))


def describe(sweep: SweepConfig) -> dict[str, Any]:
    return asdict(sweep)


if __name__ == "__main__":
    main()
