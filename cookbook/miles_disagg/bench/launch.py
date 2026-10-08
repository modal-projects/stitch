"""Run the sampler benchmark for one configuration: deploy its engines, ramp every
selected variant in parallel (one CPU driver call each), and stop the app.

    MODAL_FUNCTION_RUNTIME=runc MODAL_ENVIRONMENT=stitch-dev \\
      uv run --extra modal python -m cookbook.miles_disagg.bench.launch \\
        --config b200-bf16 --variants base --engines 1 \\
        --traces bench-traces/qwen36-swebench-pro-base-bf16-36x1

``--traces`` is a directory on the eval volume (``stitch-swebench-pro-eval``) holding
the trajectories an eval launch dumped. Results land on the ``stitch-sampler-bench``
volume under ``<config>/<run_id>/``: one CSV of points and one manifest per variant.
``--plan`` prints the engines, server arguments, ramp, GPU-hours and cost, and exits
without touching Modal.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import subprocess
from datetime import UTC, datetime
from typing import Any

from cookbook.miles_disagg.bench import configs, sweep

_WAIT_SECONDS = 600
# Image pull, checkpoint load and CUDA-graph capture, measured on the training pools.
BOOT_MINUTES = 20.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, choices=list(configs.CONFIGS))
    parser.add_argument("--variants", default="", help="comma-separated; default all")
    parser.add_argument("--engines", type=int, default=1, help="engines per variant")
    parser.add_argument("--traces", help="trace directory on the eval volume")
    parser.add_argument("--experiment", default=configs.DEFAULT_EXPERIMENT)
    parser.add_argument("--run", default="", help="checkpoint run; empty is the base")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--tag", default="", help="app-name suffix")
    sweep.add_sweep_arguments(parser)
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=0,
        help="OpenAI-format top candidates per output token (costs engine CPU per turn)",
    )
    parser.add_argument(
        "--text-prompts",
        action="store_true",
        help="send messages only, so each engine renders and tokenizes every turn itself",
    )
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="let Flash route sessions instead of pinning each to an engine",
    )
    parser.add_argument("--plan", action="store_true", help="print the plan and exit")
    parser.add_argument("--keep-app", action="store_true", help="do not stop the app")
    return parser


def plan(
    bench: configs.BenchConfig,
    variants: Any,
    engines: int,
    ramp: sweep.SweepConfig,
    tag: str = "",
) -> dict[str, Any]:
    """What a launch would run, with an upper bound on its GPU time: every variant
    boots, then runs its whole ramp (early stops only shorten it)."""
    rows = []
    for variant in variants:
        points = ramp.sessions_up_to(variant.max_batch)
        hours = (
            BOOT_MINUTES * 60 + len(points) * (ramp.warmup_s + ramp.window_s)
        ) / 3600
        gpus = engines * bench.gpus_per_engine
        rows.append(
            {
                "variant": variant.name,
                "server": configs.server_name(bench, variant),
                "gpu": f"{bench.gpu}:{bench.gpus_per_engine} x {engines}",
                "sessions_per_engine": list(points),
                "sglang_overrides": variant.sglang_overrides(),
                "max_hours": round(hours, 2),
                "max_gpu_hours": round(hours * gpus, 2),
                "max_usd": round(hours * gpus * bench.price_per_gpu_hour, 2),
            }
        )
    return {
        "app": configs.app_name(bench, tag),
        "config": bench.key,
        "label": bench.label,
        "source": bench.source,
        "base_sglang_args": bench.pool.sglang_args,
        "environment": bench.pool.environment,
        "variants": rows,
        "max_wall_hours": max(row["max_hours"] for row in rows),
        "max_usd": round(sum(row["max_usd"] for row in rows), 2),
    }


def main() -> None:
    args = _parser().parse_args()
    bench = configs.config(args.config)
    variants = configs.select_variants(bench, args.variants)
    ramp = sweep.sweep_config_from_args(args)
    if args.stream and not args.text_prompts:
        raise SystemExit("token prompts read each response whole: drop --stream")
    if args.plan:
        print(json.dumps(plan(bench, variants, args.engines, ramp, args.tag), indent=2))
        return
    if os.environ.get("MODAL_FUNCTION_RUNTIME") != "runc":
        raise SystemExit(
            "set MODAL_FUNCTION_RUNTIME=runc: bench engines must not run on gVisor"
        )
    if not os.environ.get("MODAL_ENVIRONMENT"):
        raise SystemExit("MODAL_ENVIRONMENT must name the environment (stitch-dev)")
    if not args.traces:
        raise SystemExit("--traces is required to launch")
    os.environ.update(
        BENCH_CONFIG=bench.key,
        BENCH_VARIANTS=",".join(variant.name for variant in variants),
        BENCH_ENGINES=str(args.engines),
        BENCH_EXPERIMENT=args.experiment,
        BENCH_RUN=args.run,
        BENCH_VERSION=str(args.version),
        BENCH_TAG=args.tag,
    )
    raise SystemExit(_launch(args, ramp))


def _launch(args: argparse.Namespace, ramp: sweep.SweepConfig) -> int:
    import modal
    from modal.exception import NotFoundError

    from stitch.pools.modal_flash import ModalFlashPool

    bench_app = importlib.import_module("cookbook.miles_disagg.bench.app")
    name = bench_app.APP_NAME
    for pool in bench_app.POOLS.values():
        try:
            ModalFlashPool(name, pool.name).gateway_url()
        except (NotFoundError, RuntimeError):
            continue
        print(f"[{name}] already deployed; stop it first: modal app stop {name}")
        return 1
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    options = {
        "traces": args.traces,
        "run_id": run_id,
        "sweep": dataclasses.asdict(ramp),
        "replay": {
            "seed": args.seed,
            "logprobs": not args.no_logprobs,
            "stream": args.stream,
            "top_logprobs": args.top_logprobs,
        },
        # As Miles' session server sends turns: token prompts, rendered with the chat
        # template options its Qwen3.6 TITO tokenizer uses.
        "token_prompts": not args.text_prompts,
        "chat_template_kwargs": configs.CHAT_TEMPLATE_KWARGS,
        "direct": not args.gateway,
        **_git(),
        "launched_at": datetime.now(UTC).isoformat(),
    }
    print(
        f"[{name}] deploying; stop it any time with: modal app stop {name}", flush=True
    )
    rows: list[dict[str, Any]] = []
    failed = []
    try:
        bench_app.app.deploy()
        driver = modal.Function.from_name(name, "sweep_variant")
        calls = {variant: driver.spawn(variant, options) for variant in bench_app.POOLS}
        for variant, call in calls.items():
            print(f"[{name}] {variant}: driver call {call.object_id}", flush=True)
        for variant, call in calls.items():
            while True:
                try:
                    rows.extend(call.get(timeout=_WAIT_SECONDS))
                    break
                except TimeoutError:
                    print(f"[{name}] {variant} still running", flush=True)
                except Exception as error:  # noqa: BLE001
                    print(f"[{name}] {variant} FAILED: {error!r}", flush=True)
                    failed.append(variant)
                    break
    finally:
        if not args.keep_app:
            subprocess.run(["modal", "app", "stop", "--yes", name], check=False)
    print(
        f"[{name}] points on volume {bench_app.BENCH_VOLUME_NAME} at "
        f"{bench_app.results_path(run_id)}/",
        flush=True,
    )
    for row in _variant_summary(rows, ramp):
        print(json.dumps(row), flush=True)
    return 1 if failed else 0


def _variant_summary(
    rows: list[dict[str, Any]], ramp: sweep.SweepConfig
) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    summary = []
    for variant, points in by_variant.items():
        entry: dict[str, Any] = {"variant": variant}
        for target in ramp.targets:
            value, kind = sweep.throughput_at_speed(
                points, target, speed_key=ramp.speed_metric
            )
            entry[f"tok_s_per_gpu@{target:g}"] = value
            entry[f"kind@{target:g}"] = kind
        summary.append(entry)
    return summary


def _git() -> dict[str, Any]:
    def run(*command: str) -> str:
        return subprocess.run(
            command, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "stitch_commit": run("git", "rev-parse", "HEAD"),
        "stitch_dirty": bool(run("git", "status", "--porcelain")),
    }


if __name__ == "__main__":
    main()
