"""Fetch every run's W&B attempts, stitch them, and write the figures and their data.

Writes to OUT:
  MANIFEST.json            the runs, each attempt's W&B state and the steps it supplied
  data/<run>.jsonl.gz      each run's stitched history, every row tagged with its attempt
  catalog/<prefix>/<metric>.png   every numeric metric across all runs
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
import time
from pathlib import Path

from cookbook.miles_disagg.figures import fetch, history, plots
from cookbook.miles_disagg.figures.runs import STUDY_RUNS, load_runs


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _steps_supplied(rows: list[dict]) -> dict[str, dict[str, list[int]]]:
    """Per attempt and step axis, the first and last step it supplied after stitching."""
    supplied: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        key = history.step_axis(row)
        span = supplied.setdefault(str(row[history.ATTEMPT_KEY]), {}).setdefault(
            key, [row[key], row[key]]
        )
        span[0], span[1] = min(span[0], row[key]), max(span[1], row[key])
    return supplied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, help="JSON run list (default: STUDY_RUNS)")
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache" / "stitch-figures"
    )
    parser.add_argument("--smoothing", type=float, default=0.6)
    parser.add_argument(
        "--only", default="", help="catalog only metrics starting with this prefix"
    )
    parser.add_argument("--no-catalog", action="store_true")
    args = parser.parse_args()

    import wandb

    api = wandb.Api(timeout=120)
    runs = load_runs(args.runs) if args.runs else STUDY_RUNS
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stitch_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "smoothing": args.smoothing,
        "runs": [],
    }
    histories = {}
    for run in runs:
        attempts = [fetch.fetch_run(api, run_id, args.cache) for run_id in run.attempts]
        rows = history.stitch([rows for rows, _ in attempts])
        histories[run.label] = rows
        supplied = _steps_supplied(rows)
        manifest["runs"].append(
            {
                "label": run.label,
                "fleet": run.fleet,
                "arm": run.arm,
                "note": run.note,
                "attempts": [
                    {**meta, "supplied": supplied.get(str(index), {})}
                    for index, (_, meta) in enumerate(attempts)
                ],
            }
        )
        data = args.out / "data" / f"{_slug(run.label)}.jsonl.gz"
        data.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(data, "wt") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
        print(
            f"{run.label}: {len(rows)} rows from {len(attempts)} attempt(s)", flush=True
        )

    if not args.no_catalog:
        names = sorted(
            {m for rows in histories.values() for m in history.metrics(rows)}
        )
        names = [name for name in names if name.startswith(args.only)]
        drawn = 0
        for name in names:
            prefix, _, rest = name.partition("/")
            out = args.out / "catalog" / prefix / _slug(rest or prefix)
            drawn += plots.metric_figure(
                name, runs, histories, smoothing=args.smoothing, out=out
            )
        manifest["catalog_metrics"] = drawn
        print(f"catalog: {drawn} metrics", flush=True)
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
