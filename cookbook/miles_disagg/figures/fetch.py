"""Fetch each W&B run's full history once, and again only while the run is live.

Rows come from ``scan_history``, which returns every logged row unsampled. Each W&B run
is cached as gzipped JSON lines beside a small metadata file; a cached run whose state
was final when it was fetched is never fetched again.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

PROJECT = "nan-playground/fully-async-rl-modal"
FINAL_STATES = frozenset({"finished", "crashed", "failed", "killed"})


def _paths(cache_dir: Path, run_id: str) -> tuple[Path, Path]:
    return cache_dir / f"{run_id}.jsonl.gz", cache_dir / f"{run_id}.meta.json"


def cached(cache_dir: Path, run_id: str) -> tuple[list[dict[str, Any]], dict] | None:
    rows_path, meta_path = _paths(cache_dir, run_id)
    if not rows_path.is_file() or not meta_path.is_file():
        return None
    with gzip.open(rows_path, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    return rows, json.loads(meta_path.read_text())


def fetch_run(
    api: Any, run_id: str, cache_dir: Path, *, project: str = PROJECT
) -> tuple[list[dict[str, Any]], dict]:
    """A W&B run's rows and metadata, from the cache when the run had already ended."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    hit = cached(cache_dir, run_id)
    if hit is not None and hit[1]["state"] in FINAL_STATES:
        return hit
    run = api.run(f"{project}/{run_id}")
    rows = [dict(row) for row in run.scan_history()]
    meta = {
        "run_id": run_id,
        "name": run.name,
        "group": run.group,
        "state": run.state,
        "created_at": str(run.created_at),
        "rows": len(rows),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    rows_path, meta_path = _paths(cache_dir, run_id)
    with gzip.open(rows_path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return rows, meta
