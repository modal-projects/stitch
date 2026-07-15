"""Scrape every replica's ``/server_info`` into JSONL, and summarize.

The certification runs' observability spine: ``/server_info`` is point-in-time, so this
turns it into a timeseries — applied-version timelines per replica, per-version
convergence lag (first replica at vN -> last replica at vN), stage/commit timings, and
not-ready windows.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

REDISCOVER_EVERY = 30.0  # seconds between replica-list refreshes (containers come and go)


async def poll(
    app_name: str,
    cls_name: str = "Server",
    *,
    interval: float = 2.0,
    duration: float | None = None,
    out_path: str,
    replicas: list[str] | None = None,  # static URL list: skip Modal discovery
) -> None:
    import httpx

    fixed = list(replicas) if replicas else None
    pool = None
    if fixed is None:
        from stitch.pools.modal_flash import ModalFlashPool

        pool = ModalFlashPool(app_name, cls_name)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    replicas = fixed or []
    last_discover = 0.0
    deadline = time.time() + duration if duration else None
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        with out.open("a") as f:
            while deadline is None or time.time() < deadline:
                if pool is not None and (not replicas or time.time() - last_discover >= REDISCOVER_EVERY):
                    replicas = await asyncio.to_thread(pool.discover_replicas)
                    last_discover = time.time()
                for row in await asyncio.gather(*(_probe(client, url) for url in replicas)):
                    f.write(json.dumps(row) + "\n")
                f.flush()
                await asyncio.sleep(interval)


async def _probe(client: Any, url: str) -> dict[str, Any]:
    row: dict[str, Any] = {"t": time.time(), "replica": url}
    try:
        row["info"] = (await client.get(f"{url.rstrip('/')}/server_info")).json()
    except Exception as exc:  # noqa: BLE001 — an unreachable replica is itself a data point
        row["error"] = str(exc)[:200]
    return row


def summarize(path: str) -> dict[str, Any]:
    """Reduce a poll JSONL to the certification quantities."""
    from stitch.versions import VersionRef

    first_seen: dict[str, dict[int, float]] = defaultdict(dict)  # replica -> version -> t
    timings: dict[tuple[str, float], dict[str, Any]] = {}  # (replica, metrics.at) -> metrics
    unready: dict[str, float] = defaultdict(float)  # replica -> seconds observed not ready
    prev_t: dict[str, float] = {}
    errors = 0

    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        replica, t = row["replica"], row["t"]
        info = row.get("info")
        if info is None:
            errors += 1
            continue
        if info.get("applied"):
            v = VersionRef.parse(info["applied"]).version
            first_seen[replica].setdefault(v, t)
        if not info.get("ready") and replica in prev_t:
            unready[replica] += t - prev_t[replica]
        if (m := info.get("metrics")) and "at" in m:
            timings[(replica, m["at"])] = m
        prev_t[replica] = t

    versions = sorted({v for per in first_seen.values() for v in per})
    convergence = {
        v: round(max(per[v] for per in first_seen.values() if v in per)
                 - min(per[v] for per in first_seen.values() if v in per), 3)
        for v in versions
        if sum(v in per for per in first_seen.values()) > 1
    }
    stage = sorted(m["stage_s"] for m in timings.values() if "stage_s" in m)
    commit = sorted(m["commit_s"] for m in timings.values() if "commit_s" in m)
    return {
        "replicas": len(first_seen),
        "versions_seen": versions,
        "convergence_lag_s": convergence,  # first-replica-at-vN -> last-replica-at-vN
        "stage_s": _dist(stage),
        "commit_s": _dist(commit),
        "unready_s": {k: round(v, 1) for k, v in unready.items() if v},
        "probe_errors": errors,
    }


def _dist(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    return {"n": len(xs), "p50": xs[len(xs) // 2], "p95": xs[int(len(xs) * 0.95)], "max": xs[-1]}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poll")
    p.add_argument("--app", required=True)
    p.add_argument("--cls", default="Server")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("path")
    args = ap.parse_args()
    if args.cmd == "poll":
        asyncio.run(poll(args.app, args.cls, interval=args.interval, duration=args.duration, out_path=args.out))
    else:
        print(json.dumps(summarize(args.path), indent=2))
