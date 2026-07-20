"""Drive an A/B routing benchmark against a live stitch Router.

Runs locally against the Router's public URL: walks a block schedule (flipping
the routing policy / tuner via the ``/router/*`` admin API at each boundary)
while continuously archiving the router's per-request samples and stats
snapshots. Load comes from elsewhere (a live training run, or the
``tools.probes.app::rl_replay`` generator) — this script only steers and records.

    uv run python -m tools.probes.router_bench \\
      --router-url https://... \\
      --schedule tune:10,gorgo:10,session-affinity:10,session-affinity:10,gorgo:10 \\
      --out results/geo-bench-1

Schedule entries are ``<label>:<minutes>``. ``tune`` means: policy gorgo with the
online-ES tuner enabled (objective neg_p95_e2e); on leaving it the tuned weights
are snapshotted, the tuner disabled, and the weights reset to defaults so later
fixed-weight blocks are clean. Any other label is a policy name. The default
ABBA order cancels linear workload drift in the paired means.

Outputs in --out: ``blocks.json`` (schedule + wall-clock boundaries + tuned-weight
snapshot), ``samples.jsonl`` (deduped router samples, labeled with the block active
at poll time — samples also carry their own dispatch-time ``policy`` field, which
the report trusts over the block label), ``stats.jsonl`` (periodic /router/stats +
/router/tune snapshots).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

POLL_INTERVAL = 5.0


def _retrying(fn, what: str, attempts: int = 12, sleep_s: float = 5.0):
    """Admin calls must survive transient gateway blips (502s during Flash
    rescheduling killed a run's block transition once) — retry hard; a lost
    boundary invalidates the whole schedule."""
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[bench] {what} failed ({exc}); retrying in {sleep_s}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"{what} failed after {attempts} attempts") from last


def _post(client: httpx.Client, url: str, body: dict) -> dict:
    def once() -> dict:
        r = client.post(url, json=body)
        r.raise_for_status()
        return r.json()

    return _retrying(once, f"POST {url.rsplit('/', 1)[-1]}")


def _get(client: httpx.Client, url: str) -> dict:
    r = client.get(url)
    r.raise_for_status()
    return r.json()


def _enter_block(client: httpx.Client, base: str, label: str) -> None:
    if label == "tune":
        _post(client, f"{base}/router/policy", {"policy": "gorgo"})
        _post(client, f"{base}/router/tune",
              {"enabled": True, "mode": "online-es", "objective_metric": "neg_p95_e2e"})
        print(f"[bench] block tune: policy=gorgo, ES tuner ON (neg_p95_e2e)")
    else:
        _post(client, f"{base}/router/policy", {"policy": label})
        print(f"[bench] block: policy={label}")


def _leave_tune(client: httpx.Client, base: str) -> dict:
    """Disable the tuner and reset weights to defaults; return the tuned snapshot."""
    tuned = _retrying(
        lambda: _get(client, f"{base}/router/hyperparameters")["hyperparameters"], "GET hyperparameters"
    )
    tune_state = _retrying(lambda: _get(client, f"{base}/router/tune")["auto_tune"], "GET tune")
    _post(client, f"{base}/router/tune", {"enabled": False})

    def reset() -> None:
        r = client.put(f"{base}/router/hyperparameters", json={})
        r.raise_for_status()

    _retrying(reset, "PUT hyperparameters")
    print(f"[bench] tuner OFF; weights reset. tuned={tuned.get('defaults')} "
          f"applied_count={tune_state.get('applied_count')} last_score={tune_state.get('last_score')}")
    return {"tuned_hyperparameters": tuned, "tune_state": tune_state}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--router-url", required=True)
    p.add_argument("--schedule", default="tune:10,gorgo:10,session-affinity:10,session-affinity:10,gorgo:10")
    p.add_argument("--out", required=True)
    p.add_argument("--poll-interval", type=float, default=POLL_INTERVAL)
    args = p.parse_args()

    base = args.router_url.rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    schedule = []
    for entry in args.schedule.split(","):
        label, _, minutes = entry.strip().partition(":")
        schedule.append((label, float(minutes) * 60.0))

    blocks: list[dict] = []
    seen: set[tuple] = set()
    samples_f = (out / "samples.jsonl").open("a")
    stats_f = (out / "stats.jsonl").open("a")

    with httpx.Client(timeout=30.0) as client:
        health = _get(client, f"{base}/health")
        print(f"[bench] router healthy: {health}")

        def poll(block_idx: int, label: str) -> None:
            now = time.time()
            try:
                recent = _get(client, f"{base}/router/samples?limit=1000")["recent"]
                for s in recent:
                    key = (s.get("recorded_at_monotonic"), s.get("target"))
                    if key in seen:
                        continue
                    seen.add(key)
                    samples_f.write(json.dumps({**s, "block": block_idx, "block_label": label, "polled_at": now}) + "\n")
                stats = _get(client, f"{base}/router/stats")
                tune = _get(client, f"{base}/router/tune")["auto_tune"]
                stats_f.write(json.dumps({"t": now, "block": block_idx, "block_label": label,
                                          "stats": stats, "tune": tune}) + "\n")
                samples_f.flush(); stats_f.flush()
            except Exception as exc:  # noqa: BLE001 — keep polling through blips
                print(f"[bench] poll error (continuing): {exc}")

        for idx, (label, seconds) in enumerate(schedule):
            _enter_block(client, base, label)
            block = {"index": idx, "label": label, "start": time.time(), "planned_seconds": seconds}
            end = block["start"] + seconds
            while time.time() < end:
                poll(idx, label)
                time.sleep(min(args.poll_interval, max(0.0, end - time.time())))
            if label == "tune":
                block["tune_result"] = _leave_tune(client, base)
            block["end"] = time.time()
            blocks.append(block)
            (out / "blocks.json").write_text(json.dumps(blocks, indent=2))

        # Trailing sweep: catch samples for requests still in flight at the end.
        for _ in range(6):
            poll(len(schedule) - 1, schedule[-1][0])
            time.sleep(args.poll_interval)

    samples_f.close(); stats_f.close()
    print(f"[bench] done: {len(seen)} samples across {len(blocks)} blocks -> {out}")


if __name__ == "__main__":
    main()
