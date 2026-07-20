"""Trajectory-level report for a BFCL replay benchmark run (markdown to stdout).

Joins the replay probe's rows (per-step + per-trajectory, client-measured) with a
router_bench driver's ``blocks.json`` schedule. Trajectories are attributed to the
block that contains their entire span (straddlers dropped); because the workload is
teacher-forced and deterministic, the same episode ids recur under both arms, so the
headline is a **paired per-episode comparison**: for each episode, median trajectory
wall-time under each arm, then the distribution of per-episode deltas with a
bootstrap CI over episodes.

    uv run python -m tools.probes.bfcl_report --rows <bfcl_replay.jsonl> --dir <driver-out>
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

BOOTSTRAP_ITERS = 2000


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))]


def median(xs: list[float]) -> float:
    return pct(xs, 0.50)


def block_policy(label: str) -> str:
    return "gorgo" if label == "tune" else label


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", required=True, help="probe output bfcl_replay.jsonl")
    p.add_argument("--dir", required=True, help="router_bench driver output dir (blocks.json)")
    p.add_argument("--arms", default="gorgo,session-affinity")
    args = p.parse_args()
    arm_a, arm_b = [a.strip() for a in args.arms.split(",")]

    blocks = json.loads((Path(args.dir) / "blocks.json").read_text())
    if "*" in args.rows:  # preempted runs leave multiple uniquely-named row files
        paths = sorted(Path(args.rows).parent.glob(Path(args.rows).name))
    else:
        paths = [Path(args.rows)]
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line]
    steps = [r for r in rows if r["shape"] == "bfcl"]
    trajs = [r for r in rows if r["shape"] == "bfcl_trajectory" and r["complete"]]

    def block_of(start: float, end: float) -> dict | None:
        for b in blocks:
            if start >= b["start"] and end <= b.get("end", float("inf")):
                return b
        return None

    # Attribute trajectories: the row's t is the END; start = end - duration.
    per_block_traj: dict[int, list[dict]] = defaultdict(list)
    dropped = 0
    for tr in trajs:
        b = block_of(tr["t"] - tr["trajectory_seconds"], tr["t"])
        if b is None:
            dropped += 1
            continue
        tr["label"] = b["label"]
        per_block_traj[b["index"]].append(tr)
    steps_by_block: dict[int, list[dict]] = defaultdict(list)
    for s in steps:
        if "latency" not in s:
            continue
        b = block_of(s["t"], s["t"] + s["latency"])
        if b is not None:
            steps_by_block[b["index"]].append(s)

    print(f"# BFCL multi-turn routing benchmark\n")
    print(f"Trajectories: {len(trajs)} complete ({dropped} straddled a block boundary, dropped). "
          f"Steps: {len(steps)}.\n")

    print("## Per-block\n")
    print("| block | label | trajectories | traj p50 (s) | traj p95 (s) | step p50 (s) | 409s | errors |")
    print("|---|---|---|---|---|---|---|---|")
    for b in blocks:
        ts = [t["trajectory_seconds"] for t in per_block_traj.get(b["index"], [])]
        ss = steps_by_block.get(b["index"], [])
        lat = [s["latency"] for s in ss]
        r409 = sum(s.get("retries_409", 0) for s in ss)
        errs = sum("error" in s for s in ss)
        print(f"| {b['index']} | {b['label']} | {len(ts)} | {median(ts):.2f} | {pct(ts, 0.95):.2f} | "
              f"{median(lat):.3f} | {r409} | {errs} |")

    # Paired per-episode comparison over fixed-weight blocks.
    fixed = {b["index"]: b["label"] for b in blocks if b["label"] != "tune"}
    by_arm_episode: dict[str, dict[str, list[float]]] = {arm_a: defaultdict(list), arm_b: defaultdict(list)}
    for idx, label in fixed.items():
        if label not in by_arm_episode:
            continue
        for tr in per_block_traj.get(idx, []):
            by_arm_episode[label][tr["episode"]].append(tr["trajectory_seconds"])

    common = sorted(set(by_arm_episode[arm_a]) & set(by_arm_episode[arm_b]))
    deltas = [median(by_arm_episode[arm_a][e]) - median(by_arm_episode[arm_b][e]) for e in common]
    a_all = [x for e in common for x in by_arm_episode[arm_a][e]]
    b_all = [x for e in common for x in by_arm_episode[arm_b][e]]

    print(f"\n## Headline: paired per-episode trajectory wall-time ({arm_a} vs {arm_b})\n")
    print(f"- episodes paired: {len(common)}")
    if common:
        rng = random.Random(0)
        boot = sorted(
            sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(BOOTSTRAP_ITERS)
        )
        mean_delta = sum(deltas) / len(deltas)
        base = median(b_all)
        print(f"- **mean per-episode delta**: {mean_delta:+.2f}s "
              f"({mean_delta / base:+.1%} of {arm_b} median), "
              f"95% CI [{boot[int(0.025 * len(boot))]:+.2f}, {boot[int(0.975 * len(boot))]:+.2f}]s"
              + ("  **(CI excludes 0)**" if boot[int(0.025 * len(boot))] * boot[int(0.975 * len(boot))] > 0 else ""))
        print(f"- episodes faster under {arm_a}: {sum(d < 0 for d in deltas)}/{len(deltas)}")
        print(f"- pooled trajectory p50: {arm_a} {median(a_all):.2f}s vs {arm_b} {median(b_all):.2f}s; "
              f"p95: {pct(a_all, 0.95):.2f}s vs {pct(b_all, 0.95):.2f}s")

    print(f"\n## Per-step-index latency (does the win grow as prefixes lengthen?)\n")
    print(f"| step K | {arm_a} p50 (s) | {arm_b} p50 (s) | delta |")
    print("|---|---|---|---|")
    by_k: dict[str, dict[int, list[float]]] = {arm_a: defaultdict(list), arm_b: defaultdict(list)}
    for idx, label in fixed.items():
        if label not in by_k:
            continue
        for s in steps_by_block.get(idx, []):
            by_k[label][s["step"]].append(s["latency"])
    for k in sorted(set(by_k[arm_a]) & set(by_k[arm_b]))[:10]:
        a, b = median(by_k[arm_a][k]), median(by_k[arm_b][k])
        print(f"| {k} | {a:.3f} | {b:.3f} | {(a - b) / b:+.1%} |")

    print("\n## Caveats\n")
    print("- Teacher-forced replay: prompts are ground-truth prefixes, identical across arms "
          "(a feature for pairing; a live agent loop would diverge).")
    print("- E2E == TTFT (buffering sidecar); short tool-call decodes keep steps TTFT-dominated.")
    print("- Trajectory pairing uses per-episode medians across repeats within each arm.")


if __name__ == "__main__":
    main()
