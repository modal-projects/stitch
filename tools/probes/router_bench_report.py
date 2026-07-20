"""Turn a router_bench output directory into a benchmark report (markdown to stdout).

Attribution rules: a sample belongs to a block by poll-time label, but its own
``policy`` field (stamped at dispatch by the router) must match the block's policy —
samples dispatched under the previous policy that completed after a flip are moved to
the "transition" bucket and excluded from block stats. The headline comparison pools
fixed-weight blocks per arm (the ``tune`` block is reported separately) and pairs
blocks ABBA-style; CIs are request-level bootstraps on the pooled percentile deltas.

    uv run python -m tools.probes.router_bench_report --dir results/geo-bench-1
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


def block_policy(label: str) -> str:
    return "gorgo" if label == "tune" else label


def load(out_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    blocks = json.loads((out_dir / "blocks.json").read_text())
    samples = [json.loads(line) for line in (out_dir / "samples.jsonl").read_text().splitlines() if line]
    stats = [json.loads(line) for line in (out_dir / "stats.jsonl").read_text().splitlines() if line]
    return blocks, samples, stats


def bootstrap_delta_ci(a: list[float], b: list[float], p: float, iters: int = BOOTSTRAP_ITERS,
                       seed: int = 0) -> tuple[float, float, float]:
    """(delta, lo, hi) for pct(a,p) - pct(b,p) with a 95% request-level bootstrap CI."""
    rng = random.Random(seed)
    delta = pct(a, p) - pct(b, p)
    if not a or not b:
        return delta, float("nan"), float("nan")
    deltas = sorted(
        pct(rng.choices(a, k=len(a)), p) - pct(rng.choices(b, k=len(b)), p) for _ in range(iters)
    )
    return delta, deltas[int(0.025 * iters)], deltas[int(0.975 * iters)]


def describe(rows: list[dict]) -> dict:
    lat = [r["total_seconds"] for r in rows]
    prompt = [r["prompt_tokens"] for r in rows]
    completion = [r["completion_tokens"] for r in rows]
    hit = [1.0 - r["uncached_tokens"] / max(1, r["prompt_tokens"]) for r in rows]
    targets = defaultdict(int)
    for r in rows:
        targets[r["target"]] += 1
    return {
        "n": len(rows),
        "e2e_p50_s": pct(lat, 0.50), "e2e_p95_s": pct(lat, 0.95), "e2e_p99_s": pct(lat, 0.99),
        "cache_hit_mean": sum(hit) / len(hit) if hit else float("nan"),
        "prompt_tokens_p50": pct([float(x) for x in prompt], 0.50),
        "completion_tokens_p50": pct([float(x) for x in completion], 0.50),
        "target_share": {t: n / len(rows) for t, n in sorted(targets.items())},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--arms", default="gorgo,session-affinity")
    args = p.parse_args()
    out_dir = Path(args.dir)
    arm_a, arm_b = [a.strip() for a in args.arms.split(",")]
    blocks, samples, stats = load(out_dir)

    per_block: dict[int, list[dict]] = defaultdict(list)
    transition = 0
    for s in samples:
        if s.get("policy") and s["policy"] != block_policy(s["block_label"]):
            transition += 1  # dispatched under the previous block's policy
            continue
        per_block[s["block"]].append(s)

    print(f"# Router benchmark report — {out_dir.name}\n")
    print(f"Samples: {len(samples)} archived, {transition} excluded as cross-boundary transitions.\n")

    print("## Per-block\n")
    print("| block | label | minutes | n | p50 (s) | p95 (s) | cache-hit | completion p50 | replica share |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in blocks:
        d = describe(per_block.get(b["index"], []))
        share = ", ".join(f"{t.rsplit('/', 1)[-1][:24]}: {v:.0%}" for t, v in d["target_share"].items()) or "-"
        mins = (b.get("end", b["start"]) - b["start"]) / 60
        print(f"| {b['index']} | {b['label']} | {mins:.1f} | {d['n']} | {d['e2e_p50_s']:.3f} | "
              f"{d['e2e_p95_s']:.3f} | {d['cache_hit_mean']:.1%} | {d['completion_tokens_p50']:.0f} | {share} |")

    fixed = [b for b in blocks if b["label"] != "tune"]
    a_rows = [r for b in fixed if b["label"] == arm_a for r in per_block.get(b["index"], [])]
    b_rows = [r for b in fixed if b["label"] == arm_b for r in per_block.get(b["index"], [])]
    a_lat = [r["total_seconds"] for r in a_rows]
    b_lat = [r["total_seconds"] for r in b_rows]

    print(f"\n## Headline: {arm_a} vs {arm_b} (fixed-weight blocks pooled)\n")
    for name, q in (("p50", 0.50), ("p95", 0.95)):
        delta, lo, hi = bootstrap_delta_ci(a_lat, b_lat, q)
        base = pct(b_lat, q)
        rel = delta / base if base else float("nan")
        sig = "" if lo <= 0 <= hi else "  **(CI excludes 0)**"
        print(f"- **{name} E2E**: {arm_a} {pct(a_lat, q):.3f}s vs {arm_b} {base:.3f}s -> "
              f"delta {delta:+.3f}s ({rel:+.1%}), 95% CI [{lo:+.3f}, {hi:+.3f}]{sig}")
    da, db = describe(a_rows) if a_rows else None, describe(b_rows) if b_rows else None
    if da and db:
        print(f"- **cache-hit**: {arm_a} {da['cache_hit_mean']:.1%} vs {arm_b} {db['cache_hit_mean']:.1%}")
        print(f"- **workload comparability**: completion p50 {da['completion_tokens_p50']:.0f} vs "
              f"{db['completion_tokens_p50']:.0f}; prompt p50 {da['prompt_tokens_p50']:.0f} vs {db['prompt_tokens_p50']:.0f}")

    tune_blocks = [b for b in blocks if b["label"] == "tune"]
    if tune_blocks:
        print("\n## Tuner segment\n")
        for b in tune_blocks:
            d = describe(per_block.get(b["index"], []))
            tr = b.get("tune_result", {})
            ts = tr.get("tune_state", {})
            print(f"- block {b['index']}: n={d['n']}, p95={d['e2e_p95_s']:.3f}s, "
                  f"ES steps={ts.get('applied_count')}, last_score={ts.get('last_score')}")
            tuned = (tr.get("tuned_hyperparameters") or {}).get("defaults")
            if tuned:
                print(f"  tuned weights: { {k: round(v, 4) for k, v in tuned.items()} }")

    if stats:
        last = stats[-1]["stats"]
        print("\n## Final router state\n")
        print(f"- total_requests={last['total_requests']}, trie sequences={last['radix_trie']['num_sequences']}, "
              f"fallbacks={last['fallback_counts']}")
        rtts = {u: v.get("network_rtt_s") for u, v in last["replicas"].items()}
        print(f"- replica RTTs (s): { {u.rsplit('/', 1)[-1][:32]: (round(r, 4) if r else r) for u, r in rtts.items()} }")

    print("\n## Caveats\n")
    print("- E2E == TTFT in this stack (the sidecar buffers upstream responses); with small "
          "max_tokens the metric isolates network+queue+prefill.")
    print("- CIs are request-level bootstraps; requests within a burst share queue state, so "
          "effective n is smaller than raw n — treat borderline CIs as inconclusive.")
    print(f"- Block counts per arm: {arm_a}={sum(1 for b in fixed if b['label'] == arm_a)}, "
          f"{arm_b}={sum(1 for b in fixed if b['label'] == arm_b)} (ABBA cancels linear drift).")


if __name__ == "__main__":
    main()
