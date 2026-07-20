# bfcl-geo-high — BFCL multi-turn, high load (headline: −6.8% trajectory wall-time)

Date: 2026-07-20 (UTC). Environment: Modal `alessio-dev`, app `stitch-qwen3-4b-geo`.

## Setup

- **Fleet**: 2×H200, geo-split — `Server` (us-east) + `ServerB` (us-west), min=max=1 each,
  Qwen/Qwen3-4B on the stitch serving stack (sglang fork `stitch-sglang-v0.5.15-post1`,
  sidecar commit_mode=in_place, `--enable-metrics`). Router (stitch.routing) co-located
  us-east, min=max=1. Observed router→replica RTT ≈ 30–50ms (us-east) vs 90–120ms (us-west).
- **Workload**: teacher-forced BFCL `multi_turn_base` replay (200 episodes, 1160 steps,
  ground-truth prefixes with real tool results; `tools/probes/bfcl_prep.py`), 128 concurrent
  episodes, max_tokens=256, temperature=0, group session id per episode instance.
  Steps within an episode are sequential → trajectory wall-time = Σ step E2E.
- **Schedule** (`tools/probes/router_bench.py`): ABBA — `tune:10, gorgo:10,
  session-affinity:10, session-affinity:10, gorgo:10` (minutes). Tune segment: online-ES on
  neg_p95_e2e; weights reset to defaults (1.0) before the fixed-weight blocks.
- **Code**: stitch `ecf92c8` (+ probe hardening `HEAD`), gorgo package `0d98f7a`
  (github.com/atoniolo76/GORGO), pool claimed at base run `bench-b55bdb50`.

## Headline (paired per-episode, fixed-weight blocks)

- **Mean per-episode trajectory delta: −1.55s (−6.8% of session-affinity median),
  95% bootstrap CI [−1.74, −1.36]s — CI excludes 0.**
- Episodes faster under gorgo: **186/200**.
- Pooled trajectory p50: gorgo **20.80s** vs session-affinity **22.79s**; p95: **35.58s** vs **39.96s**.
- Per-step p50 delta: stable **−10% to −12% at every step index** (0–9).
- 18,899 complete trajectories, 109,812 steps; zero 409s; errors ≤ 29/block.

## Interpretation

At high load (128 episodes vs 2×64 slots) queues dominate; gorgo's win comes from joint
queue + prefix-cache placement, with RTT as tiebreaker. The per-request advantage
compounds across sequential steps into trajectory wall-time — the regime that matters
for agentic RL rollouts. Compare: single-turn saturated barrier test (geo-bench-1) showed
−26.6% per-request p50 but only −2.4% per barrier step; multi-turn sequential structure
is what converts per-request wins into wall-clock.

## Caveats

- Teacher-forced replay: prompts identical across arms (by design, for pairing); a live
  agent loop would diverge.
- E2E ≡ TTFT (sidecar buffers); steps include ~3.4–4.1s decode (256 tokens, Qwen3
  thinking mode), so the routing-sensitive fraction is diluted — deltas are conservative.
- Raw data: `report.md`, `blocks.json` (committed); `samples.jsonl` (129k router samples)
  and `probe_rows.jsonl` (110k step rows) local + on the `stitch-probe-results` volume
  under `/bfcl-geo-high/` (not committed; ~100MB).
