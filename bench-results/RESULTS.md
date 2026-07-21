# GORGO vs session-affinity — benchmark series (2026-07-20/21, Modal alessio-dev)

All runs: geo-split fleet (Qwen3-4B, 1×H200 us-east + 1×H200 us-west, pinned), stitch Router
co-located us-east, ABBA schedule `tune → gorgo → SA → SA → gorgo` (10-min blocks) driven by
`tools/probes/router_bench.py`, weights at defaults (1.0) in fixed blocks. BFCL runs are
teacher-forced `multi_turn_base` replays (200 episodes, real tool results; episodes pair
exactly across arms). Per-run details in each directory's RUN.md / report.md.

| run | workload | load | trajectory delta (paired) | sync batch completion (B=512) |
|---|---|---|---|---|
| geo-bench-1 | single-turn GSM8K bursts, 16 tok | saturated (512-burst) | −26.6% per request p50 | −2.4% per barrier step |
| bfcl-geo-high | BFCL multi-turn, 256 tok | 128 episodes | **−6.8%** (186/200 faster) | **−11.4%** |
| bfcl-geo-moderate2 | BFCL multi-turn, 256 tok | 48 episodes | −1.6% (150/200) | −1.8% |
| bfcl-geo-moderate-ttft | BFCL multi-turn, 16 tok | 48 episodes | **−8.6%** (167/200), p95 −27% | **−26.1%** |

## Reading

- **Synchronous rollout speed** (batch gated by its slowest trajectory): gorgo's win is a
  *tail* effect. Affinity hash-splits episodes regardless of load; coincident long episodes
  on one replica queue up while the other idles, and the batch waits. Gorgo trades a few
  percent at the median step (occasional cross-region/cold-prefix hop to balance) for a
  collapsed tail: trajectory p95 −11% (high, decode-heavy) to −27% (TTFT-dominated).
- **Decode dilutes, contention amplifies.** With 256-token thinking-mode decodes, ~60–75%
  of each step is routing-insensitive: deltas compress toward −2% (moderate) / −7% (high).
  With decode ≈ 0 (max_tokens=16), the routing-sensitive component shows −8.6% mean and
  −26% batch completion at the same moderate load.
- **Every result is paired and CI-backed**: identical teacher-forced workload across arms,
  per-episode deltas, bootstrap CIs excluding zero in all four runs.
- **Tuner**: the online ES converges within its 10-min segment under real load (69–156
  steps); under near-idle load it walks weights to the range rails — enable it only with
  representative traffic flowing.

## Part B — live training composition (partb-training-3)

Real slime GRPO training drove rollouts through the router while policies flipped live:
**ten delta publishes (v1..v10) applied cleanly on both replicas** (IDLE→PREPARING→IDLE,
zero ERROR, no 409 storms), training unaffected. Routing delta on this workload:
**null after length normalization** (5.35 vs 5.37 ms/completion-token) — single-turn,
decode-dominated, symmetric single-region fleet is the predicted null case. The first
attempt exposed a real weight-sync bug (base seed used the raw hub id as a directory;
first publish wedged the pool) — fixed in stitch `5ff61c0` and validated by the retry.

## Known limitations

- E2E ≡ TTFT everywhere (the stitch sidecar buffers responses); TTFT/decode split is
  inferred via the max_tokens=16 run, not measured in-stream.
- Sync-batch numbers are E[max of B] simulations from steady-state trajectory
  distributions; true batch burst/drain dynamics land in Part B (live slime training).
- Router RTT probes ride through loaded sidecars — part load signal, part network.
- Infra failures observed and hardened against during the series: one driver killed by a
  transient 502 at a block boundary (now retried), one probe container preempted by Modal
  (rows now flushed every 2 min to uniquely-named files). Affected run (bfcl-geo-moderate,
  first attempt) discarded and rerun clean.
