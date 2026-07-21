# partb-training-3 — Part B: live slime training through the router

Date: 2026-07-21 (UTC). App `stitch-qwen3-4b-gorgo` in `alessio-dev`;
stitch `5ff61c0`, gorgo `0d98f7a`.

## What this run proves (primary goal: composition)

Real slime GRPO training (Qwen3-4B, GSM8K, 8×H200 trainer + 2×H200 rollout,
single-region, `commit_mode=in_place`) drove its rollouts through the stitch Router
while the bench driver alternated the routing policy live (`gorgo:6, SA:6, SA:6,
gorgo:6`).

- **Ten delta publishes (weight_v1..v10) staged and committed cleanly on both replicas
  under live rollout traffic and mid-flight policy flips** — sync states cycled
  IDLE → PREPARING → IDLE with zero ERROR states and no 409 storms.
- Training progressed normally across all policy flips (reward pipeline unaffected —
  routing is transparent to the version-gate semantics, as designed).

## Bug found and fixed by this benchmark (first attempt, discarded)

The first Part B attempt wedged on the first publish: the sglang fork's weight-sync
base seed uses `server_args.model_path` as a *directory* (`weight_updater.
_reseed_base_dir`), and the cookbook launched sglang with the raw hub id
`Qwen/Qwen3-4B` → `FileNotFoundError` in `_reset_checkpoint` → base prefetch never
succeeded → every publish cycled ERROR→re-prefetch→ERROR while bounded-staleness
constraints (correctly) stalled rollouts. Fix: `cookbook/common/server.py` resolves
the model to its local HF-cache snapshot dir before launch (stitch `5ff61c0`).
Validated: base prefetch completed in ~4 min on fresh replicas, then v1..v10 applied.

## Routing result on this workload: null (expected)

Raw p50 favored affinity (+6.4% against gorgo) but gorgo's blocks drew ~2.3% longer
generations; length-normalized the arms are identical — 5.35 vs 5.37 ms/completion-
token; within the 512–1024-completion-token bucket p50 4.44s vs 4.35s, p95 26.29s vs
26.28s. Single-turn GSM8K GRPO on a symmetric single-region 2-replica fleet with
~1100-token decodes is the predicted null case for routing (see RESULTS.md): >90% of
E2E is decode, there are no multi-turn chains, no RTT asymmetry, and 55% of slime's
/generate requests carried no session-affinity key at all (3,211/5,801 fell back to
random under the SA arm — worth wiring `rollout_session_affinity_header` stamping for
GRPO groups if affinity is ever load-bearing here).

## Data

`report.md`, `blocks.json`, `samples.jsonl` (5,632 router samples) in this directory;
version-transition log in the session monitor stream. Trainer ran ~10 rollout steps
before the app was stopped (benchmark done; run not trained to completion).
