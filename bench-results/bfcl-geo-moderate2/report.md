# BFCL multi-turn routing benchmark

Trajectories: 13843 complete (1668 straddled a block boundary, dropped). Steps: 80367.

## Per-block

| block | label | trajectories | traj p50 (s) | traj p95 (s) | step p50 (s) | 409s | errors |
|---|---|---|---|---|---|---|---|
| 0 | tune | 2460 | 10.79 | 19.06 | 2.071 | 0 | 0 |
| 1 | gorgo | 2455 | 10.92 | 18.67 | 2.079 | 0 | 0 |
| 2 | session-affinity | 2417 | 11.15 | 19.43 | 2.064 | 0 | 4 |
| 3 | session-affinity | 2415 | 11.08 | 19.39 | 2.066 | 0 | 17 |
| 4 | gorgo | 2428 | 10.93 | 19.13 | 2.084 | 0 | 13 |

## Headline: paired per-episode trajectory wall-time (gorgo vs session-affinity)

- episodes paired: 200
- **mean per-episode delta**: -0.17s (-1.6% of session-affinity median), 95% CI [-0.21, -0.14]s  **(CI excludes 0)**
- episodes faster under gorgo: 150/200
- pooled trajectory p50: gorgo 10.93s vs session-affinity 11.12s; p95: 18.92s vs 19.41s

## Per-step-index latency (does the win grow as prefixes lengthen?)

| step K | gorgo p50 (s) | session-affinity p50 (s) | delta |
|---|---|---|---|
| 0 | 2.069 | 2.060 | +0.5% |
| 1 | 2.075 | 2.063 | +0.6% |
| 2 | 2.086 | 2.073 | +0.6% |
| 3 | 2.088 | 2.071 | +0.8% |
| 4 | 2.080 | 2.052 | +1.3% |
| 5 | 2.077 | 2.046 | +1.5% |
| 6 | 2.082 | 2.059 | +1.1% |
| 7 | 2.107 | 2.117 | -0.5% |
| 8 | 2.112 | 2.090 | +1.1% |
| 9 | 2.089 | 2.128 | -1.9% |

## Caveats

- Teacher-forced replay: prompts are ground-truth prefixes, identical across arms (a feature for pairing; a live agent loop would diverge).
- E2E == TTFT (buffering sidecar); short tool-call decodes keep steps TTFT-dominated.
- Trajectory pairing uses per-episode medians across repeats within each arm.
