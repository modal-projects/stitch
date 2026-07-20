# BFCL multi-turn routing benchmark

Trajectories: 8404 complete (5396 straddled a block boundary, dropped). Steps: 48862.

## Per-block

| block | label | trajectories | traj p50 (s) | traj p95 (s) | step p50 (s) | 409s | errors |
|---|---|---|---|---|---|---|---|
| 0 | tune | 0 | nan | nan | nan | 0 | 0 |
| 1 | gorgo | 636 | 10.79 | 19.07 | 2.109 | 0 | 2 |
| 2 | session-affinity | 2372 | 11.31 | 19.69 | 2.109 | 0 | 11 |

## Headline: paired per-episode trajectory wall-time (gorgo vs session-affinity)

- episodes paired: 187
- **mean per-episode delta**: -0.15s (-1.3% of session-affinity median), 95% CI [-0.21, -0.09]s  **(CI excludes 0)**
- episodes faster under gorgo: 123/187
- pooled trajectory p50: gorgo 10.79s vs session-affinity 11.31s; p95: 19.07s vs 19.33s

## Per-step-index latency (does the win grow as prefixes lengthen?)

| step K | gorgo p50 (s) | session-affinity p50 (s) | delta |
|---|---|---|---|
| 0 | 2.114 | 2.098 | +0.7% |
| 1 | 2.097 | 2.107 | -0.5% |
| 2 | 2.110 | 2.110 | +0.0% |
| 3 | 2.114 | 2.114 | +0.0% |
| 4 | 2.094 | 2.101 | -0.3% |
| 5 | 2.086 | 2.086 | -0.0% |
| 6 | 2.120 | 2.125 | -0.2% |
| 7 | 2.160 | 2.188 | -1.3% |
| 8 | 2.135 | 2.144 | -0.4% |
| 9 | 2.035 | 2.169 | -6.2% |

## Caveats

- Teacher-forced replay: prompts are ground-truth prefixes, identical across arms (a feature for pairing; a live agent loop would diverge).
- E2E == TTFT (buffering sidecar); short tool-call decodes keep steps TTFT-dominated.
- Trajectory pairing uses per-episode medians across repeats within each arm.
