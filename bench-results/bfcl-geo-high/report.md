# BFCL multi-turn routing benchmark

Trajectories: 18899 complete (2840 straddled a block boundary, dropped). Steps: 109812.

## Per-block

| block | label | trajectories | traj p50 (s) | traj p95 (s) | step p50 (s) | 409s | errors |
|---|---|---|---|---|---|---|---|
| 0 | tune | 3087 | 22.44 | 39.29 | 4.124 | 0 | 8 |
| 1 | gorgo | 3119 | 22.37 | 38.69 | 4.144 | 0 | 12 |
| 2 | session-affinity | 3089 | 22.72 | 39.84 | 4.103 | 0 | 7 |
| 3 | session-affinity | 3083 | 22.88 | 40.11 | 4.115 | 0 | 4 |
| 4 | gorgo | 3681 | 19.19 | 32.98 | 3.382 | 0 | 29 |

## Headline: paired per-episode trajectory wall-time (gorgo vs session-affinity)

- episodes paired: 200
- **mean per-episode delta**: -1.55s (-6.8% of session-affinity median), 95% CI [-1.74, -1.36]s  **(CI excludes 0)**
- episodes faster under gorgo: 186/200
- pooled trajectory p50: gorgo 20.80s vs session-affinity 22.79s; p95: 35.58s vs 39.96s

## Per-step-index latency (does the win grow as prefixes lengthen?)

| step K | gorgo p50 (s) | session-affinity p50 (s) | delta |
|---|---|---|---|
| 0 | 3.675 | 4.072 | -9.8% |
| 1 | 3.615 | 4.092 | -11.7% |
| 2 | 3.689 | 4.132 | -10.7% |
| 3 | 3.685 | 4.127 | -10.7% |
| 4 | 3.651 | 4.107 | -11.1% |
| 5 | 3.617 | 4.088 | -11.5% |
| 6 | 3.633 | 4.101 | -11.4% |
| 7 | 3.746 | 4.254 | -12.0% |
| 8 | 3.724 | 4.179 | -10.9% |
| 9 | 3.664 | 4.114 | -10.9% |

## Caveats

- Teacher-forced replay: prompts are ground-truth prefixes, identical across arms (a feature for pairing; a live agent loop would diverge).
- E2E == TTFT (buffering sidecar); short tool-call decodes keep steps TTFT-dominated.
- Trajectory pairing uses per-episode medians across repeats within each arm.
