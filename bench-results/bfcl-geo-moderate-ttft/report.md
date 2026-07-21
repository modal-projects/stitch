# BFCL multi-turn routing benchmark

Trajectories: 22898 complete (2814 straddled a block boundary, dropped). Steps: 133097.

## Per-block

| block | label | trajectories | traj p50 (s) | traj p95 (s) | step p50 (s) | 409s | errors |
|---|---|---|---|---|---|---|---|
| 0 | tune | 4305 | 5.95 | 11.66 | 1.008 | 0 | 6 |
| 1 | gorgo | 4329 | 6.02 | 11.48 | 0.978 | 0 | 37 |
| 2 | session-affinity | 3038 | 7.95 | 20.35 | 0.979 | 0 | 15 |
| 3 | session-affinity | 4297 | 6.03 | 12.88 | 0.915 | 0 | 16 |
| 4 | gorgo | 4115 | 6.35 | 13.02 | 0.941 | 0 | 37 |

## Headline: paired per-episode trajectory wall-time (gorgo vs session-affinity)

- episodes paired: 200
- **mean per-episode delta**: -0.58s (-8.6% of session-affinity median), 95% CI [-0.68, -0.47]s  **(CI excludes 0)**
- episodes faster under gorgo: 167/200
- pooled trajectory p50: gorgo 6.18s vs session-affinity 6.67s; p95: 12.22s vs 16.84s

## Per-step-index latency (does the win grow as prefixes lengthen?)

| step K | gorgo p50 (s) | session-affinity p50 (s) | delta |
|---|---|---|---|
| 0 | 0.937 | 0.896 | +4.6% |
| 1 | 0.949 | 0.930 | +2.0% |
| 2 | 0.965 | 0.941 | +2.6% |
| 3 | 0.958 | 0.939 | +2.0% |
| 4 | 0.969 | 0.957 | +1.2% |
| 5 | 0.975 | 0.962 | +1.3% |
| 6 | 0.990 | 1.007 | -1.7% |
| 7 | 0.975 | 0.983 | -0.9% |
| 8 | 1.016 | 0.985 | +3.2% |
| 9 | 1.005 | 1.013 | -0.8% |

## Caveats

- Teacher-forced replay: prompts are ground-truth prefixes, identical across arms (a feature for pairing; a live agent loop would diverge).
- E2E == TTFT (buffering sidecar); short tool-call decodes keep steps TTFT-dominated.
- Trajectory pairing uses per-episode medians across repeats within each arm.
