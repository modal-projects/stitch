# Router benchmark report — partb-training-3

Samples: 5632 archived, 259 excluded as cross-boundary transitions.

## Per-block

| block | label | minutes | n | p50 (s) | p95 (s) | cache-hit | completion p50 | replica share |
|---|---|---|---|---|---|---|---|---|
| 0 | gorgo | 6.0 | 1076 | 6.911 | 27.407 | 27.6% | 1161 | ta-01ky14k3pbm3p837pb6rr: 51%, ta-01ky14k5khfs1f55jmvez: 49% |
| 1 | session-affinity | 6.0 | 1225 | 5.819 | 29.790 | 28.6% | 1009 | ta-01ky14k3pbm3p837pb6rr: 53%, ta-01ky14k5khfs1f55jmvez: 47% |
| 2 | session-affinity | 6.0 | 1536 | 6.392 | 28.564 | 27.9% | 1121 | ta-01ky14k3pbm3p837pb6rr: 50%, ta-01ky14k5khfs1f55jmvez: 50% |
| 3 | gorgo | 6.0 | 1536 | 6.216 | 28.738 | 27.5% | 1067 | ta-01ky14k3pbm3p837pb6rr: 57%, ta-01ky14k5khfs1f55jmvez: 43% |

## Headline: gorgo vs session-affinity (fixed-weight blocks pooled)

- **p50 E2E**: gorgo 6.549s vs session-affinity 6.153s -> delta +0.397s (+6.4%), 95% CI [+0.125, +0.627]  **(CI excludes 0)**
- **p95 E2E**: gorgo 28.095s vs session-affinity 29.126s -> delta -1.032s (-3.5%), 95% CI [-1.942, +0.034]
- **cache-hit**: gorgo 27.5% vs session-affinity 28.2%
- **workload comparability**: completion p50 1103 vs 1078; prompt p50 84 vs 83

## Final router state

- total_requests=5801, trie sequences=5632, fallbacks={'missing-affinity-key': 3211}
- replica RTTs (s): {'ta-01ky14k5khfs1f55jmvezf5agr-80': 0.2463, 'ta-01ky14k3pbm3p837pb6rrbzs2r-80': 0.3706}

## Caveats

- E2E == TTFT in this stack (the sidecar buffers upstream responses); with small max_tokens the metric isolates network+queue+prefill.
- CIs are request-level bootstraps; requests within a burst share queue state, so effective n is smaller than raw n — treat borderline CIs as inconclusive.
- Block counts per arm: gorgo=2, session-affinity=2 (ABBA cancels linear drift).
