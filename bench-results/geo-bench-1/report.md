# Router benchmark report — geo-bench-1

Samples: 128849 archived, 1548 excluded as cross-boundary transitions.

## Per-block

| block | label | minutes | n | p50 (s) | p95 (s) | cache-hit | completion p50 | replica share |
|---|---|---|---|---|---|---|---|---|
| 0 | tune | 10.0 | 26466 | 0.194 | 2.151 | 74.5% | 16 | ta-01ky0dvfntbzd9bxdan1n: 64%, ta-01ky0dvg1cykvm8yma82g: 36% |
| 1 | gorgo | 10.0 | 25929 | 0.176 | 2.202 | 83.4% | 16 | ta-01ky0dvfntbzd9bxdan1n: 67%, ta-01ky0dvg1cykvm8yma82g: 33% |
| 2 | session-affinity | 10.0 | 25025 | 0.242 | 2.855 | 87.8% | 16 | ta-01ky0dvfntbzd9bxdan1n: 49%, ta-01ky0dvg1cykvm8yma82g: 51% |
| 3 | session-affinity | 10.0 | 24771 | 0.238 | 2.781 | 90.4% | 16 | ta-01ky0dvfntbzd9bxdan1n: 50%, ta-01ky0dvg1cykvm8yma82g: 50% |
| 4 | gorgo | 10.0 | 25110 | 0.176 | 2.749 | 97.0% | 16 | ta-01ky0dvfntbzd9bxdan1n: 64%, ta-01ky0dvg1cykvm8yma82g: 36% |

## Headline: gorgo vs session-affinity (fixed-weight blocks pooled)

- **p50 E2E**: gorgo 0.176s vs session-affinity 0.240s -> delta -0.064s (-26.6%), 95% CI [-0.068, -0.060]  **(CI excludes 0)**
- **p95 E2E**: gorgo 2.458s vs session-affinity 2.813s -> delta -0.355s (-12.6%), 95% CI [-0.443, -0.278]  **(CI excludes 0)**
- **cache-hit**: gorgo 90.1% vs session-affinity 89.1%
- **workload comparability**: completion p50 16 vs 16; prompt p50 83 vs 82

## Tuner segment

- block 0: n=26466, p95=2.151s, ES steps=19, last_score=-0.855046264
  tuned weights: {'rtt_weight': 0.2908, 'prefill_weight': 0.6217, 'load_weight': 1.0, 'queue_weight': 1.0, 'prefill_rate': 1.0, 'queue_rate': 1.0}

## Final router state

- total_requests=129710, trie sequences=129428, fallbacks={'missing-affinity-key': 487}
- replica RTTs (s): {'ta-01ky0dvfntbzd9bxdan1nd7f8r-80': 0.0518, 'ta-01ky0dvg1cykvm8yma82gc93gr-80': 0.088}

## Caveats

- E2E == TTFT in this stack (the sidecar buffers upstream responses); with small max_tokens the metric isolates network+queue+prefill.
- CIs are request-level bootstraps; requests within a burst share queue state, so effective n is smaller than raw n — treat borderline CIs as inconclusive.
- Block counts per arm: gorgo=2, session-affinity=2 (ABBA cancels linear drift).
