# Stitch

Stitch is the versioned control plane for disaggregated reinforcement learning.
It lets policy training and rollout inference run as independent, elastic
systems while preserving which policy produced every trajectory.

This matters for asynchronous and agentic RL: policy updates continue while
long rollouts are in flight, rollout workers join and leave, and different
consumers tolerate different amounts of staleness. Stitch turns an inference
fleet into a coherent, versioned rollout service. It coordinates policy
publication, replica convergence, request admission, and weight activation
without prescribing the training algorithm, inference engine, storage system,
or compute provider.

```text
Trainer ── publish policy versions ──> Store
   │                                      ▲
   │ version-constrained requests         │ reconcile
   ▼                                      │
Pool gateway ───────────────────────> Rollout replicas ──> Inference engines
```

## What Stitch provides

- **A versioned rollout service.** Requests can require a minimum or exact
  policy version. Incompatible replicas return a retryable `409`, and responses
  report the versions at generation start and end.
- **Continuous policy updates.** Replicas stage and verify the next full
  checkpoint or delta while serving. Weight activation briefly pauses the
  engine and gates new requests.
- **Elastic rollout capacity.** New replicas load the base policy, catch up to
  the current version, and enter rotation only when ready.
- **Failure-safe convergence.** Version bytes become durable before the shared
  pointer advances. A replica reports a version only after its engine activates
  it successfully.
- **Replaceable infrastructure.** Trainers, stores, inference engines, and
  rollout pools meet at small, separate interfaces.

The store is the source of truth. Replicas reconcile independently against its
monotonic version pointer, so a missed notification delays an update but cannot
prevent convergence. This decentralized model lets the rollout fleet scale and
recover without becoming part of the trainer's process lifecycle.

## Measured delta updates

These single-update measurements use the pinned cookbook stack, 64 vCPUs, and
element-wise deltas covering every checkpoint tensor. Remote transfer, delta
generation, and one-time CPU destination initialization are excluded.
Preparation runs while inference remains available; only activation pauses the
engine.

| Model | TP | Canonical checkpoint | Preparation | Engine pause | Total update |
| --- | ---: | --- | ---: | ---: | ---: |
| GLM-4.5-Air FP8 | 4 | RAM | 41.2 s | 1.0 s | 42.2 s |
| Kimi K2.6 NVFP4 | 4 | RAM | 55.6 s | 2.78 s | 58.4 s |
| Kimi K2.6 NVFP4 | 4 | NVMe | 102.3 s | 2.76 s | 105.1 s |
| GLM-5.2 NVFP4 | 4 | RAM | 59.3 s | 2.14 s | 61.5 s |
| Kimi K3 MXFP4 | 8 | RAM | 122.3 s | 3.82 s | 126.1 s |
| Kimi K3 MXFP4 | 8 | NVMe | 283.8 s | 3.79 s | 287.5 s |

Each profiler reconstructs and checksums the complete target and validates
generation before, during, and after activation. See
[`Profile an update`](cookbook/README.md#profile-an-update) to reproduce these
measurements and
[`SGLANG_FORK.md`](cookbook/common/SGLANG_FORK.md#cpu-destination) for memory
sizing and destination tradeoffs.

## Integrations

The core package is trainer-, engine-, and provider-agnostic through the
[`Store`](src/stitch/stores/base.py),
[`Engine`](src/stitch/engines/base.py), and
[`Pool`](src/stitch/pools/base.py) interfaces.

Stitch includes Modal Volume and S3 stores, SGLang engines, Modal Flash pools,
and reference Miles and Slime deployments. See the
[`cookbook`](cookbook/README.md) to choose an update mode, launch a run, scale
the rollout fleet, and validate an update. Fork pins and re-porting notes are
in [`SGLANG_FORK.md`](cookbook/common/SGLANG_FORK.md) and
[`MILES_FORK.md`](cookbook/miles_disagg/MILES_FORK.md).

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
