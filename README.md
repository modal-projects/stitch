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
- **Elastic rollout capacity.** New replicas load an eligible policy
  checkpoint, catch up to the current version, and enter rotation only when
  ready.
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

These are verified single-update measurements from the pinned v0.5.17 cookbook
stack. Every row completed checksum verification, generated successfully before,
during, and after the update, changed from the base, and reproduced the exact
post-update text, tokens, and logprobs. The synthetic XOR deltas are element-wise
over rollout-visible values and change nearly every trained tensor.
Quantized-value densities are 0.6% for FP8, 0.3% for Kimi K2.6 NVFP4 and Kimi
K3 MXFP4, and 0.375% for mixed GLM-5.2; high-precision values change at 1%.
Remote transfer, delta generation, and one-time CPU destination initialization
are excluded. Preparation runs while inference remains available; only
activation pauses the engine.

| Model | TP | Update path | Preparation | Engine pause | Total update |
| --- | ---: | --- | ---: | ---: | ---: |
| GLM-4.5-Air FP8 | 4 | CPU cache; canonical in RAM | 26.6 s | 0.99 s | 27.5 s |
| GLM-4.5-Air FP8 | 4 | CPU cache; canonical on NVMe | 85.5 s | 0.98 s | 86.5 s |
| GLM-4.5-Air FP8 | 4 | Disk checkpoint | 18.5 s | 27.15 s | 45.7 s |
| Kimi K2.6 NVFP4 | 4 | CPU cache; canonical in RAM | 72.5 s | 2.82 s | 75.3 s |
| Kimi K2.6 NVFP4 | 4 | CPU cache; canonical on NVMe | 279.0 s | 3.23 s | 282.2 s |
| Kimi K2.6 NVFP4 | 4 | Disk checkpoint | 149.0 s | 165.09 s | 314.1 s |
| GLM-5.2 mixed NVFP4/BF16 | 4 | CPU cache; canonical in RAM | 116.4 s | 3.30 s | 119.7 s |
| GLM-5.2 mixed NVFP4/BF16 | 4 | CPU cache; canonical on NVMe | 164.5 s | 3.26 s | 167.7 s |
| GLM-5.2 mixed NVFP4/BF16 | 4 | Disk checkpoint | 128.0 s | 299.38 s | 427.4 s |
| GLM-5.2 FP8 | 4 | CPU cache; canonical in RAM | 108.0 s | 3.44 s | 111.5 s |
| GLM-5.2 FP8 | 4 | CPU cache; canonical on NVMe | 340.2 s | 3.47 s | 343.7 s |
| GLM-5.2 FP8 | 4 | Disk checkpoint | 183.8 s | 239.60 s | 423.4 s |
| Kimi K3 MXFP4 | 8 | CPU cache; canonical in RAM | 120.5 s | 3.84 s | 124.3 s |
| Kimi K3 MXFP4 | 8 | CPU cache; canonical on NVMe | 1,089.7 s | 3.83 s | 1,093.6 s |
| Kimi K3 MXFP4 | 8 | Disk checkpoint | 704.0 s | 300.37 s | 1,004.4 s |

These are wall-clock samples, not a hardware distribution. NVMe preparation
reads and writes a complete canonical checkpoint and therefore tracks the
assigned host's local-storage bandwidth; the Kimi K2.6 and Kimi K3 NVMe samples
are therefore host-specific rather than model-only transformation costs. K3's
canonical checkpoint and eight rank images occupy 3.22 TB before engine and
staging overhead; the all-RAM sample reached 3.29 TB after staging. Its supplied
recipe therefore keeps the canonical checkpoint on NVMe to preserve operating
headroom. The GLM-5.2 FP8, mixed GLM-5.2, and K3 deltas are 11.71 GB, 10.04 GB,
and 24.13 GB compressed, respectively.

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
and reference Miles, Slime, and standalone deployments. See the
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
