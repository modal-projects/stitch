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

These B300 measurements use the pinned v0.5.20 stack. Every row also passed
a folded multi-version catch-up, a repeated update, exact canonical/rank/live
checksums, independent native-load equality, inference during preparation, and
fail-closed checks. Remote transfer, delta generation, and one-time destination
initialization are excluded. Only activation pauses the engine.

| Model | TP / EP | Update path | Preparation | Engine pause | Total update |
| --- | ---: | --- | ---: | ---: | ---: |
| GLM-5.2 mixed NVFP4/BF16 | 4 / 1 | Disk checkpoint | 77.00 s | 171.69 s | 248.87 s |
| GLM-5.2 mixed NVFP4/BF16 | 4 / 1 | CPU rank images; canonical on NVMe | 200.44 s | 2.94 s | 203.68 s |
| GLM-5.2 mixed NVFP4/BF16 | 4 / 1 | CPU rank images; canonical in RAM | 64.83 s | 2.91 s | 67.90 s |
| GLM-5.2 FP8 | 4 / 4 | Disk checkpoint | 116.18 s | 114.32 s | 230.61 s |
| GLM-5.2 FP8 | 4 / 4 | CPU rank images; canonical on NVMe | 191.57 s | 3.43 s | 195.10 s |
| GLM-5.2 FP8 | 4 / 4 | CPU rank images; canonical in RAM | 69.81 s | 3.51 s | 73.47 s |
| GLM-5.3-Flash native FP8 | 8 / 1 | Disk checkpoint | 22.62 s | 14.04 s | 36.83 s |
| GLM-5.3-Flash native FP8 | 8 / 1 | CPU rank images; canonical on NVMe | 134.81 s | 0.79 s | 135.86 s |
| GLM-5.3-Flash native FP8 | 8 / 1 | CPU rank images; canonical in RAM | 49.01 s | 0.78 s | 50.01 s |
| Kimi K2.6 NVFP4 | 4 / 1 | Disk checkpoint | 87.50 s | 176.82 s | 264.41 s |
| Kimi K2.6 NVFP4 | 4 / 1 | CPU rank images; canonical on NVMe | 199.60 s | 2.80 s | 202.59 s |
| Kimi K2.6 NVFP4 | 4 / 1 | CPU rank images; canonical in RAM | 77.82 s | 2.80 s | 80.78 s |
| Kimi K3 MXFP4 | 8 / 1 | Disk checkpoint | 146.71 s | 87.64 s | 234.52 s |
| Kimi K3 MXFP4 | 8 / 1 | CPU rank images; canonical on NVMe | 1,589.07 s | 3.97 s | 1,593.40 s |
| Kimi K3 MXFP4 | 8 / 1 | CPU rank images; canonical in RAM | 241.62 s | 3.82 s | 245.72 s |

These are single-run wall-clock samples, not hardware-independent constants.
Preparation follows host memory bandwidth; NVMe preparation also reads and
writes a complete canonical checkpoint. K3 retains a 1.56 TB canonical
checkpoint and eight 207.47 GB rank images in all-RAM mode, before engine and
bounded staging overhead.

See
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
