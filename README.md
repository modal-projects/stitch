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

### Repeated BF16 comparison

These Llama3.1 8B BF16 samples compare the previous Stitch pin `7686f6b711fc`
with v0.5.21 pin `ffbaf2cf907e`. Each runtime has three identified host groups
and two fresh-engine runs per path on each host. Every allocation used one
H100 80 GB HBM3, CPU request and limit of eight cores, and a 128 GiB memory
limit. Mounted caches
were uncontrolled; the engines used the same fixed 0-to-1 delta.

Values are equal-weight host means in seconds, with `±` the half-width of a
95% Student confidence interval. The interval assumes approximately normal,
independent host means. Repeats do not increase the host count. Wide intervals
can extend below zero; they do not represent negative measured durations.

| Runtime | Path | Prepare RPC | Commit RPC | Scheduler pause upper bound | Prepare-to-commit envelope |
| --- | --- | ---: | ---: | ---: | ---: |
| v0.5.20 | Disk checkpoint | 9.665 ± 7.033 | 2.855 ± 1.429 | 2.874 ± 1.433 | 12.617 ± 8.481 |
| v0.5.20 | CPU / NVMe | 16.686 ± 9.028 | 0.293 ± 0.001 | 0.330 ± 0.041 | 17.105 ± 9.079 |
| v0.5.20 | CPU / RAM | 7.466 ± 3.822 | 0.298 ± 0.018 | 0.318 ± 0.020 | 7.846 ± 3.872 |
| v0.5.21 | Disk checkpoint | 10.736 ± 6.630 | 3.254 ± 1.041 | 3.276 ± 1.040 | 14.100 ± 7.699 |
| v0.5.21 | CPU / NVMe | 15.656 ± 4.891 | 0.392 ± 0.421 | 0.415 ± 0.420 | 16.153 ± 4.520 |
| v0.5.21 | CPU / RAM | 5.582 ± 0.869 | 0.392 ± 0.420 | 0.415 ± 0.418 | 6.069 ± 1.201 |

Each run explicitly paused and resumed the production engine around commit.
The pause column bounds the scheduler flag interval; the envelope includes
client and probe overhead. Encoding, publication, engine startup, destination
initialization, and fleet coordination are excluded. Preparation includes
source reads and mounted-cache misses. Exact native weights, generation
during preparation, and rejection of a missing target passed for every run.

Sampled current-memory intervals are available for each path. The optional
kernel lifetime-peak counter was absent on some hosts, so no whole-cohort
peak interval is reported. These small-model results do not clear the
large-model RAM prepare gap or establish training throughput.

### Repeated TP2 producer comparison

These runs use the same Llama3.1 8B BF16 revision and fixed 0-to-1 delta.
Each runtime has three identified host groups and two fresh-engine repeats
per host. Each allocation used two H100 80 GB HBM3 GPUs, eight CPU cores
(request and limit), and a 128 GiB memory limit. Caches were uncontrolled.
Values are seconds, reported with the same host-level 95% intervals as above.

| Metric | v0.5.20 | v0.5.21 |
| --- | ---: | ---: |
| Prepare RPC | 6.684 ± 2.532 | 3.576 ± 0.646 |
| Commit RPC | 0.222 ± 0.226 | 0.319 ± 0.618 |
| Scheduler pause upper bound | 0.248 ± 0.223 | 0.342 ± 0.616 |
| Synthetic creation + encoding | 21.143 ± 16.230 | 19.053 ± 7.616 |
| Probe input check | 4.379 ± 17.066 | 0.449 ± 0.060 |
| Publication | 4.203 ± 3.395 | 6.086 ± 7.725 |
| Publication-to-generation | 8.448 ± 4.945 | 4.671 ± 0.430 |
| Creation-to-generation envelope | 38.173 ± 37.563 | 30.258 ± 14.851 |
| Resume-to-generation | 0.298 ± 0.040 | 0.287 ± 0.015 |

The maintained synthetic delta generator creates and encodes the input; this
column does not measure a trainer's encoding throughput. Publication uses the
real Publisher and ModalVolumeStore. The creation-to-generation envelope
includes the separately measured frozen-input check. Publication-to-generation
includes post-publication checks and observer overhead. Stage, commit, pause,
resume and first v1 generation use the production engine. Optimizer work,
engine startup, destination initialization, Pool wake, routing and fleet costs
are excluded. Rank timers can include cached startup costs and must not be added.

All native-weight, old-version generation and missing-target checks passed.
Slower successful hosts and runs remain included. The wide intervals do not
establish a general speedup or clear the large-model RAM prepare gap.

### Repeated GLM RAM comparison

These GLM-5.2 mixed NVFP4/BF16 runs compare the previous Stitch pin
`7686f6b711fc` with v0.5.21 pin `ffbaf2cf907e`. Each allocation used one
TP4 serving host with four B300 GPUs and a 64-core CPU request and limit.
The declared memory request and limit were 1 TiB and 3 TiB; observed cgroup
limits are retained with each run. Each host allocation ran two fresh-engine
lineages using the same fixed v1–v4 artifacts. Reused host kernels or GPU
hardware are grouped together. Mounted caches were uncontrolled.

Each cell is an equal-weight host mean in seconds, with the half-width of its
95% Student interval. Host counts and run counts apply to each transition.
The interval assumes approximately normal, independent host means. Repeats do
not increase the independent host count. The folded 1 → 3 update is a separate
population from either one-delta update.

| Runtime | Transition | Prepare RPC | Commit RPC | Scheduler pause upper bound | Prepare-to-commit envelope | Hosts | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.5.20 | 0 → 1 | 83.020 ± 109.536 | 2.844 ± 0.224 | 2.892 ± 0.247 | 86.024 ± 109.808 | 3 | 6 |
| v0.5.20 | 1 → 3 | 114.903 ± 133.505 | 2.826 ± 0.190 | 2.862 ± 0.204 | 117.882 ± 133.722 | 3 | 6 |
| v0.5.20 | 3 → 4 | 83.433 ± 110.257 | 2.814 ± 0.165 | 2.851 ± 0.155 | 86.402 ± 110.346 | 3 | 6 |
| v0.5.21 | 0 → 1 | 41.241 ± 0.914 | 2.740 ± 0.001 | 2.890 ± 0.089 | 44.122 ± 0.853 | 3 | 6 |
| v0.5.21 | 1 → 3 | 52.381 ± 1.225 | 2.740 ± 0.001 | 2.872 ± 0.074 | 55.240 ± 1.232 | 3 | 6 |
| v0.5.21 | 3 → 4 | 40.555 ± 1.328 | 2.740 ± 0.000 | 2.905 ± 0.015 | 43.428 ± 1.293 | 3 | 6 |

Each run explicitly paused and resumed the production engine around commit.
The scheduler-pause column bounds the scheduler flag interval. The envelope
includes client and probe overhead. Preparation includes source reads and
mounted-cache misses. Prior encoding, publication, engine startup,
destination initialization, reference-checkpoint materialization and native
validation are excluded. Rank-response timers can retain startup costs and
must not be added as current update phases.

Every included lineage passed final native v4 weight equality, missing-target
preservation and old-version generation during preparation. These observations
cover one serving host per allocation; they do not establish training or fleet
throughput. Raw runs, host variation and incomplete-run dispositions remain in
the validation evidence.

Baseline preparation varies strongly across hosts. For 0 → 1, baseline host
means range from 37.016 to 124.917 seconds; v0.5.21 host means range from
40.819 to 41.497 seconds. This matched cohort does not reproduce the earlier
large preparation slowdown. These observations do not isolate CPU, NUMA or
cache effects, or establish a general speedup or absence of regression.

### Historical large-model measurements

The following B300 observations use the previous v0.5.20 stack. They are
historical single-run samples and do not validate the v0.5.21 runtime. Each
row has one timing sample; independent host confidence intervals are unavailable.
Correctness checks covered folded catch-up, repeated updates, exact weights,
native-load equality, inference during preparation, and failed preparation.

The diagnostic boundary excludes prior delta encoding and publication, engine
startup, and destination initialization. Preparation includes source reads and
any mounted-cache misses. The commit RPC is distinct from a measured scheduler
pause and from end-to-end synchronization.

| Model | TP / EP | Update path | Preparation RPC | Commit RPC | Prepare-to-commit envelope |
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

The [reference recipe catalog](cookbook/README.md#reference-recipes) covers
agentic NVFP4 training, a small BF16 math starter, and standalone FP8 serving.
[Weight-update profiles](tools/README.md#weight-update-validation) cover a broader
set of architectures independently of the training recipes.

## Integrations

The core package is trainer-, engine-, and provider-agnostic through the
[`Store`](src/stitch/stores/base.py),
[`Engine`](src/stitch/engines/base.py), and
[`Pool`](src/stitch/pools/base.py) interfaces.

Stitch includes Modal Volume and S3 stores, SGLang engines, Modal Flash pools,
and reference Miles and standalone deployments. See the
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
