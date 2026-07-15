# probes — pressure-test harness (dev-only, never pedagogical)

Disposable instrumentation for certifying the pool + weight-sync protocol under
realistic and adversarial conditions. Unlike `cookbook/` (best-practice recipes),
nothing here demonstrates how stitch *should* be used — probes exist to find where
it breaks and to measure perf under pressure. Expect rough edges; harden a probe
only when it graduates into the standing regression harness.

## Design

A pool certification doesn't need a trainer. The serving side needs **publishes**
and **traffic**, so the harness decouples them:

- **`replay_publisher.py`** — replays a *recorded* delta chain (past runs persist on
  the delta Volume) through the real `publish_version()` path at a controlled
  cadence, under a fresh `run_id`. Real delta densities, sizes, and chain structure;
  zero trainer GPUs. The target pool must serve the same base model.
- **`traffic.py`** — reward-free load shapes against the pool gateway:
  `long_decode`, `long_prefill`, `agentic` (multi-turn sessions with growing
  context, synthetic tool-result injections, session affinity), `mixed`. Responses
  carry `weight_version_start/end`, so the generator doubles as the
  straddle-attribution collector.
- **`poller.py`** — scrapes every replica's `/server_info` into JSONL and
  summarizes: applied-version timelines, per-version convergence lag,
  stage/commit timings, not-ready windows.
- **`app.py`** — Modal wrapper to run the above from containers (the replay
  publisher needs the delta Volume mounted).

## Conventions

- **Environment:** everything probe-related lives in the `stitch-dev` Modal
  environment (`modal environment create stitch-dev`, once). The target pool must
  be deployed in the *same* environment — `ModalFlashPool` resolves names in the
  caller's environment — so deploy the cookbook recipe with `-e stitch-dev` for
  probe runs.
- **Results:** JSONL baselines land on the `stitch-probe-results` Volume, one
  directory per tagged run. Baselines are *recorded and human-judged*, not CI
  gates — do not wire thresholds into CI.

## Quickstart

```bash
# step 0 — local shakeout: the whole harness against the real serving stack
# (real SGLangEngine/Reconciler/create_app over a mock engine; no Modal, no GPUs)
uv run --extra sglang --extra modal --with uvicorn python -m tools.probes.local_shakeout

# once
uv run --extra modal modal environment create stitch-dev

# deploy the target pool (example: glm45_air_fp8) into stitch-dev
EXPERIMENT_CONFIG=glm45_air_fp8 uv run --extra modal modal deploy -m cookbook.miles_disagg.app -e stitch-dev

# deploy the probe app, bound to that recipe's delta volume
PROBE_DELTA_VOLUME=stitch-delta-glm45-air-fp8 \
  uv run --extra modal modal deploy -m tools.probes.app -e stitch-dev

# start the poller (background), then traffic, then the replay
uv run --extra modal modal run -e stitch-dev -m tools.probes.app::poll --pool-app stitch-glm45-air-fp8 --tag demo &
uv run --extra modal modal run -e stitch-dev -m tools.probes.app::traffic \
  --pool-app stitch-glm45-air-fp8 --shape agentic --concurrency 32 --duration 1800 --tag demo &
uv run --extra modal modal run -e stitch-dev -m tools.probes.app::replay \
  --pool-app stitch-glm45-air-fp8 --source-run <run_id> --cadence-s 60 --tag demo
```

## Known limitations (skeleton)

- **No TTFT.** The versioned proxy buffers responses (no streaming), so only
  end-to-end latency is measurable. Revisit when the streaming-proxy decision lands.
- **Synthetic filler text.** Prompt content is pseudo-text at controlled lengths;
  swap in real document corpora when content realism starts to matter.
- **Token counts are approximated** from word counts (~0.75 words/token).
- The replay publisher **copies** each version dir under the new run prefix
  (that's the real publish path) — delta volumes grow per replay and nothing GCs.
- Version floor tracking polls the gateway's `/server_info`, which answers from an
  arbitrary replica — probe-grade, not exact.
