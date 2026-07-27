# probes — pressure-test harness (dev-only, never pedagogical)

Disposable instrumentation for certifying the pool + weight-sync protocol under
realistic and adversarial conditions. Unlike `cookbook/` (best-practice recipes),
nothing here demonstrates how stitch *should* be used — probes exist to find where
it breaks and to measure perf under pressure. Expect rough edges; harden a probe
only when it graduates into the standing regression harness.

## Design

A pool certification needs traffic plus replica-state observation:

- **`traffic.py`** — reward-free load shapes against the pool gateway:
  `long_decode`, `long_prefill`, `agentic` (multi-turn sessions with growing
  context, synthetic tool-result injections, session affinity), `mixed`. Responses
  carry `weight_version_start/end`, so the generator doubles as the
  straddle-attribution collector.
- **`poller.py`** — scrapes every replica's `/server_info` into JSONL and
  summarizes: applied-version timelines, per-version convergence lag,
  stage/commit timings, not-ready windows.
- **`app.py`** — Modal wrapper to run the probes from containers.

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
# once
uv run --extra modal modal environment create stitch-dev

# deploy the target pool (example: glm45_air_fp8) into stitch-dev
EXPERIMENT_CONFIG=glm45_air_fp8 uv run --extra modal modal deploy -m cookbook.miles_disagg.app -e stitch-dev

# deploy the probe app
uv run --extra modal modal deploy -m tools.probes.app -e stitch-dev

# start the poller and traffic
uv run --extra modal modal run -e stitch-dev -m tools.probes.app::poll --pool-app stitch-glm45-air-fp8 --tag demo &
uv run --extra modal modal run -e stitch-dev -m tools.probes.app::traffic \
  --pool-app stitch-glm45-air-fp8 --shape agentic --concurrency 32 --duration 1800 --tag demo &
```

## Known limitations (skeleton)

- **No TTFT.** The versioned proxy buffers responses (no streaming), so only
  end-to-end latency is measurable. Revisit when the streaming-proxy decision lands.
- **Synthetic filler text.** Prompt content is pseudo-text at controlled lengths;
  swap in real document corpora when content realism starts to matter.
- **Token counts are approximated** from word counts (~0.75 words/token).
- Version floor tracking polls the gateway's `/server_info`, which answers from an
  arbitrary replica — probe-grade, not exact.
