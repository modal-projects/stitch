# Rollout probes

This directory contains a development harness for load-testing a rollout pool
and observing weight-version convergence. It is not a deployment example.

- `traffic.py` sends long-decode, long-prefill, agentic, or mixed traffic and
  records version attribution.
- `poller.py` records each replica's version, convergence lag, staging and
  engine pause timing, and not-ready windows.
- `app.py` runs both probes on Modal and stores JSONL results.

## Run

The probes and target pool must use the same Modal environment. Results are
written under the run tag on the `stitch-probe-results` Volume.

```bash
uv run --extra modal modal environment create stitch-dev

EXPERIMENT_CONFIG=glm45_air_fp8 \
  uv run --extra modal modal deploy -m cookbook.miles_disagg.app -e stitch-dev

uv run --extra modal modal deploy -m tools.probes.app -e stitch-dev

uv run --extra modal modal run -e stitch-dev \
  -m tools.probes.app::poll \
  --pool-app stitch-glm45-air-fp8 --tag demo

uv run --extra modal modal run -e stitch-dev \
  -m tools.probes.app::traffic \
  --pool-app stitch-glm45-air-fp8 \
  --shape agentic --concurrency 32 --duration 1800 --tag demo
```

## Limits

- The versioned proxy does not stream, so the harness measures end-to-end
  latency but not time to first token.
- Traffic uses synthetic text and approximate token counts.
- Version-floor polling samples an arbitrary replica through `/server_info`;
  use per-replica logs for exact attribution.
- Recorded baselines are reviewed by humans and are not CI thresholds.

## Sidecar config check (CPU-only)

Validates the sidecar entrypoint (``python -m stitch.sidecar``) in a bare
stitch-only image — flag coverage (`--help`), disk-mode validation, and
store-factory resolution without sglang/Modal deps. Real launches pass the
consuming package's factory (this repo's recipes use ``--store-factory
cookbook.common.storage:create_store``; see ``cookbook.common.process``).

```bash
uv run --extra modal modal run -e stitch-dev -m tools.probes.sidecar_config
```

Prints one machine-readable ``PROBE_RESULT ok=... detail=...`` line.

## GLM-5.2 image preflight (CPU-only)

This builds the exact pinned trainer and serving images, then validates model
arguments and the DFlash contract without reserving GPUs:

```bash
modal run -e stitch-dev tools/probes/glm5_2_nvfp4_dflash_preflight.py
```

The command finishes with one machine-readable ``VERDICT ... PASS`` line.
