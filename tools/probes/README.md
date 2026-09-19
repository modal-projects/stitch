# Rollout probes

This directory contains a development harness for load-testing a rollout pool
and observing its weight versions.

- `traffic.py` sends long-decode, long-prefill, agentic, or mixed traffic and
  records version attribution.
- `poller.py` samples discovered replicas' versions, staging and
  engine pause timing, and not-ready windows.
- `app.py` runs both probes on Modal and stores JSONL results.

## Run

Prepare and launch the maintained `glm5_3_fp8` standalone recipe using the
[cookbook](../../cookbook/README.md), then use its run ID below. The probes and
target pool must use the same Modal environment. Results are written under the
run tag on the `stitch-probe-results` Volume.

```bash
export MODAL_ENVIRONMENT=stitch-dev
export RUN_ID=your-existing-run-id
export STITCH_POOL_APP="stitch-standalone-glm5-3-fp8-${RUN_ID}"

uv run --extra modal modal profile current
uv run --extra modal modal deploy -e "$MODAL_ENVIRONMENT" -m tools.probes.app
```

Run the polling and traffic commands in separate terminals with the variables
above set in each:

```bash
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m tools.probes.app::poll \
  --pool-app "$STITCH_POOL_APP" --tag "$RUN_ID"

uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m tools.probes.app::traffic \
  --pool-app "$STITCH_POOL_APP" --model zai-org/GLM-5.3 \
  --shape agentic --concurrency 32 --duration 1800 --tag "$RUN_ID"
```

## Limits

- The versioned proxy does not stream, so the harness measures end-to-end
  latency but not time to first token.
- Traffic uses synthetic text and approximate token counts.
- Version-floor polling samples an arbitrary replica through `/server_info`;
  use per-replica logs for exact attribution.
- Poller summaries are diagnostic and must not be used for convergence
  acceptance. They do not establish a complete participant set; the reported
  convergence lag includes only replicas observed at that version. Empty or
  incomplete observations can therefore omit lagging replicas.

## Sidecar config check (CPU-only)

Validates the sidecar entrypoint (``python -m stitch.sidecar``) in a bare
stitch-only image — flag coverage (`--help`), disk-mode validation, and
store-factory resolution without sglang/Modal deps. Real launches pass the
consuming package's factory (this repo's recipes use ``--store-factory
cookbook.common.storage:create_store``; see ``cookbook.common.process``).

```bash
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" -m tools.probes.sidecar_config
```

Prints one machine-readable ``PROBE_RESULT ok=... detail=...`` line.
