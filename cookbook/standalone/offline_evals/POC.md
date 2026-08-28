# Standalone publish & pin POC — deployment & eval guide

Continuous offline evals on one persistent pool: back-to-back checkpoint
evals, each pinned to its own version while the router rolls the fleet
forward. Exact commands to deploy the standalone pool, publisher, and eval
client, from the repo root. The example config is `qwen3_0p6b` (Qwen3-0.6B,
3×L40S, cpu delta updates) so the whole flow runs cheaply end to end.

## Prerequisites

- `uv` with Modal credentials configured
- A Modal workspace you can deploy apps into
- `EXPERIMENT_CONFIG=qwen3_0p6b` and a `RUN_ID` of your choice (the examples
  use `RUN_ID=eval-run-1`); both env vars scope every app name and store path

## Step 0: Prepare the base model (once per config)

The publisher's delta fabrication anchors at the config's base checkpoint
(`BASE_CHECKPOINT_PATH`, i.e. v0). Prep downloads and stages it:

```bash
EXPERIMENT_CONFIG=qwen3_0p6b \
  uv run --extra modal modal run -d -m cookbook.standalone.prep_app::download_base
```

## Step 1: Deploy the inference pool (router + servers)

```bash
EXPERIMENT_CONFIG=qwen3_0p6b RUN_ID=eval-run-1 \
  uv run --extra modal python -m cookbook.standalone.launch
```

This spins up 3 GPU Server replicas (L40S, 1 GPU each) plus the CPU-only
RouterRegistry and Router, and prints the Router URL at the end. Save it. The
Registry URL follows the same pattern and can be looked up with:

```bash
uv run --extra modal python -c \
  "import modal; print(modal.Cls.from_name('stitch-standalone-qwen3-0p6b-eval-run-1', 'RouterRegistry').get_url())"
```

## Step 2: Deploy the publisher (single writer)

In a separate terminal, with the same env:

```bash
EXPERIMENT_CONFIG=qwen3_0p6b RUN_ID=eval-run-1 \
  uv run --extra modal python -c \
  "from cookbook.standalone.offline_evals.publish_app import app; app.deploy()"
```

One publisher container (max 1, single concurrent input) mounting the same run
volume as the pool. Save its URL for the eval client.

Publisher endpoints:

| Endpoint | Body / response |
| --- | --- |
| `POST /fabricate` | `{run_id, from_version, new_version}` → 202 `{status, job_id, version, path}`; copies the staged `from_version` dir under the new version id |
| `POST /fabricate_delta` | `{run_id, base_version, new_version, num_tensors=4}` → 202 `{status, job_id, version, base_version, path}`; writes a small synthetic XOR delta (embed/lm_head tensors excluded) against the anchor |
| `POST /publish` | `{run_id, version, source}` → 202 `{job_id, version, path}`; 200 `{status: "no-op"}` when the pointer is already ≥ version; 409 while a materialization job is still running; 400 on a foreign run id, a missing source, a FULL publish in cpu delta mode, or a delta not based on the current pointer |
| `GET /job/{job_id}` | `{job_id, status: pending\|success\|failure\|unknown, result?, error?}` |
| `GET /status` | `{latest_version, staged_versions}` — `latest_version` is the store pointer |

Router / registry endpoints the client uses:

| Endpoint | Contract |
| --- | --- |
| `POST /v1/chat/completions` | Proxied to a replica. Pin with `weight_version.exact_version` in the body AND the `stitch-exact-version` header (one value), plus one `Modal-Session-Id` per trajectory. 2xx responses carry `weight_version_start`/`weight_version_end` stamps |
| `POST /sessions/{session_id}/end` | Session tombstone: idempotent, always 200. Ends the trajectory's pin immediately |
| `GET /loads` (registry) | `[{task_id, upstream, load, load_stale, ready, applied_version, draining, live_sessions, active_requests}]` |

## Step 3: Run the eval client

The `qwen3_0p6b` config runs cpu delta mode, which is delta-only: the run is
born at v0 (the prepped base checkpoint) and every publish is a delta, so the
client runs with `--delta --base-version 0`:

```bash
uv run python -m cookbook.standalone.offline_evals.eval_client \
  --router-url <ROUTER_URL> \
  --publisher-url <PUBLISHER_URL> \
  --registry-url <REGISTRY_URL> \
  --run-id eval-run-1 \
  --delta --base-version 0 \
  --evals 3 --eval-minutes 2 --straggler-minutes 15 \
  --sessions 6 --stragglers 2 --think-seconds 0.1 \
  --results /tmp/eval-run-1-results.jsonl
```

`--registry-url` is optional; it enables `/loads` fleet-snapshot recording.

The client runs back-to-back evals with straggler overlap:

1. Sessions pinned to v0 open against the router.
2. At each eval boundary the client calls `POST /fabricate_delta
   {base_version: current pin, new_version: next}` and then `POST /publish`
   for the staged dir. The boundary counts only once `GET /status` shows
   the pointer at the new version.
3. New sessions pin to the new version while the previous eval's straggler
   sessions keep running on the old version until their straggler deadline.

## The two core assertions

Checked from the recorded metrics when the run completes (any failure aborts
with details):

1. **Pinning**: every 200 response carries both version stamps, and both equal
   the request's pinned version. A missing stamp is a failure, not a skip.
2. **Boundary overlap**: in every `[publish accepted, +straggler window]`
   interval, at least one 200 was served at the old version AND at least one
   at the new version — both versions serve concurrently during the rollout.

## The session tombstone contract

Each trajectory uses exactly one `Modal-Session-Id`. When the trajectory ends —
completed or given up — the client makes a best-effort `POST
/sessions/{session_id}/end` (one retry) so the router releases the pin
immediately. The router's TTL is only the backstop for clients that never
call it; sessions are expected to end by tombstone.

## Expected observations

1. **No version mismatch**: every request pinned to version N is served by
   weight version N (stamps start == end == N).
2. **Concurrent versions**: during overlap windows, old-version stragglers and
   new-version sessions serve at the same time.
3. **Transient 409/503 ride-out**: while a rollout is staging, pinned requests
   may 409 (retried with capped exponential backoff) or 503 (retried after the
   server's `Retry-After`); sessions resume once a matching replica is
   available. Resolved streaks are recorded as gaps, not failures.

## Warmup delta (cpu update mode)

The first cpu-mode delta triggers a full CPU-image compile; at large-model
scale a compile takes minutes and runs under generation pause on every replica
simultaneously. Publish a small warmup delta right after the pool is ready and
BEFORE opening traffic, wait for fleet convergence, then run evals with
`--base-version <warmup version>`. The engine recompiles on each subsequent
delta.

## Operational notes

Holds are enforced by routing policy: a mismatched pin reaching a held
replica reconciles it via the stock 409 wake; the registry logs an error
whenever a replica flips without being commanded.

## Troubleshooting

**Pool or publisher won't start**: check `modal app logs <app-name>`; confirm
Step 0 completed and the run volume exists (`modal volume list`).

**Eval client gets 404 on /v1/chat/completions**: confirm the Router URL and
that the pool app is running (`modal app list`).

**Straggler sessions end prematurely**: the straggler window is too short for
the rollout's staging/compile gap — increase `--straggler-minutes` (the
overlap assertion's window is the same duration) or slow the request rate with
`--think-seconds`.

## Teardown

Stop the two apps deployed above when the run is done:

```bash
modal app stop stitch-standalone-qwen3-0p6b-eval-run-1 --yes
modal app stop stitch-standalone-qwen3-0p6b-publisher-eval-run-1 --yes
```
