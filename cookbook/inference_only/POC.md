# Publish & Pin POC — Deployment & Eval Guide

Exact commands to deploy the POC infrastructure and run evaluations. From the
repo root; select the Modal environment once with `export MODAL_ENVIRONMENT=your_environment`.

## Prerequisites

- `uv` installed with Modal support
- `EXPERIMENT_CONFIG=qwen3_0p6b_poc` for the POC model (Qwen 3-0.6B)
- Modal environment: `$MODAL_ENVIRONMENT`
- A prepped base model (v1) materialized in the store

## Step 0: Prepare Base Model (if not already done)

The publisher needs a v1 checkpoint to fabricate from. This is typically done via a separate prep app:

```bash
EXPERIMENT_CONFIG=qwen3_0p6b_poc \
  uv run --extra modal modal run -d -m cookbook.inference_only.prep_app::download_model
```

This downloads Qwen 3-0.6B and stages it in the checkpoint store. Once complete, you have `updates/weight_v000001/` ready.

## Step 1: Deploy Inference Pool (Router + Servers)

```bash
EXPERIMENT_CONFIG=qwen3_0p6b_poc RUN_ID=pp-poc-run-1 \
  uv run --extra modal python -m cookbook.inference_only.launch
```

This:
- Spins up 3 GPU Server replicas (L40S, 1 GPU each)
- Starts the RouterRegistry and Router (CPU-only)
- Boots sglang + sidecar on each Server with `--version-lease-ttl 120`
- Pool URL: something like `https://pp-poc-run-1--router.modal.run`

**Note**: Get the Router URL from the deploy output (the deploy logs print each
web endpoint's URL), or look it up with
`uv run --extra modal python -c "import modal; print(modal.Cls.from_name('stitch-qwen3-0p6b-poc-pp-poc-run-1', 'Router').get_url())"`.
Save it for the next step.

## Step 2: Deploy Publisher (Single-Writer)

In a separate terminal:

```bash
EXPERIMENT_CONFIG=qwen3_0p6b_poc RUN_ID=pp-poc-run-1 \
  uv run --extra modal python -c "from cookbook.inference_only.publish_app import app; app.deploy()"
```

This:
- Spins up a single Publisher container (CPU, max_containers=1, single concurrent input)
- Mounts the same run volume as the pool
- Serves `/publish`, `/fabricate`, `/status` endpoints
- Publisher URL: something like `https://pp-poc-run-1--publisher.modal.run`

**Note**: Save the Publisher URL for the next step.

## Step 3: Run Evaluation Client

With both pool and publisher deployed, run the eval client locally (no Modal needed):

```bash
python -m cookbook.inference_only.eval_client \
  --router-url https://<ROUTER_URL> \
  --publisher-url https://<PUBLISHER_URL> \
  --registry-url <REGISTRY_URL> \
  --run-id pp-poc-run-1 \
  --evals 3 \
  --eval-minutes 1 \
  --straggler-minutes 6 \
  --sessions 6 \
  --stragglers 2 \
  --think-seconds 0.1 \
  --base-version 1 \
  --results /tmp/pp-poc-results.jsonl
```

**Note**: `--registry-url` is the RouterRegistry URL (same URL pattern as the
Router URL, with the RouterRegistry class slug); it enables fleet_snapshot
recording.

**Note**: `--straggler-minutes` must exceed the sidecar lease TTL
(`--version-lease-ttl 120`, i.e. 2 minutes); otherwise straggler leases expire
before the straggler window closes and the [publish, +straggler-window]
overlap check fails by construction.

The client will:
1. Spin up sessions pinned to version 1 (the base version)
2. At the 1-minute mark, fabricate+publish version 2
3. Spin up new sessions pinned to version 2 (2 old straggler sessions continue on version 1 until the straggler deadline)
4. At the 2-minute mark, fabricate+publish version 3
5. Spin up new sessions pinned to version 3 and continue for 3 evals

Output:
- Stdout: timestamped events (`[t+MM:SS] eval 1 start`, etc.)
- `/tmp/pp-poc-results.jsonl`: Per-request metrics (status, latency, served version, etc.)
- Final summary: per-eval request counts, per-version served counts, assertion checks

## Expected Observations

With 3 replicas, 3 versions, and correct routing:

1. **No version mismatch**: Every request pinned to version N should be served by weight version N.
2. **Concurrent versions**: During overlap windows (when old stragglers persist into new eval), both versions should be serving requests.
3. **409 handling**: Any 409 from a version mismatch should be retried and routed to a matching replica.

## Troubleshooting

**Router or Server won't start**:
- Check Modal logs: `modal app logs --env $MODAL_ENVIRONMENT <app-name>`
- Ensure the run volume exists: `modal volume list --env $MODAL_ENVIRONMENT | grep stitch-pp-poc`

**Publisher 500 errors on /publish**:
- Verify the source model path exists
- Check that the sidecar on Servers can reach the store
- Look at Modal logs for the Publisher container

**Eval client gets 404 on /v1/chat/completions**:
- Confirm the Router URL is correct and publicly accessible
- Check that the pool is in `Running` state: `modal app list --env $MODAL_ENVIRONMENT`

**Straggler sessions end prematurely**:
- The straggler deadline calculation may be off if `--eval-minutes` is too short
- Increase `--straggler-minutes` or adjust `--think-seconds` to slow down the request rate

## Limitations

- **Router re-admission race**: the Router replaces its container table wholesale on each ~1s poll, so a replica evicted after a 409 can be re-admitted on the next poll while the client's bounded retry is still in flight — a mid-retry request may land once more on the same stale replica before the retry budget moves on. The 409-retry cap still guarantees termination.

## Delta mode (GLM)

The `glm5_2_fp8_delta_poc` config runs the same POC with `SGLANG_DELTA_UPDATE_MODE='cpu'`: replicas hold the ~756GB FP8 canonical checkpoint in a CPU weight cache (`--enable-cpu-weight-cache`) and commit XOR deltas in seconds instead of reconstructing the full checkpoint on disk.

**A FRESH run is required.** cpu update mode is delta-only: it rejects FULL publications, including any FULL catch-up a replica would need to join a run that already published full versions. The run must be born at v0 (the base checkpoint at `/checkpoints/glm5-2-fp8`) and only ever see delta publishes. Do not reuse a run ID/volume that has full-version history.

```bash
# Prep base checkpoint once (if not already done)
EXPERIMENT_CONFIG=glm5_2_fp8_delta_poc \
  uv run --extra modal modal run -d -m cookbook.inference_only.prep_app::download_model

# Terminal 1: pool (3 replicas, B300x4, leases TTL 120 as in the disk-mode POC)
EXPERIMENT_CONFIG=glm5_2_fp8_delta_poc RUN_ID=pp-delta-run-1 \
  uv run --extra modal python -m cookbook.inference_only.launch

# Terminal 2: publisher
EXPERIMENT_CONFIG=glm5_2_fp8_delta_poc RUN_ID=pp-delta-run-1 \
  uv run --extra modal python -c "from cookbook.inference_only.publish_app import app; app.deploy()"

# Terminal 3: eval client in delta mode, born at v0
python -m cookbook.inference_only.eval_client \
  --router-url <ROUTER_URL> \
  --publisher-url <PUBLISHER_URL> \
  --registry-url <REGISTRY_URL> \
  --run-id pp-delta-run-1 \
  --delta --base-version 0 \
  --evals 3 --eval-minutes 2 --straggler-minutes 15 \
  --sessions 6 --stragglers 2 --think-seconds 0.1 \
  --results /tmp/pp-delta-results.jsonl
```

With `--delta`, each eval boundary calls `POST /fabricate_delta {base_version: current pin, new_version: next}` — the publisher runs the synthetic-delta generator (`tools/profiling/_synthetic_delta.py`) against the anchor weights and stages a real XOR/zstd delta (a few MB, touching 4 tensors by default) as `updates/weight_vNNNNNN` — then `/publish` for the staged dir. The pointer still only moves via `/publish`; all pinning, lease, snapshot, and assertion behavior is unchanged.

## Cleanup

```bash
# Cancel the pool deployment
modal app stop --env $MODAL_ENVIRONMENT stitch-qwen3-0p6b-poc-pp-poc-run-1

# Cancel the publisher deployment
modal app stop --env $MODAL_ENVIRONMENT stitch-qwen3-0p6b-poc-publisher-pp-poc-run-1

# Optionally delete the volume (if you want to start fresh)
# modal volume delete --env $MODAL_ENVIRONMENT stitch-pp-poc
```

## Warmup delta (cpu update mode)

The first cpu-mode delta triggers a one-time full CPU-weight-image compile
(~5 min at GLM scale) which runs under generation pause on every replica
simultaneously. Publish a small warmup delta immediately after the pool is
ready and BEFORE opening traffic, and wait for fleet convergence. Note that
today the engine still recompiles on each subsequent delta (~5-6 min at GLM
scale, measured); a dirty-group fast path is under investigation. Run evals
with --base-version <warmup version>.
