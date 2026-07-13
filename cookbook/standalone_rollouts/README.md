# Standalone SGLang Rollout Provider

This cookbook deploys a standalone Modal Flash pool of SGLang rollout servers
that implements the customer hot-load API. External trainers upload checkpoints
or deltas to S3, call the provider hot-load endpoint, poll readiness, then send
rollout traffic to the same provider URL.

It is a **log-as-truth** design: a durable, monotonic `latest` pointer in the S3
transport is the source of truth, and the elastic pool reconciles to it by pull.

1. Modal starts one SGLang server + a stitch weight-sync sidecar per warm
   container. Each sidecar is a `WeightSyncManager` that reconciles its engine
   to `latest` (on startup, on a wake, and on a periodic poll).
2. The **front door** — a singleton `App.server` (`min_containers=1`, pinned to
   the same routing region as the pool) and the only writer of `latest` — serves
   the customer API. `POST /hot_load/...` advances
   `latest` (monotonic CAS; a rewind is rejected) and best-effort wakes the
   pool. The pool pulls the new `weight_v{N}/`, applies the disk delta host-side
   (slime `disk_delta`: chain-replay + per-tensor checksum), and reloads SGLang.
3. `GET /hot_load/...` reports readiness by enumerating the **live** containers
   and querying each `/server_info` — no self-reported replica state, so a
   scaled-down replica can't haunt the readiness fraction.
4. Inference (`/generate`, `/v1/chat/completions`, `/v1/completions`, …) is
   proxied to the SGLang gateway.

## Layout

| File | What it is |
|---|---|
| `modal_serve.py` | Standalone Modal rollout-provider app; owns the Modal `App`, the Server pool, and the singleton front door |
| `frontdoor.py` | Front-door hot-load adapter logic (advance `latest`, live-readiness, proxy) — injected I/O, unit-tested |
| `provider.py` | Per-container sidecar: a `WeightSyncManager` over a slime-layout board on the transport |
| `configs/moonlight_hot_load.py` | Moonlight-16B-A3B provider config (DeepSeek-V3-arch MoE; routing replay) |
| `slime/` | Optional SLIME integration-test harness for this provider |

## Compatibility Notes

The provider targets the pinned modal-projects/slime fork commit (see
`modal_serve.py`). Each version is a canonical HF/SafeTensors directory
`weight_v{N}/` with a `model.safetensors.index.json`. Deltas are applied
**host-side** (slime `disk_delta`: XOR or overwrite encoding, zstd compression,
per-tensor checksums — all declared in the index's `metadata` block) onto a
local full checkpoint, then reloaded through the ordinary
`update_weights_from_disk` path; there is no engine-side delta receiver. The
front door normalizes hot-load POST metadata into the index before advancing
`latest`, so a customer-produced delta is directly applicable.

Session affinity is delegated to Modal's Flash gateway: external clients send
the neutral `x-session-affinity` header to the front door, which rewrites it to
`Modal-Session-ID` *before* the gateway so related requests route to the same
replica. The rewrite must happen pre-gateway, so it lives in the front door
rather than the per-container sidecar.

## Deploy the Provider

Work from the repo root. The provider expects:

- `huggingface-secret` for downloading the base model into the HF cache Volume.
- `stitch-api-shim-provider` for optional hot-load API auth.

The default config mounts the S3 bucket `modal-stitch-s3-transport` at
`/mnt/stitch-s3-transport` with this Modal OIDC role:

```text
arn:aws:iam::459781239556:role/modal-buckets/stitch-s3-transport-role
```

The mounted bucket prefix is `standalone-rollouts/moonlight/`, so the external
S3 location for uploaded snapshots is:

```text
s3://modal-stitch-s3-transport/standalone-rollouts/moonlight/<identity>/
```

Override `STITCH_SHIM_S3_BUCKET_NAME`, `STITCH_SHIM_S3_KEY_PREFIX`,
`STITCH_SHIM_S3_REGION`, or `STITCH_SHIM_S3_OIDC_AUTH_ROLE_ARN` at deploy time
if you create another bucket or prefix. `STITCH_SHIM_S3_REGION` is optional,
but useful when Modal's S3 Mountpoint cannot auto-detect the bucket region.

Create the secret:

```bash
ENVIRONMENT=...
uv run --extra modal modal secret create -e "$ENVIRONMENT" stitch-api-shim-provider \
  STITCH_SHIM_API_KEY=... \
  STITCH_SHIM_BASE_SNAPSHOT_IDENTITY=... \
  STITCH_SHIM_PROVIDER_MODEL=moonlight \
  STITCH_SHIM_PROVIDER_DEPLOYMENT=rollout-prod
```

The S3 OIDC role must allow `s3:PutObject` on the prefix: the singleton front
door writes the `latest` pointer there.

Deploy:

```bash
alias m="uv run --extra modal modal"

m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.modal_serve::download_model
m deploy -e "$ENVIRONMENT" --strategy recreate -m cookbook.standalone_rollouts.modal_serve::app
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.modal_serve::print_url
# Authenticated smoke from inside Modal (reads the provider secret; the API key
# never leaves Modal): polls GET /hot_load readiness + a base completion.
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.modal_serve::check
```

`--strategy recreate` is required: the front door is the sole ledger writer,
and a rolling deploy can overlap old and new revisions.

The default app is `stitch-moonlight-api-shim`. To create a separate deployment,
add a config under `configs/` and deploy with:

```bash
PROVIDER_CONFIG=my_provider_config m deploy -e "$ENVIRONMENT" --strategy recreate \
  -m cookbook.standalone_rollouts.modal_serve::app
```

## External Trainer Contract

Upload each snapshot to:

```text
s3://modal-stitch-s3-transport/standalone-rollouts/moonlight/<identity>/
```

Signal it:

```bash
curl -X POST "$GATEWAY/hot_load/v1/models/hot_load" \
  -H "Authorization: Bearer $STITCH_SHIM_API_KEY" \
  -H "Provider-Model: moonlight" \
  -H "Provider-Deployment: rollout-prod" \
  -H "Content-Type: application/json" \
  -d '{"identity":"weight_v000001"}'
```

For a compatible SGLang/SLIME delta:

```bash
curl -X POST "$GATEWAY/hot_load/v1/models/hot_load" \
  -H "Authorization: Bearer $STITCH_SHIM_API_KEY" \
  -H "Provider-Model: moonlight" \
  -H "Provider-Deployment: rollout-prod" \
  -H "Content-Type: application/json" \
  -d '{
    "identity": "weight_v000002",
    "incremental_snapshot_metadata": {
      "previous_snapshot_identity": "weight_v000001",
      "compression_format": "zstd",
      "checksum_format": "xxh3-128"
    },
    "reset_prompt_cache": "new_session"
  }'
```

Poll readiness:

```bash
curl "$GATEWAY/hot_load/v1/models/hot_load" \
  -H "Authorization: Bearer $STITCH_SHIM_API_KEY" \
  -H "Provider-Model: moonlight" \
  -H "Provider-Deployment: rollout-prod"
```

A replica counts as ready only when `readiness` is true and
`current_snapshot_identity` matches the requested identity.

## Optional SLIME Test Harness

The provider app above is standalone and can be used by any external trainer
that follows the same S3 + hot-load contract. For end-to-end testing, this
cookbook also includes `slime/`, a Modal-hosted SLIME trainer harness that
copies sparse deltas into the mounted S3 transport and calls the deployed
provider.

The provider owns the Modal app. Deploying `modal_serve.py` publishes only the
SGLang rollout provider. Deploying `slime/modal_train.py` imports that same
app and adds the trainer functions, so the resulting Modal app contains both
the standalone provider and the SLIME integration-test trainer.

The trainer reuses `stitch-api-shim-provider` for optional shim API auth. It
derives the provider URL from the deployed Flash `Server` class, and the S3
transport uses the same Modal `CloudBucketMount` OIDC role as the provider.

Then redeploy the same Modal app with the trainer functions included and launch
the trainer:

```bash
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.modal_serve::download_model
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.slime.modal_train::prepare_dataset
m deploy -e "$ENVIRONMENT" --strategy recreate -m cookbook.standalone_rollouts.slime.modal_train::app
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.slime.modal_train::launch_train
```

The trainer (`slime/configs/moonlight_slime_trainer.py`) runs **async-first**
(`train_async`, one-step off-policy) with **rollout routing replay** enabled, in
SLIME publish-only mode: it launches no rollout engines, routes `/generate` to
the provider front door, and writes each `weight_v{N}/` to a local disk dir.

Staleness is gated by a **publish-hook readiness barrier**, not a per-request
pin (the key difference from `slime_disagg`, which uses a min-version request
gate): the `announce_and_wait` publish hook copies each new version to the S3
transport, POSTs the hot-load API, and blocks until the front door reports every
live replica ready — so the next rollouts always run on current weights.
