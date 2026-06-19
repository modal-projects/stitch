# Standalone SGLang Rollout Provider

This cookbook deploys a standalone Modal Flash pool of SGLang rollout servers
that implements the customer hot-load API in
[docs/api-shim.md](../../docs/api-shim.md). External trainers upload checkpoints
or deltas to S3, call the provider hot-load endpoint, poll readiness, then send
rollout traffic to the same provider URL.

It is a **log-as-truth** design: a durable, monotonic `latest` pointer in the S3
transport is the source of truth, and the elastic pool reconciles to it by pull.

1. Modal starts one SGLang server + a stitch weight-sync sidecar per warm
   container. Each sidecar is a `WeightSyncManager` that reconciles its engine
   to `latest` (on startup, on a wake, and on a periodic poll).
2. The **front door** — a singleton ASGI app (`max_containers=1`) and the only
   writer of `latest` — serves the customer API. `POST /hot_load/...` advances
   `latest` (monotonic CAS; a rewind is rejected) and best-effort wakes the
   pool. The pool pulls the new `weight_v{N}/`, applies the disk delta host-side
   (slime `disk_delta`: chain-replay + per-tensor checksum), and reloads SGLang.
3. `GET /hot_load/...` reports readiness by enumerating the **live** containers
   and querying each `/server_info` — no self-reported replica state, so a
   scaled-down replica can't haunt the readiness fraction.
4. Inference (`/generate`, `/v1/chat/completions`, `/v1/completions`, …) is
   proxied to the SGLang gateway.

There is no Modal `Dict` desired-mailbox and no per-replica self-report: a
scaled-up container catches up by reading `latest` with no push.

## Layout

| File | What it is |
|---|---|
| `modal_serve.py` | Standalone Modal rollout-provider app; owns the Modal `App`, the Server pool, and the singleton front door |
| `frontdoor.py` | Front-door hot-load adapter logic (advance `latest`, live-readiness, proxy) — injected I/O, unit-tested |
| `provider.py` | Per-container sidecar: a `WeightSyncManager` over a slime-layout board on the transport |
| `configs/qwen3_4b_hot_load.py` | Qwen3-4B provider config |
| `slime/` | Optional SLIME integration-test harness for this provider |

## Compatibility Notes

The provider targets slime's `disk-delta-weight-sync` branch + PR #5. Each
version is a canonical HF/SafeTensors directory `weight_v{N}/` with a
`model.safetensors.index.json`; the engine applies deltas **host-side** (slime
`disk_delta`) onto a local full checkpoint and then reloads through the ordinary
`update_weights_from_disk` path — there is no engine-side `load_format="delta"`
receiver. The delta format is XOR (or `overwrite`) encoding, zstd compression,
and xxh3-128 (or blake3/adler32) per-tensor checksums; the version's
`index.json` carries `delta_encoding`/`compression_format`/`checksum_format`.

A customer-produced delta (XOR + adler32 + zstd) is directly applicable by this
applier. The only adapter work is metadata *location*: the customer sends
`compression_format`/`checksum_format`/`previous_snapshot_identity` in the POST
body and ships a weight-map-only `index.json`, whereas the applier reads those
from the index's `metadata` block — so a customer-facing front door normalizes
the POST metadata into the index before advancing `latest`.

Session affinity is delegated to Modal's Flash gateway. External clients send
the neutral `x-session-affinity` header to the **front door** (the advertised
provider URL, an ASGI app in front of the pool); the front door rewrites it to
`Modal-Session-ID` *before* the gateway, which then consistently routes related
requests to the same replica. The rewrite must happen pre-gateway, so it lives
in the front door rather than the per-container sidecar. Requests without the
header are routed normally.

## Deploy the Provider

Work from the repo root. The provider expects:

- `huggingface-secret` for downloading the base model into the HF cache Volume.
- `stitch-api-shim-provider` for optional hot-load API auth.

The default config mounts the S3 bucket `modal-stitch-s3-transport` at
`/mnt/stitch-s3-transport` with this Modal OIDC role:

```text
arn:aws:iam::459781239556:role/modal-buckets/stitch-s3-transport-role
```

The mounted bucket prefix is `standalone-rollouts/qwen3-4b/`, so the external
S3 location for uploaded snapshots is:

```text
s3://modal-stitch-s3-transport/standalone-rollouts/qwen3-4b/<identity>/
```

Override `STITCH_SHIM_S3_BUCKET_NAME`, `STITCH_SHIM_S3_KEY_PREFIX`,
`STITCH_SHIM_S3_REGION`, or `STITCH_SHIM_S3_OIDC_AUTH_ROLE_ARN` at deploy time
if you create another bucket or prefix. `STITCH_SHIM_S3_REGION` is optional,
but useful when Modal's S3 Mountpoint cannot auto-detect the bucket region.

Create the secret:

```bash
uv run --extra modal modal secret create stitch-api-shim-provider \
  STITCH_SHIM_API_KEY=... \
  STITCH_SHIM_PROVIDER_MODEL=qwen3-4b \
  STITCH_SHIM_PROVIDER_DEPLOYMENT=rollout-prod
```

The S3 OIDC role must allow `s3:PutObject` on the prefix: the singleton front
door writes the `latest` pointer there.

Deploy:

```bash
alias m="uv run --extra modal modal"

m run -m cookbook.standalone_rollouts.modal_serve::download_model
m deploy -m cookbook.standalone_rollouts.modal_serve
m run -m cookbook.standalone_rollouts.modal_serve::print_url
# Authenticated smoke from inside Modal (reads the provider secret; the API key
# never leaves Modal): polls GET /hot_load readiness + a base completion.
m run -m cookbook.standalone_rollouts.modal_serve::check
```

The default app is `stitch-qwen3-4b-api-shim`. To create a separate deployment,
add a config under `configs/` and deploy with:

```bash
PROVIDER_CONFIG=my_provider_config m deploy -m cookbook.standalone_rollouts.modal_serve
```

## External Trainer Contract

Upload each snapshot to:

```text
s3://modal-stitch-s3-transport/standalone-rollouts/qwen3-4b/<identity>/
```

Signal it:

```bash
curl -X POST "$GATEWAY/hot_load/v1/models/hot_load" \
  -H "Authorization: Bearer $STITCH_SHIM_API_KEY" \
  -H "Provider-Model: qwen3-4b" \
  -H "Provider-Deployment: rollout-prod" \
  -H "Content-Type: application/json" \
  -d '{"identity":"weight_v000001"}'
```

For a compatible SGLang/SLIME delta:

```bash
curl -X POST "$GATEWAY/hot_load/v1/models/hot_load" \
  -H "Authorization: Bearer $STITCH_SHIM_API_KEY" \
  -H "Provider-Model: qwen3-4b" \
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
  -H "Provider-Model: qwen3-4b" \
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
m run -m cookbook.standalone_rollouts.modal_serve::download_model
m run -m cookbook.standalone_rollouts.slime.modal_train::prepare_dataset
m deploy -m cookbook.standalone_rollouts.slime.modal_train
m run -m cookbook.standalone_rollouts.slime.modal_train::launch_train
```

The trainer sets `rollout_endpoint_url` (SLIME publish-only mode): it launches
no rollout engines, routes `/generate` to the provider front door, and writes
each `weight_v{N}/` straight to the mounted S3 transport via
`--update-weight-disk-dir`. Its `custom_delta_pre_push_path` hook
(`announce_and_wait`) POSTs the customer hot-load API for the new version and
blocks until the provider pool reports it ready, so the next rollout only runs
once enough replicas serve the new weights. A SLIME request hook adds an exact
`weight_version` constraint to each rollout request; if Modal's opaque routing
lands one on a lagging replica, the provider returns a retryable `409` before
spending rollout compute. For a lower-latency async setup, drop the readiness
wait and switch the request hook to `min` mode with a positive version lag.

The first weight update only seeds SLIME's baseline snapshot and publishes
nothing; subsequent updates publish `weight_v000001`, `weight_v000002`, … .
