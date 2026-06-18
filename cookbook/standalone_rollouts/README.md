# Standalone SGLang Rollout Provider

This cookbook deploys a standalone Modal Flash pool of SGLang rollout servers
that implements the hot-load API in [docs/api-shim.md](../../docs/api-shim.md).
External trainers upload checkpoints or compatible deltas to S3, call the
provider hot-load endpoint, poll readiness, then send rollout traffic to the
same Modal gateway.

The provider flow is:

1. Modal starts one SGLang server per warm container.
2. A FastAPI shim sidecar exposes `/hot_load/v1/models/hot_load`.
3. `POST /hot_load/v1/models/hot_load` records the desired snapshot identity in
   a shared Modal Dict.
4. Every replica notices the desired identity, materializes
   `<mounted-s3-transport>/<identity>/`, applies it to its local SGLang server,
   and reports readiness back into the Modal Dict.
5. `GET /hot_load/v1/models/hot_load` returns the aggregated replica states.
6. Inference requests (`/generate`, `/v1/chat/completions`,
   `/v1/completions`, etc.) are proxied to SGLang.

## Layout

| File | What it is |
|---|---|
| `modal_serve.py` | Standalone Modal rollout-provider app; owns the Modal `App` |
| `provider.py` | Hot-load API shim and transport-to-SGLang apply logic |
| `configs/qwen3_4b_hot_load.py` | Qwen3-4B provider config |
| `slime/` | Optional SLIME integration-test harness for this provider |

## Compatibility Notes

Full snapshots should be Hugging Face/SafeTensors directories that SGLang can
load with `update_weights_from_disk(load_format="auto")`.

Delta snapshots are currently applied through SGLang's `load_format="delta"`
path. That is compatible with SLIME sparse-delta safetensors. The customer shim
spec describes compressed XOR deltas; to support that exact format, add a
materialization step in `provider.py` that reconstructs a full local HF
checkpoint or converts the XOR files into an SGLang-compatible delta before
calling `update_weights_from_disk`.

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
  STITCH_SHIM_PROVIDER_DEPLOYMENT=rollout-prod \
  STITCH_SHIM_BASE_SNAPSHOT_IDENTITY=base
```

Deploy:

```bash
alias m="uv run --extra modal modal"

m run -m cookbook.standalone_rollouts.modal_serve::download_model
m deploy -m cookbook.standalone_rollouts.modal_serve
m run -m cookbook.standalone_rollouts.modal_serve::print_url
m run -m cookbook.standalone_rollouts.modal_serve::smoke \
  --api-key ... \
  --provider-model qwen3-4b \
  --provider-deployment rollout-prod
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
      "compression_format": "deltas_zstd",
      "checksum_format": "adler32"
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

The trainer uses SLIME's opaque HTTP endpoint mode for rollout traffic and
publish-only disk deltas for weight updates. A SLIME request hook adds an exact
`weight_version` constraint to each rollout request after the publish hook has
waited for the provider pool to report the version ready. If Modal's opaque
routing lands a request on a lagging replica, the provider returns retryable
`409` before spending rollout compute.

The harness sets `update_weight_delta_publish_wait="sync"` because its publish
hook polls this provider's pool-readiness endpoint. That makes `update_weights`
block until the configured readiness threshold is met before the next rollout
dispatch starts. For a lower-latency async setup, use SLIME's default publish
wait mode and switch the request hook to the staleness/admission constraint that
matches the trainer's off-policy correction path, for example `min` mode with a
positive request-version lag.
The first weight update seeds SLIME's local delta snapshot and does not announce
a provider update; subsequent updates publish `weight_v000001`,
`weight_v000002`, and so on to the provider.
