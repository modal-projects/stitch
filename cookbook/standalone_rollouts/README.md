# Standalone SGLang Rollout Provider

This cookbook deploys a Modal Flash pool of SGLang rollout servers behind the
customer hot-load API. A trainer uploads a delta under an opaque identity,
signals its parent identity, waits for pool readiness, then sends inference to
the same public URL.

The transport is an append-only log:

1. The configured base checkpoint is stitch v0. SGLang boots from, and the
   sidecar seeds deltas from, the same resolved `BASE_CHECKPOINT` directory.
2. The singleton front door owns `identities.json` and `latest`. It validates a
   POST, derives a provider index without changing the upload, commits the new
   identity to the ledger, then advances `latest`.
3. Each sidecar pulls the ledger and pointer, builds an ephemeral local
   `weight_vN` view, applies the contiguous delta tail, and reloads SGLang.
4. Readiness comes from live `/server_info` responses. A replica is ready only
   when it is idle, error-free, runless, and its integer version maps to the
   requested opaque identity.

Only `/generate`, `/v1/chat/completions`, and `/v1/completions` are proxied.
Control routes are not exposed through the front door.

## Files

| File | Responsibility |
|---|---|
| `modal_serve.py` | Deployment App, utility App, SGLang pool, and singleton front door |
| `opaque_protocol.py` | Strict request, startup recovery, and readiness rules |
| `opaque_frontdoor.py` | Authenticated derive/ledger/pointer transaction |
| `ledger.py` | Opaque identity to integer-version append-only mapping |
| `delta_view.py` | Provider-owned indexes and immutable local version directories |
| `provider.py` | Per-container `WeightSyncManager` sidecar |
| `slime/` | Optional SLIME integration harness |

## Fixed Contract

`STITCH_SHIM_BASE_SNAPSHOT_IDENTITY` is required. It names the exact bytes
loaded from `BASE_CHECKPOINT`; it does not select or upload another checkpoint.
A POST without `incremental_snapshot_metadata` is only an idempotent assertion
of that configured base while v0 remains current. Arbitrary full-snapshot
activation and rewinds are unsupported.

Every delta must extend the current head. Its `identity` and
`previous_snapshot_identity` are opaque strings, but each must be one safe
filesystem path component. Exact retries must repeat the same parent and
formats. Forks, old-identity retries, unknown parents, and contradictory retries
return 409.

The supported decoder contract is:

- delta encoding: XOR, fixed and not a customer field;
- compression: `zstd`;
- checksum: `adler32` (default), `xxh3-128`, or `blake3`.

An upload becomes immutable when its POST is accepted. The customer owns
`<identity>/model.safetensors.index.json` and its referenced shard objects. The
provider owns `identities.json`, `latest`, and `.stitch/`; it never rewrites the
customer index or exposes unreferenced shards to the decoder.

One S3 prefix is one durable chain. Restarting or redeploying the provider on
that prefix is supported. An independent training run requires a fresh prefix
and a recreate deploy; it does not reset or reuse the old chain.

## Deploy

Work from the repository root. The default config uses:

- app: `stitch-moonlight-api-shim`;
- bucket: `modal-stitch-s3-transport`;
- prefix: `standalone-rollouts/moonlight/`;
- mount: `/mnt/stitch-s3-transport`.

The provider needs `huggingface-secret` for an HF base and
`stitch-api-shim-provider` for its mandatory customer auth/base settings:

```bash
ENVIRONMENT=...
uv run --extra modal modal secret create -e "$ENVIRONMENT" stitch-api-shim-provider \
  STITCH_SHIM_API_KEY=... \
  STITCH_SHIM_BASE_SNAPSHOT_IDENTITY=... \
  STITCH_SHIM_PROVIDER_MODEL=moonlight \
  STITCH_SHIM_PROVIDER_DEPLOYMENT=rollout-prod
```

The S3 role needs read/write/delete access to the configured prefix. Override
`STITCH_SHIM_S3_BUCKET_NAME`, `STITCH_SHIM_S3_KEY_PREFIX`,
`STITCH_SHIM_S3_REGION`, or `STITCH_SHIM_S3_OIDC_AUTH_ROLE_ARN` when creating a
separate deployment.

```bash
alias m="uv run --extra modal modal"

m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.modal_serve::download_model
m deploy -e "$ENVIRONMENT" --strategy recreate \
  -m cookbook.standalone_rollouts.modal_serve::app
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.modal_serve::print_url
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.modal_serve::check
```

`--strategy recreate` is a correctness requirement. Modal rolling deploys can
overlap old and new revisions, but the front door must remain the sole ledger
writer. Utility functions live on a separate Modal App so `modal run` cannot
warm a second front door or GPU pool.

To deploy another provider config:

```bash
PROVIDER_CONFIG=my_provider_config m deploy -e "$ENVIRONMENT" --strategy recreate \
  -m cookbook.standalone_rollouts.modal_serve::app
```

## External Trainer

Upload a plain HF-compatible delta directory to:

```text
s3://modal-stitch-s3-transport/standalone-rollouts/moonlight/<opaque-identity>/
```

Then signal the uploaded identity and its current parent:

```bash
curl -X POST "$GATEWAY/hot_load/v1/models/hot_load" \
  -H "Authorization: Bearer $STITCH_SHIM_API_KEY" \
  -H "Provider-Model: moonlight" \
  -H "Provider-Deployment: rollout-prod" \
  -H "Content-Type: application/json" \
  -d '{
    "identity": "checkpoint-step-100",
    "incremental_snapshot_metadata": {
      "previous_snapshot_identity": "<configured-base-identity>",
      "compression_format": "zstd",
      "checksum_format": "xxh3-128"
    },
    "reset_prompt_cache": "new_session"
  }'
```

The next delta names `checkpoint-step-100` as its parent. Poll readiness with:

```bash
curl "$GATEWAY/hot_load/v1/models/hot_load" \
  -H "Authorization: Bearer $STITCH_SHIM_API_KEY" \
  -H "Provider-Model: moonlight" \
  -H "Provider-Deployment: rollout-prod"
```

Count a replica only when `readiness` is true and
`current_snapshot_identity` equals the signalled opaque identity.

## SLIME Harness

The optional harness publishes SLIME disk deltas through the same external
contract. It requires a front-door-seeded, base-only prefix before training;
missing/corrupt control state, advanced state, a base mismatch, or leftover
uploads fail before the GPU training command starts.

Deploying `slime/modal_train.py::app` adds the Trainer to the provider App.
Dataset preparation remains on the utility App; launch and its clean-prefix
check use a dedicated CPU-only preflight App.
`launch_train` first runs a CPU-only clean-prefix check, before allocating the
Trainer GPUs, and the Trainer repeats the check immediately before execution.

```bash
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.modal_serve::download_model
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.slime.modal_train::prepare_dataset
m deploy -e "$ENVIRONMENT" --strategy recreate \
  -m cookbook.standalone_rollouts.slime.modal_train::app
m run -e "$ENVIRONMENT" -m cookbook.standalone_rollouts.slime.modal_train::launch_train
```

The rank-zero publish hook copies each local `weight_vN` directory to the flat
transport prefix, POSTs it with the preceding identity, validates the acceptance
response, and blocks until every live replica reports the new identity. A retry
compares existing upload bytes and never overwrites them.
