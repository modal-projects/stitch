# RL Rollout Spec

Each training iteration:

1. **Train** — update model weights (Megatron)
2. **Sync weights** — upload delta checkpoint to S3, signal provider to hot-load it, poll until ready
3. **Rollout** — call provider's inference API to generate trajectories

```
┌──────────────┐       S3 bucket        ┌──────────────────┐
│   Trainer    │ ────── upload ────────> │    Provider       │
│  (Megatron)  │                        │  (Inference GPUs) │
│              │ <── hot-load API ────> │                   │
│              │                        │                   │
│              │ ── /v1/completions ──> │                   │
│              │ <── responses ──────── │                   │
└──────────────┘                        └──────────────────┘
```

---

## 1. Object Storage

Provider must read checkpoints from an S3-compatible bucket.

- S3 API support (AWS S3, GCS with S3 interop, MinIO, etc.)
- Read access to a shared bucket (credentials exchanged out-of-band)
- Path convention: `s3://<bucket>/<prefix>/<checkpoint_identity>/`

The standalone Modal cookbook mounts the shared S3 prefix with
`modal.CloudBucketMount` and OIDC, then reads checkpoints from the mounted
filesystem path. That is an implementation detail of the provider; external
trainers still publish to the `s3://.../<checkpoint_identity>/` location.

Each checkpoint directory contains:

- `model.safetensors.index.json` — weight map (tensor name → filename)
- `model-NNNNN.safetensors` — weight files (<5GB each, roughly grouped by layer)
- `config.json`, `tokenizer.json`, `tokenizer_config.json` — model config

Format: HuggingFace SafeTensors, unsharded.

---

## 2. Hot-Load API

### 2a. Signal Checkpoint Ready

Tell the provider a checkpoint is uploaded and should be loaded.

**Full checkpoint:**

```
POST /hot_load/v1/models/hot_load

Headers:
  Authorization: Bearer <api_key>
  Provider-Model: <model_identifier>
  Provider-Deployment: <deployment_identifier>

Body:
{
  "identity": "<checkpoint_identity>"
}
```

**Delta checkpoint:**

```
POST /hot_load/v1/models/hot_load

Headers: (same)

Body:
{
  "identity": "<checkpoint_identity>",
  "incremental_snapshot_metadata": {
    "previous_snapshot_identity": "<previous_checkpoint_identity>",
    "compression_format": "<compression_format>",
    "checksum_format": "<checksum_format>"
  },
  "reset_prompt_cache": "new_session"
}
```

### 2b. Poll Readiness

Trainer polls until enough replicas have loaded the new weights.

```
GET /hot_load/v1/models/hot_load
<!---->
Headers: (same)

Response:
{
  "replicas": [
    {
      "readiness": true,
      "current_snapshot_identity": "<checkpoint_identity>"
    },
    {
      "readiness": false,
      "current_snapshot_identity": "<old_identity>",
      "readiness_reason": "downloading weights"
    }
  ]
}
```

Loaded when:

```
count(readiness == true AND current_snapshot_identity == target) / total >= threshold
```

Default threshold 1.0 (all replicas). Configurable (e.g., 0.8 to tolerate errored replicas).

---

## 3. Delta Checkpoints

After the initial full checkpoint, subsequent updates are sent as delta checkpoints to minimize upload size and load time.

**Format:**

- XOR of byte representations of old and new weight tensors, then compressed
- Safetensors files contain compressed diff tensors (not actual weights)
- Checksum per tensor for integrity (currently adler32)
- Compression format is specified in `incremental_snapshot_metadata.compression_format` so trainer and provider agree on the algorithm

**Provider must:**

- Accept `incremental_snapshot_metadata` in the hot-load signal (includes `compression_format` and `checksum_format`)
- Store previous checkpoint weights to apply the XOR diff
- Pipeline: decompress (using specified format) → XOR with previous → new weights
- Be prepared to support additional compression formats as they are added

**Delta lifecycle:**

1. First checkpoint is always a full snapshot (a pre-uploaded base)
2. Subsequent checkpoints are deltas against the immediately preceding one
3. On training resume, the first delta is against the original base (may be larger than typical deltas since the model has diverged after many training steps)

---

## 4. Session Affinity

Route related requests to the same replica for KV cache reuse.

The provider receives a routing key on each inference request and uses consistent hashing on this key to pin all requests with the same key to the same backend replica.

**Header:**

```
x-session-affinity: <affinity_key>
```

**Key format:** For GRPO, the affinity key groups all N completions from the same prompt: `{eval_name}_{group_index}`. For multi-turn rollouts, a UUID is assigned per sample and reused across turns.

This matters for GRPO (N completions per prompt share a prefix) and multi-turn agent rollouts (subsequent turns reuse the KV cache from earlier turns).

---

## 5. Request Weight Admission

Inference requests may include a `weight_version` constraint. This is
admission control, not the trainer's staleness policy: it lets a provider reject
requests that land on replicas which cannot produce a sample the trainer has
already decided would be usable.

```json
{
  "weight_version": {
    "exact_version": 123
  }
}
```

or:

```json
{
  "weight_version": {
    "min_required_version": 123
  }
}
```

If the local replica cannot satisfy the constraint, return `409` with a
retryable error body:

```json
{
  "error": {
    "type": "WeightVersionNotReady",
    "current_version": 122,
    "target_version": 123
  }
}
```

`exact_version` also rejects replicas that have already advanced past the target
with `WeightVersionTooOld`. Trainers use these fields to avoid wasted rollout
compute with opaque routing; trainer schedules and off-policy correction still
define which versions are acceptable.

---

## 6. MoE Router Replay

For MoE models, the provider returns per-token expert routing decisions so the trainer can replay them during backprop (reduces training-inference divergence — [R3 paper](https://arxiv.org/abs/2510.11370)).

**Request:** include in the inference request body:

```json
{
  "include_routing_matrix": true,
  "logprobs": true
}
```

**Response:** each token in `logprobs.content[i]` includes:

```json
{
  "token": "...",
  "logprob": -0.00014,
  "routing_matrix": "<base64-encoded uint8 array>"
}
```

`routing_matrix` shape: `[num_moe_layers, num_active_experts]` (e.g., 58 layers × 8 experts = 464 bytes per token for DeepSeek V3).

The routing decisions are replayed during the training forward pass, ensuring MoE layers use the same expert assignments as inference.

---

## 7. Stitch Integration Notes

See [../cookbook/standalone_rollouts/README.md](../cookbook/standalone_rollouts/README.md)
for a standalone SGLang rollout-provider example. It also includes an optional
SLIME publish-only harness that writes each weight version through a Modal
S3 bucket mount, announces it through this hot-load API, and polls pool
readiness before rollouts continue.

The pool-readiness query is the main protocol feature worth carrying back into
`stitch`: trainers need a provider-agnostic way to ask "how much of the rollout
pool can serve the target weights?" without knowing how the provider schedules
or replaces replicas. `stitch.protocol.RolloutPoolState` models this shape:

```json
{
  "protocol_version": 1,
  "replicas": [
    {
      "replica_id": "replica-a",
      "readiness": true,
      "current_snapshot_identity": "weight_v000123",
      "current_version": 123,
      "sync_state": "IDLE"
    },
    {
      "replica_id": "replica-b",
      "readiness": false,
      "current_snapshot_identity": "weight_v000122",
      "readiness_reason": "downloading weights"
    }
  ]
}
```

Providers do not need to use both `current_snapshot_identity` and
`current_version`; one stable target identifier is enough. A trainer counts a
replica as ready only when `readiness` is true and the target identity/version
matches, then compares that fraction with its configured threshold.
