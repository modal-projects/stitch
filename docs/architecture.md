# Disaggregated Rollout Architecture

This package separates rollout versioning from trainer implementation details.
The core protocol is trainer-agnostic: trainers publish immutable Hugging
Face-named weight artifacts for version `v`, rollout servers sync to `v`, and
completion requests declare which weight versions are acceptable.

## Layers

- Core protocol: request policy, response metadata, version manifests, latest
  pointer, and sync state names.
- Bulletin board: durable storage for immutable manifests and artifacts.
- Trainer adapters: framework-specific hooks that publish canonical artifacts.
- Engine adapters: inference-engine-specific prepare/commit operations.
- Provider adapters: infrastructure-specific helpers such as Modal Volume commit
  and Modal Flash container discovery.

Example app orchestration stays outside this package. Modal image construction,
Flash smoke checks, process lifecycle helpers, and one-off launcher entrypoints
belong in the consuming example or application.

## Trainer Boundary

Trainer adapters should converge on canonical Hugging Face tensor names. Slime
and Miles already expose Megatron-to-HF iterator shapes, so those adapters can
publish the same artifact protocol even though their process topology differs.
FSDP, SkyRL, and JAX adapters should be added only when they can emit the same
canonical tensor batches or already materialized artifacts.

## Engine Boundary

Canonical HF form simplifies trainer export, but each rollout engine still owns
application semantics. The first engine adapter targets SGLang disk deltas via
`/flush_cache` and `/update_weights_from_disk`. Future adapters can implement
the same prepare/commit contract without changing the request protocol.

## V1 Constraints

The v1 SGLang sidecar uses a conservative commit policy: it waits for active
proxied requests to drain before applying weights. This preserves exact-version
correctness and avoids assuming that every engine can safely update while a
request is generating. The protocol still records start/end versions so looser
async policies can be added later.
