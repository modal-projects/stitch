# stitch

Stitch synchronizes policy weights from an RL trainer to an elastic rollout
fleet.

The trainer publishes immutable, versioned full checkpoints or deltas to a
shared store. Every rollout replica independently converges to the published
version, and every response reports which version generated it.

## What Stitch provides

- **Asynchronous weight updates.** Replicas stage and verify the next version
  while serving. Only the final engine commit gates new requests.
- **Version-correct rollouts.** Requests can require a minimum or exact weight
  version. A replica that cannot satisfy the requirement returns a retryable
  `409` instead of serving stale weights.
- **Dynamic rollout fleets.** New replicas load the base checkpoint, catch up to
  the current version, and enter rotation without trainer coordination.
- **Failure-safe publication.** Checkpoint bytes become durable before the
  shared pointer advances, and a replica reports a new version only after its
  engine commits successfully.
- **Pluggable infrastructure.** Trainers, stores, inference engines, and rollout
  pools integrate through separate `Store`, `Engine`, and `Pool` interfaces.

## How it works

1. The trainer publishes `weight_vNNNNNN` and advances the shared `latest`
   pointer.
2. Each replica materializes and verifies the required full checkpoint or delta
   lineage.
3. The replica prepares the target while generation continues, then commits it
   through a short admission gate.
4. Requests and responses carry the weight-version information needed for
   staleness control and sample attribution.

Replicas reconcile independently. A missed notification only delays an update:
periodic reconciliation still converges the fleet to `latest`.

## Included integrations

- Modal Volume and S3 checkpoint stores
- SGLang rollout engines
- Modal Flash rollout pools
- Miles and Slime deployment recipes

The [`cookbook guide`](cookbook/README.md) covers configuration, launch,
live scaling, and validation. Fork pins and re-porting guidance live in
[`SGLANG_FORK.md`](cookbook/common/SGLANG_FORK.md) and
[`MILES_FORK.md`](cookbook/miles_disagg/MILES_FORK.md).

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
