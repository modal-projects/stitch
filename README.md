# Stitch

Stitch synchronizes policy weights from an RL trainer to an elastic rollout
fleet.

## Guarantees

- **Updates overlap serving.** Replicas stage and verify the next full
  checkpoint or delta before briefly gating new requests for the engine commit.
- **Rollouts are versioned.** Requests can require a minimum or exact version;
  incompatible replicas return a retryable `409`, and responses report their
  start and end versions.
- **The fleet is elastic.** New replicas load the base, catch up to the current
  version, and enter rotation without trainer coordination.
- **Publication is failure-safe.** Checkpoint bytes become durable before the
  shared pointer advances, and a replica reports a version only after its
  engine commits it.
- **Integrations are pluggable.** Stores, inference engines, and rollout pools
  implement the separate `Store`, `Engine`, and `Pool` interfaces.

The trainer publishes immutable versions to a shared store. Replicas reconcile
independently, so a missed notification delays an update but does not prevent
convergence.

## Integrations

Stitch includes Modal Volume and S3 stores, SGLang engines, Modal Flash pools,
and reference Miles and Slime deployments.

See the [cookbook](cookbook/README.md) to choose an update mode, launch a run,
scale the rollout fleet, and validate an update. Fork pins and re-porting notes
are in [SGLANG_FORK.md](cookbook/common/SGLANG_FORK.md) and
[MILES_FORK.md](cookbook/miles_disagg/MILES_FORK.md).

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
