# stitch

A framework-agnostic protocol for disaggregated reinforcement learning of
LLMs.

When training and rollout generation run on separate machines, the rollout
servers need a way to pick up new weights as the trainer produces them, and
each rollout request needs to know which weight version it was served with.
`stitch` provides the protocol and glue for that. Trainers publish immutable,
versioned weight artifacts to a shared store, rollout servers sync to a
requested version, and completion requests declare which versions they accept.

`stitch` is unopinionated about algorithm/training framework but is strongly
opinionated about supporting workloads that are:
- async-first
- agentic-first
- elastic rollout compute

## How it works

1. After an optimizer step, the trainer writes weight artifacts (e.g. sparse
   deltas) for version `v` under the store, then advances the `latest` pointer.
   The version's HF `model.safetensors.index.json` *is* its manifest.
2. A sidecar in front of each rollout server reconciles to `latest`. When a
   request arrives pinned to version `v` (a `weight_version` constraint in the
   request body), the replica applies versions in order until it reaches `v`,
   then proxies the request to the engine.
3. Responses carry the version they were served with, and a replica that hasn't
   caught up returns `409` (retryable) so the trainer can wait or reroute.

## Layout

Core library (`src/stitch/`, framework- and provider-agnostic):
- `versions.py`: domain vocabulary — `VersionRef`, `VersionManifest`
  (`kind = full` anchor | `delta`), the monotonic-writer pointer rule
  (`decide_pointer_move`), and replica/pool readiness state.
- `sync.py`: `Reconciler` drives one replica to the store's `latest` pointer
  (stage the chain, reload once, flip the served version), with `quiesce`
  (drain→apply) and `in_place` (pause/apply/continue) commit modes.
- `service.py`: the versioned proxy sidecar (`create_app`) + cross-replica
  `readiness`.
- `publish.py`: trainer-side `publish_version()` / `claim_run()` /
  `constrain_request()`.
- Three ports (each a plain base class + instances):
  - `stores/`: `Store` + `ModalVolumeStore`, `S3Store` — where versions and the
    pointer live.
  - `engines/`: `Engine` + `SGLangEngine` — drive one inference engine
    (`/pull_weights`, `/update_weights_from_disk`).
  - `pools/`: `Pool` + `ModalFlashPool` — reach / enumerate / wake / scale the
    replica set (a client to a running pool, not its deployment).

Recipes (`cookbook/`, non-core deployments):
- `common/`: framework-agnostic shared layer (image builds, the sidecar
  entrypoint, the shared publish/claim/request hooks, launch/Ray/smoke helpers).
- `miles_disagg/`, `slime_disagg/`: the miles and slime disaggregated-rollout
  recipes, with per-experiment configs under `configs/`.

The core package has no required dependencies; extras pull in what each adapter
needs (`modal`, `sglang`, `boto3`).

## The sglang fork

Rollout engines run a patched sglang (the disaggregated `/pull_weights`, correct
quantized reloads, and the O(delta) partial reload are not upstream yet). The pin
and its docs live next to each other in `cookbook/common/`:
**[`SGLANG_FORK.md`](cookbook/common/SGLANG_FORK.md)** documents the full patch stack,
the upstreaming PRs, and how to re-port the patches onto a newer sglang release.

## Elastic rollout — spin up engines mid-run

The pool is a set of independent, self-syncing replicas: each reads the
authoritative `latest` pointer and converges itself, so adding a replica
mid-run needs no coordination. Scale the Modal Flash pool up and the new
containers boot, base-seed, replay the delta chain to the current version, and
join the rotation on their own:

```bash
# bump the floor from 2 -> 4; Flash boots 2 more containers that self-sync
python -c "from stitch.pools.modal_flash import ModalFlashPool; \
  ModalFlashPool('<app-name>', 'Server').scale(min=4, max=4)"
```

(`scale` calls the deployed `Server`'s `update_autoscaler(min_containers=...,
max_containers=...)`.) A joiner pays a one-time cost — full base materialize
plus replay from the newest anchor — which periodic full anchors bound. While
it is still behind, version-pinned requests it cannot serve yet get a retryable
`409` and route to caught-up replicas, so it never emits a stale-version
generation.

## Adding adapters

Trainer adapters publish canonical Hugging Face tensor names so engine adapters
stay trainer-agnostic. New stores / engines / pools are new subclasses behind
the corresponding port (zero core edits); new deployments/models are new
`cookbook/` recipes.

## Development

```bash
uv run pytest
```
