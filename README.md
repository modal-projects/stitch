# stitch

A framework-agnostic protocol for **disaggregated reinforcement learning** of LLMs.

When training and rollout generation run on separate machines, the rollout servers need
new weights as the trainer produces them, and every rollout sample needs to know which
weight version produced it. stitch is the protocol and glue for that: trainers publish
immutable, versioned weight artifacts to a shared store; rollout replicas sync themselves
to a requested version; completion requests declare which versions they accept, and
responses report which version served them.

It is unopinionated about the algorithm and training framework, and opinionated about the
workloads it targets:
- **async-first** — the trainer never blocks on rollout, nor rollout on the trainer
- **agentic-first** — a sample can be a long multi-turn episode spanning weight versions
- **elastic** — rollout replicas join and leave mid-run

## What it does that a raw endpoint can't

1. **Weight sync** — get each published version into an elastic, multi-replica pool and
   reloaded into the live engine, in place, without stopping generation.
2. **Version correctness** — every sample is attributable to the weight version(s) that
   produced it, and the trainer can bound staleness (serve only within N versions of
   `latest`).
3. **Elastic sync** — a replica added mid-run boots, catches up over the delta chain, and
   joins the rotation on its own; one still behind rejects requests it cannot serve yet
   (retryable `409`) rather than emit a stale-version generation.
4. **Coordination** — full-vs-delta apply, session affinity, and MoE router replay, agreed
   across publish and serve.

## How it works

1. After an optimizer step, the trainer publishes version `v`'s weight artifacts (a full
   anchor, or a sparse delta against an earlier version) under the store, then advances the
   `latest` pointer. The version's HF `model.safetensors.index.json` *is* its manifest.
2. A sidecar in front of each rollout replica reconciles to `latest`. A request pinned to
   version `v` waits until the replica has applied the chain up to `v`, then proxies to the
   engine; the engine reloads new versions in place while it keeps serving.
3. Responses carry the version that served them; a replica behind the pinned version
   returns a retryable `409`, so the caller waits or reroutes and never gets a stale
   generation.

## Elastic rollout — spin up engines mid-run

The pool is a set of independent, self-syncing replicas: each reads the authoritative
`latest` pointer and converges itself, so adding one needs no coordination. Scale the Modal
Flash pool up and the new containers boot, base-seed, replay the delta chain to the current
version, and join the rotation on their own:

```bash
# bump the floor from 2 -> 4; Flash boots 2 more containers that self-sync
python -c "from stitch.pools.modal_flash import ModalFlashPool; \
  ModalFlashPool('<app-name>', 'Server').scale(min=4, max=4)"
```

A joiner pays a one-time catch-up (materialize the base, replay from the newest anchor),
which periodic full anchors bound. While it is behind, version-pinned requests it cannot
serve yet get a retryable `409` and route to caught-up replicas.

## Pinning the rollout region

Pin the rollout pool to the region nearest the trainer — or the one required by data
residency — e.g. `us-west`, `eu-west`, `ap-south`. It is a per-deployment setting on the
Server: the config's `proxy_regions` (the Flash edge the front door consistent-hashes to)
and `region` (where the GPU replicas run). Set them explicitly rather than relying on
Modal's default placement, so the trainer↔rollout hop stays in-region.

## The sglang fork

Rollout engines run a patched sglang: the disaggregated `/pull_weights`, correct quantized
reloads, and the O(delta) partial reload are not upstream yet. The pin and its rationale
live in **[cookbook/common/SGLANG_FORK.md](cookbook/common/SGLANG_FORK.md)** — the full
patch stack, the upstreaming PRs, and how to re-port onto a newer sglang release.

## Layout

- `src/stitch/` — the library: the domain vocabulary and the sync / serve / publish logic,
  plus the three ports (**Store**, **Engine**, **Pool**) and their instances. Framework-,
  engine-, and provider-agnostic; no required dependencies (extras pull in `modal` /
  `sglang` / `boto3`).
- `cookbook/` — deployments, not core: `common/` (shared image builds, the sidecar, the
  publish/claim/request hooks, launch helpers) and the `miles_disagg/` / `slime_disagg/`
  recipes with per-experiment `configs/`.

## Adding to it

- A different store / engine / pool is a new subclass behind the port — zero core edits.
- A different deployment or model is a new `cookbook/` recipe — core never changes.
- Trainer adapters publish canonical Hugging Face tensor names, so engine adapters stay
  trainer-agnostic.

## Development

```bash
uv run pytest
```
