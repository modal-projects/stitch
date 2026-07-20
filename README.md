# stitch

A framework-agnostic protocol for **disaggregated reinforcement learning** of LLMs.

When training and rollout run on separate machines, the rollout servers need new weights as
the trainer produces them, and every sample needs to know which weight version produced it.
stitch is the glue: trainers publish immutable, versioned weight artifacts to a shared store;
rollout replicas sync themselves to a requested version; requests declare which versions they
accept, and responses report which version served them.

Opinionated about the workload, not the algorithm or framework:
- **async-first** — the trainer never blocks on rollout, nor rollout on the trainer
- **agentic-first** — a sample can be a long multi-turn episode spanning weight versions
- **elastic** — rollout replicas join and leave mid-run

## What it does that a raw endpoint can't

1. **Weight sync** — each published version reloads into the live engine *in place*, across the
   pool, without stopping generation. A version is a full anchor or a sparse delta against an
   earlier one; the engine walks back to the nearest anchor and replays deltas forward.
2. **Version correctness** — every sample is attributable to the version(s) that produced it, and
   the trainer can bound staleness. A replica behind the pinned version returns a retryable `409`
   rather than a stale generation.
3. **Elastic sync** — a replica added mid-run boots, catches up over the delta chain, and joins on
   its own, with no coordination: each reconciles itself to the authoritative `latest` pointer.
4. **Coordination** — full-vs-delta apply, session affinity, and MoE router replay, agreed across
   publish and serve.

## How it works

After an optimizer step the trainer publishes version `v` under the store, then advances `latest`
— the version's HF `model.safetensors.index.json` *is* its manifest. A sidecar in front of each
replica reconciles to `latest`, reloading new versions in place while it keeps serving; a request
pinned to `v` waits until the replica has applied up to `v`, then proxies to the engine, and the
response carries the version that served it. Scale the pool up mid-run and new containers self-sync:

```bash
python -c "from stitch.pools.modal_flash import ModalFlashPool; \
  ModalFlashPool('<app-name>', 'Server').scale(min=4, max=4)"
```

## The sglang fork

Rollout engines run a patched sglang — disaggregated `/pull_weights`, correct quantized reloads,
and O(delta) partial reload, none upstream yet. The pins, the patch stack, and how to re-port live
in **[cookbook/common/SGLANG_FORK.md](cookbook/common/SGLANG_FORK.md)**.

## Layout

- `src/stitch/` — the library: the domain vocabulary, the sync / serve / publish logic, and the
  three ports (**Store**, **Engine**, **Pool**) with their instances. Framework-, engine-, and
  provider-agnostic; no required dependencies (extras pull in `modal` / `sglang` / `boto3`).
- `cookbook/` — deployments, not core: `common/` (image builds, the sidecar, the
  publish/claim/request hooks, launch helpers) and the `miles_disagg/` / `slime_disagg/` recipes
  with per-experiment `configs/`.

The line between them is the one rule: a different store / engine / pool is a new subclass behind
its port (zero core edits); a different deployment or model is a new `cookbook/` recipe. Trainer
adapters publish canonical Hugging Face tensor names, so engine adapters stay trainer-agnostic.

## Development

```bash
uv run pytest        # co-located *_test.py, incl. the in-memory core harness (no Modal/GPU)
```
