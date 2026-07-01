# stitch

A framework-agnostic protocol for disaggregated reinforcement learning of
LLMs.

When training and rollout generation run on separate machines, the rollout
servers need a way to pick up new weights as the trainer produces them, and
each rollout request needs to know which weight version it was served with.
`stitch` provides the protocol and glue for that. Trainers publish immutable,
versioned weight artifacts to a shared "bulletin board" directory, rollout
servers sync to a requested version, and completion requests declare which
versions they will accept.

`stitch` is unopinionated about algorithm/training framework but is strongly
opinionated about supporting workloads that are:
- async-first
- agentic-first
- elastic rollout compute

## How it works

1. After an optimizer step, the trainer writes weight artifacts (e.g. sparse
   deltas) for version `v` to the bulletin board, publishes a
   `manifest.json` describing them, then advances `latest.json`.
2. A sidecar in front of each rollout server watches the board. When a request
   arrives pinned to version `v` (via a `weight_version` constraint in the
   request body), the sidecar applies versions in order until it reaches `v`,
   then proxies the request to the engine.
3. Responses carry the version they were served with, and a server that hasn't
   caught up returns `409 WeightVersionNotReady` so the trainer can retry.

## Layout

- `src/stitch/protocol.py`: Wire protocol. Version manifests, artifacts,
  request policies, the `latest.json` pointer, version-namespaced cache keys.
- `src/stitch/bulletin.py`: Bulletin board storage (filesystem-backed, with
  a pluggable refresh hook for remote volumes).
- `src/stitch/sync.py`: Sync state machine that drives a server from its
  current version to a target with a single in-place commit path
  (pause/apply/advance/continue, no draining).
- `src/stitch/servers/sglang.py`: HTTP sidecar that adds version semantics
  to an SGLang server.
- `src/stitch/engines/sglang.py`: SGLang prepare/commit adapter using
  `/update_weights_from_disk`.
- `src/stitch/trainers/slime.py`: Hooks for
  [slime](https://github.com/THUDM/slime) that publish sparse-delta versions
  from training ranks.
- `src/stitch/providers/modal.py`: Modal helpers for Volume commit/reload and
  Flash container discovery.
- `cookbook/`: End-to-end examples.
  - `slime_disagg/`: SLIME plus a stitch-managed Modal Flash/SGLang pool.
  - `standalone_rollouts/`: standalone Modal/SGLang rollout provider with a hot-load API shim.

The core package has no required dependencies; extras pull in what each
adapter needs (`modal`, `sglang`, `slime`).

## Adding adapters

Trainer adapters should publish canonical Hugging Face tensor names so engine
adapters stay trainer-agnostic. Engine adapters implement the same
prepare/commit contract as the SGLang one; the request protocol doesn't
change.

## Development

```bash
uv run pytest
```
