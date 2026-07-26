# stitch

Stitch connects an RL trainer to an elastic rollout fleet. Trainers publish
immutable, versioned Hugging Face checkpoints or deltas; every rollout replica
converges to the shared `latest` pointer; requests can require a weight version;
and responses report which version generated them.

The library is framework-, engine-, and infrastructure-agnostic. The cookbook
contains working Miles and Slime deployments on Modal with SGLang rollout
engines.

## Protocol

After an optimizer step, the trainer writes `weight_vNNNNNN`, verifies that the
artifact is complete, and atomically advances `latest`. Each replica’s Stitch
sidecar then:

1. materializes the published version from the configured store;
2. asks the engine to stage and verify the target while generation continues;
3. briefly gates admission while the engine applies the staged weights; and
4. advances the replica’s served version.

A request can carry a minimum or exact `weight_version`. A replica that cannot
satisfy it returns a retryable `409` instead of silently serving stale weights.
New replicas load v0, catch up to `latest`, and enter rotation without trainer
coordination.

## SGLang delta destinations

Recipes select `SGLANG_DELTA_UPDATE_MODE` explicitly:

- `disk` reconstructs a verified target checkpoint on host-local storage through
  `/stage_weight_update`, then loads it with `/update_weights_from_disk`. It
  preserves host RAM for serving, but storage reads and loader transforms remain
  inside the commit RPC.
- `cpu` keeps a canonical checkpoint snapshot plus one rank-ready image per local
  TP rank. It applies and verifies deltas and builds the next images in the
  background, then `/update_weights_from_cpu` only copies the completed images to
  the GPUs. The one-time cache initialization starts after v0 is serving. If a
  delta arrives first, the update waits for that existing task.

CPU mode accepts delta publications only. A FULL publication or a run reset
requires disk mode or a fresh rollout replica. Both modes reconstruct every
published target and load every runtime weight storage; neither depends on
tensor-level sparsity.

The trainer records checksums of the reconstructed target tensor bytes in every
delta. Both destinations verify those same values in canonical checkpoint tensor
space before TP sharding or runtime layout conversion; that canonical validation
is the parity boundary. A missing file, malformed lineage, size mismatch, or
checksum mismatch fails staging and leaves live GPU weights untouched.

At TP4, the persistent core allocation for CPU mode is approximately:

| Model | Canonical checkpoint | Rank image × 4 | Persistent core |
| --- | ---: | ---: | ---: |
| GLM-4.5-Air FP8 | 112.6 GB | 27.2 GB × 4 | 221.3 GB |
| Kimi K2.6 NVFP4 | about 595 GB | about 151 GB × 4 | about 1.20 TB |

Allow additional memory for the engine process, delta decoding, and bounded
loader staging. The supplied recipes request `(512 GiB, 2 TiB)` for GLM and
`(1 TiB, 3 TiB)` for Kimi, expressed as `(request, limit)`.

## Launching a cookbook job

Select a config with `EXPERIMENT_CONFIG`. Preparation is idempotent and only
needs to run when its model or dataset artifacts are absent:

```bash
EXPERIMENT_CONFIG=glm45_air_fp8 \
  uv run --extra modal modal run -d -m cookbook.miles_disagg.prep_app::prepare_checkpoints
EXPERIMENT_CONFIG=glm45_air_fp8 \
  uv run --extra modal modal run -d -m cookbook.miles_disagg.prep_app::prepare_torch_dist
EXPERIMENT_CONFIG=glm45_air_fp8 \
  uv run --extra modal modal run -d -m cookbook.miles_disagg.prep_app::prepare_dataset
```

Launch an isolated rollout pool and trainer run:

```bash
EXPERIMENT_CONFIG=glm45_air_fp8 \
  uv run --extra modal python -m cookbook.miles_disagg.launch
```

Use `cookbook.slime_disagg.prep_app` and
`cookbook.slime_disagg.launch` for a Slime config. Each launch creates a unique
run ID and prints the app name and stop command.

Check an already-deployed pool without creating another Modal app:

```bash
uv run --extra modal python -m cookbook.common.smoke \
  --app-name <printed-app-name> \
  --model-name /prep/glm45-air/fp8 \
  --weight-version 10
```

## Profiling

The reusable profilers are
[`glm45_air_fp8_delta_weight_update.py`](tools/profiling/glm45_air_fp8_delta_weight_update.py)
and
[`kimi_k2_6_nvfp4_delta_weight_update.py`](tools/profiling/kimi_k2_6_nvfp4_delta_weight_update.py).
Each accepts `--update-mode disk|cpu`, exercises generation during staging,
checks post-update generation and deterministic output, and prints the complete
timing and resource breakdown. Use Modal’s default runtime, or set
`MODAL_FUNCTION_RUNTIME=runc` to measure the alternate runtime.

## Repository layout

- `src/stitch/` contains the version protocol and the Store, Engine, and Pool
  ports.
- `cookbook/common/` contains shared deployment, sidecar, image, and hook code.
- `cookbook/miles_disagg/` and `cookbook/slime_disagg/` contain trainer-specific
  recipes and model configs.
- [`SGLANG_FORK.md`](cookbook/common/SGLANG_FORK.md) and
  [`MILES_FORK.md`](cookbook/miles_disagg/MILES_FORK.md) record the exact fork
  pins and patch stacks.

## Development

```bash
uv run pytest
```
