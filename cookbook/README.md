# Cookbook

The cookbook contains working Modal deployments for Miles and Slime trainers
with SGLang rollout engines. `EXPERIMENT_CONFIG` selects a config from the
trainer’s `configs` directory.

## Weight update mode

Each recipe sets `SGLANG_DELTA_UPDATE_MODE` explicitly:

| Mode | Host state | Commit path | Use when |
| --- | --- | --- | --- |
| `disk` | Mutable checkpoint on local storage | Load the complete checkpoint from disk | Host RAM is constrained, or the trainer publishes full checkpoints |
| `cpu` | Canonical checkpoint plus rank-ready weight images in RAM | Copy prepared images to the GPUs | Deltas are used and the shortest engine pause is worth the RAM |

Both modes reconstruct and verify the complete target in canonical checkpoint
space. CPU mode accepts deltas only; switching runs requires replacing the
replica. Disk mode accepts full checkpoints and deltas and can reset a live
replica.

See the paired
[`glm45_air_fp8`](miles_disagg/configs/glm45_air_fp8.py) and
[`glm45_air_fp8_disk`](miles_disagg/configs/glm45_air_fp8_disk.py) configs.
Memory sizing and SGLang implementation details are in
[`SGLANG_FORK.md`](common/SGLANG_FORK.md).

## External speculative drafts

Set `modal.draft_volume` and, for a volume in another Modal environment,
`modal.draft_volume_env`. The rollout server mounts the volume at `/draft`;
point `--speculative-draft-model-path` at the checkpoint beneath it. Draft
weights are loaded at startup and remain fixed across target-weight updates.
The draft acceptance rate may change as the target evolves; changing both
models requires restarting the replica because they cannot yet be committed
atomically. Bundled MTP heads do not need a separate volume.

## Prepare and launch

Miles:

```bash
EXPERIMENT_CONFIG=glm45_air_fp8 \
  uv run --extra modal modal run -d -m cookbook.miles_disagg.prep_app::prepare_checkpoints
EXPERIMENT_CONFIG=glm45_air_fp8 \
  uv run --extra modal modal run -d -m cookbook.miles_disagg.prep_app::prepare_torch_dist
EXPERIMENT_CONFIG=glm45_air_fp8 \
  uv run --extra modal modal run -d -m cookbook.miles_disagg.prep_app::prepare_dataset
EXPERIMENT_CONFIG=glm45_air_fp8 \
  uv run --extra modal python -m cookbook.miles_disagg.launch
```

Slime:

```bash
EXPERIMENT_CONFIG=kimi_k2_6_int4 \
  uv run --extra modal modal run -d -m cookbook.slime_disagg.prep_app::download_model
EXPERIMENT_CONFIG=kimi_k2_6_int4 \
  uv run --extra modal modal run -d -m cookbook.slime_disagg.prep_app::prepare_dataset
EXPERIMENT_CONFIG=kimi_k2_6_int4 \
  uv run --extra modal python -m cookbook.slime_disagg.launch
```

Preparation is idempotent. Each launch creates an isolated run, waits for the
rollout pool to become ready, starts training, and prints the app name and stop
command.

## Scale a running rollout fleet

Use the app name printed by the launcher:

```bash
uv run --extra modal python -c \
  "from stitch.pools.modal_flash import ModalFlashPool; ModalFlashPool('<app-name>', 'Server').scale(min=4, max=4)"
```

New replicas stay out of rotation while loading the base checkpoint and catching
up to `latest`. They join automatically when ready; the trainer does not need
per-replica coordination.

## Verify a running pool

```bash
uv run --extra modal python -m cookbook.common.smoke \
  --app-name <app-name> \
  --model-name /prep/glm45-air/fp8 \
  --weight-version 10
```

The smoke check verifies generation and weight-version reporting through the
pool gateway and every live replica.

## Profile one weight update

The GLM-4.5-Air FP8 and Kimi K2.6 NVFP4 profilers accept
`--update-mode disk|cpu`:

```bash
uv run --extra modal modal run -d \
  tools/profiling/glm45_air_fp8_delta_weight_update.py \
  --update-mode cpu
```

They exercise generation during staging, commit the prepared target, validate
post-update generation, and report timing and resource usage. On the pinned
SGLang, DFlash and DSPARK reject logprob-returning generation requests while
speculation is enabled. The profiler therefore checks repeated deterministic
text for those algorithms and uses token IDs plus logprobs otherwise.
