# Cookbook

The cookbook contains working Modal deployments for Miles and Slime trainers
with SGLang rollout engines. Select a model through `EXPERIMENT_CONFIG`.

## Choose an update mode

Each recipe sets `SGLANG_DELTA_UPDATE_MODE`:

| Mode | Prepared state | Engine commit | Use when |
| --- | --- | --- | --- |
| `disk` | Complete checkpoint on local storage | Reload from disk | Host RAM is constrained or the trainer publishes full checkpoints |
| `cpu` | Canonical checkpoint and rank-ready weight images in RAM | Copy the images to GPU | Updates are deltas and minimizing the pause justifies the RAM |

Both modes reconstruct and checksum the complete target in canonical checkpoint
space. CPU mode accepts deltas only and requires a new replica for a new
lineage. Disk mode accepts full checkpoints and deltas and can reset a live
replica.

See the paired
[`glm45_air_fp8`](miles_disagg/configs/glm45_air_fp8.py) and
[`glm45_air_fp8_disk`](miles_disagg/configs/glm45_air_fp8_disk.py) configs.
Memory sizing and SGLang details are in
[`SGLANG_FORK.md`](common/SGLANG_FORK.md).

## External speculative drafts

Set `modal.draft_volume` and, when needed, `modal.draft_volume_env`. The server
mounts the volume at `/draft`; set `--speculative-draft-model-path` to the
checkpoint below it.

Draft weights remain fixed while target weights update. Their acceptance rate
may change as the target evolves. Updating both atomically requires restarting
the replica. Bundled MTP heads do not need a separate volume.

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

Preparation is idempotent. The launcher creates an isolated run, waits for the
rollout pool, starts training, and prints the app name and stop command.

## Scale and inspect a run

```bash
uv run --extra modal python -c \
  "from stitch.pools.modal_flash import ModalFlashPool; ModalFlashPool('<app-name>', 'Server').scale(min=4, max=4)"
```

`min` is the number of rollout replicas kept running; `max` is the autoscaling
ceiling. Setting both to `4` pins the fleet at four replicas. New replicas
remain outside rotation until they load the base and catch up. Verify the
gateway and every live replica with:

```bash
uv run --extra modal python -m cookbook.common.smoke \
  --app-name <app-name> \
  --model-name /prep/glm45-air/fp8 \
  --weight-version 10
```

## Profile an update

The model profilers support `--update-mode disk|cpu`:

```bash
uv run --extra modal modal run -d \
  tools/profiling/glm45_air_fp8_delta_weight_update.py \
  --update-mode cpu
```

They generate during staging, commit the target, validate the new version, and
report timing and resource use. SGLang rejects logprob-returning requests when
DFlash or DSPARK is enabled, so those profiles compare repeated deterministic
text instead of token IDs and logprobs.

The K3 profiler downloads the pinned public checkpoint and constructs a
checksummed XOR publication covering every checkpoint tensor:

```bash
MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
  tools/profiling/kimi_k3_mxfp4_delta_weight_update.py \
  --update-mode cpu
```
