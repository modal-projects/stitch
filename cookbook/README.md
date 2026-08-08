# Cookbook

The cookbook contains working Modal deployments for Miles and Slime trainers
with SGLang rollout engines. Select a model through `EXPERIMENT_CONFIG`.

## Choose an update mode

Each recipe sets `SGLANG_DELTA_UPDATE_MODE`:

| Mode | Prepared state | During engine pause | Use when |
| --- | --- | --- | --- |
| `disk` | Complete checkpoint on local storage | Reload from disk | Host RAM is constrained or the trainer publishes full checkpoints |
| `cpu` | Rank-ready images in RAM; canonical checkpoint in RAM or on local storage | Copy the images to GPU | Updates are deltas and minimizing the pause justifies rank-image RAM |

Both modes reconstruct and checksum the complete target in canonical checkpoint
space. CPU mode accepts deltas only and requires a new replica for a new
lineage. Disk mode accepts full checkpoints and deltas and can reset a live
replica.

Every bundled recipe serves `cpu` updates except the Kimi K3 example, which
stays on `disk` pending its own benchmark (`cpu` dominated on every model
family benchmarked so far). The profiling scripts exercise both modes.
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

All persistent storage uses Modal Volume v2:

| Volume | Mount | Contents |
| --- | --- | --- |
| `huggingface-cache` | `/root/.cache/huggingface` | Pinned source-model downloads |
| `miles-checkpoints` or `slime-checkpoints` | `/checkpoints` | Immutable prepared model layouts |
| `miles-data` or `slime-data` | `/data` | Pinned datasets |
| `stitch-<framework>-<model>` | `/stitch` | Run-scoped publications, checkpoints, and logs |
| `sglang-cache` | `/root/.cache/sglang` | Compiled SGLang kernels |
| Configured draft Volume | `/draft` | Optional external speculative draft |

Prepared model layouts are immutable artifacts with explicit paths. For
example, the Miles GLM-5.2 config uses:

```text
/checkpoints/
├── glm5-2-nvfp4/
└── glm5-2-torch-dist/
```

Each config also owns one `stitch-<framework>-<model>` Volume mounted at
`/stitch`. Its root contains only run-scoped state:

```text
/stitch/
└── <run-id>/
    ├── latest
    ├── updates/weight_vNNNNNN/
    ├── checkpoints/
    └── train.log
```

The training framework owns `updates/`; Stitch owns the `latest` commit
pointer. A future checkpoint resume starts a new run and reads the previous
run's `checkpoints/` rather than appending to its update chain.

Datasets are independent of models and runs. Miles mounts the shared
`miles-data` Volume at `/data`; Slime uses `slime-data`. Each dataset has a
stable directory; `manifest.json` records its pinned source revision.

```text
/data/
└── <dataset>/
    ├── manifest.json
    ├── <trainer-input>
    └── <dataset-specific-assets>/
```

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

The GLM-4.7-Flash SWE-bench Pro recipe provisions two eight-GPU H200 trainer
nodes and 48 rollout GPUs as twelve TP4 engines. Prepare its immutable assets in
dependency order, then launch an isolated run:

```bash
export EXPERIMENT_CONFIG=glm47_flash_swebench_pro

uv run --extra modal modal run -d \
  -m cookbook.miles_disagg.prep_app::prepare_checkpoints
uv run --extra modal modal run -d \
  -m cookbook.miles_disagg.prep_app::prepare_torch_dist
uv run --extra modal modal run -d \
  -m cookbook.miles_disagg.prep_app::prepare_dataset
uv run --extra modal python -m cookbook.miles_disagg.launch
```

Checkpoint preparation pins and materializes the BF16 Hugging Face source;
torch-dist preparation converts that source for the two-node trainer. Dataset
preparation writes the pinned SWE-bench Pro prompts, task environments, and
verifiers. Each step is idempotent.

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
  --model-name /checkpoints/glm45-air-fp8 \
  --weight-version 10
```

## Profile an update

The model profilers prepare their pinned base checkpoint and synthetic delta,
then run with `--update-mode disk|cpu`:

```bash
uv run --extra modal modal run -d \
  tools/profiling/glm45_air_fp8_delta_weight_update.py \
  --update-mode cpu
```

Prepared artifacts are reused. The profilers generate during staging, pause
the engine to activate the target, validate the new version, and report timing
and resource use. The pinned SGLang runtime returns aligned verifier logprobs
for DFlash. DSpark still rejects logprob-returning requests, so its profiles
compare repeated deterministic text instead of token IDs and logprobs.

The K3 profiler downloads the pinned public checkpoint and constructs a
checksummed XOR publication covering every checkpoint tensor:

```bash
MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
  tools/profiling/kimi_k3_mxfp4_delta_weight_update.py \
  --update-mode cpu
```
