# Cookbook

The cookbook contains runnable Modal deployments that connect Miles or Slime
trainers to elastic SGLang rollout pools through Stitch. Recipes define the
model, trainer, rollout fleet, data, and weight-update policy; the shared
infrastructure handles preparation, isolated runs, and pool lifecycle.

## Common workflow

### 1. Select a recipe

Set `EXPERIMENT_CONFIG` to a module in `miles_disagg/configs` or
`slime_disagg/configs`:

```bash
export EXPERIMENT_CONFIG=your_config_name
export MODAL_ENVIRONMENT=your_environment
```

The selected config is the authority for model revisions, Volume names,
hardware, scaling, and trainer arguments.

### 2. Create credentials

Create the secrets required by the selected recipe once in its Modal
environment:

```bash
export HF_TOKEN=your_hugging_face_token
export WANDB_API_KEY=your_wandb_api_key

uv run --extra modal modal secret create \
  huggingface-secret HF_TOKEN="$HF_TOKEN"
uv run --extra modal modal secret create \
  wandb-secret WANDB_API_KEY="$WANDB_API_KEY"
```

Preparation jobs use `huggingface-secret`. Recipes that enable W&B logging use
`wandb-secret`.

### 3. Prepare immutable inputs

Miles recipes expose checkpoint, TorchDist, and dataset preparation:

```bash
uv run --extra modal modal run -d \
  -m cookbook.miles_disagg.prep_app::prepare_checkpoints
uv run --extra modal modal run -d \
  -m cookbook.miles_disagg.prep_app::prepare_torch_dist
uv run --extra modal modal run -d \
  -m cookbook.miles_disagg.prep_app::prepare_dataset
```

Slime recipes expose model and dataset preparation:

```bash
uv run --extra modal modal run -d \
  -m cookbook.slime_disagg.prep_app::download_model
uv run --extra modal modal run -d \
  -m cookbook.slime_disagg.prep_app::prepare_dataset
```

Preparation is idempotent. A complete artifact is reused; an incomplete one
fails rather than becoming a launch input. Model preparation must finish before
dependent format conversion. Dataset preparation is independent.

### 4. Launch an isolated run

Use the launcher for the selected trainer:

```bash
# Miles
uv run --extra modal python -m cookbook.miles_disagg.launch

# Slime
uv run --extra modal python -m cookbook.slime_disagg.launch
```

The launcher creates an eight-character run ID, deploys a run-scoped rollout
pool, waits for its gateway, and starts the trainer. It prints the app name and
stop command. Repeating the command creates a separate run and checkpoint
lineage.

### 5. Inspect or change a live run

Follow logs using the app name printed by the launcher:

```bash
export APP_NAME=your_app_name
uv run --extra modal modal app logs -f "$APP_NAME"
```

Verify that the gateway and every live replica serve an expected version:

```bash
export MODEL_NAME=/checkpoints/your_model

uv run --extra modal python -m cookbook.common.smoke \
  --app-name "$APP_NAME" \
  --model-name "$MODEL_NAME" \
  --weight-version 10
```

Rollout capacity is controlled by `rollout_min_containers`,
`rollout_max_containers`, and `rollout_target_inputs`. Engine concurrency and
backpressure are controlled by `--max-running-requests` and
`--max-queued-requests` in the recipe.

After changing fleet or SGLang settings, redeploy the active run with the same
experiment and run ID:

```bash
EXPERIMENT_CONFIG=your_config_name RUN_ID=your_run_id \
  uv run --extra modal modal deploy -m cookbook.miles_disagg.app
```

Modal rolls replicas to the new configuration without changing the run's
checkpoint lineage.

## GLM-4.7 Flash example

`glm47_flash_swebench_pro` is a complete example of the workflow above. It runs
fully asynchronous Miles training on SWE-bench Pro.

| Component | Configuration |
| --- | --- |
| Trainer | 4 nodes × 8 H200 GPUs |
| Rollout | 48 replicas × 1 H200 GPU |
| Model | `zai-org/GLM-4.7-Flash`, pinned BF16 revision |
| Dataset | SWE-bench Pro, including task environments and verifiers |
| Weight sync | Checksummed XOR deltas with CPU preparation and in-place activation |

The config is
[`glm47_flash_swebench_pro.py`](miles_disagg/configs/glm47_flash_swebench_pro.py).
After creating the shared secrets, prepare and launch it with:

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

Checkpoint preparation materializes the pinned BF16 model. TorchDist
preparation converts it for the four-node trainer. Dataset preparation writes
the pinned prompts, task environments, verifiers, and source manifest.

### Weight-update performance

These are steady-state measurements on H200s. Cold replica initialization,
fleet replacement, and checkpoint-save steps are excluded.

| Stage | Sample | Mean | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| Trainer XOR delta encode and publish | 10 updates | 15.0 s | 13.6 s | 22.4 s |
| Replica preparation while serving | 58 replica updates | 15.8 s | 15.6 s | 18.0 s |
| Engine pause to activate weights | 58 replica updates | 0.75 s | 0.58 s | 1.13 s |

The trainer updates changed 0.252% of rollout-visible bytes on average and
produced 0.469 GB compressed deltas. The complete steady-state path is roughly
30–35 seconds per update; only activation pauses inference. Validation
published 25 consecutive versions, with exact-version smoke checks through
v25.

## Weight-update modes

Each recipe sets `SGLANG_DELTA_UPDATE_MODE`:

| Mode | Prepared state | During engine pause | Use when |
| --- | --- | --- | --- |
| `disk` | Complete checkpoint on local storage | Reload from disk | Host RAM is constrained or the trainer publishes full checkpoints |
| `cpu` | Rank-ready images in RAM; canonical checkpoint in RAM or on local storage | Copy the images to GPU | Updates are deltas and minimizing the pause justifies rank-image RAM |

Both modes reconstruct and checksum the complete target in canonical checkpoint
space. CPU mode accepts deltas only and requires a new replica for a new
lineage. Disk mode accepts full checkpoints and deltas and can reset a live
replica.

Every bundled recipe serves `cpu` updates except Kimi K3, which uses `disk`.
The profiling scripts exercise both modes. Memory sizing and SGLang details are
in [`SGLANG_FORK.md`](common/SGLANG_FORK.md).

## Persistent storage

All persistent storage uses Modal Volume v2:

| Volume | Mount | Contents |
| --- | --- | --- |
| `huggingface-cache` | `/root/.cache/huggingface` | Pinned source-model downloads |
| `miles-checkpoints` or `slime-checkpoints` | `/checkpoints` | Immutable prepared model layouts |
| `miles-data` or `slime-data` | `/data` | Pinned datasets |
| `stitch-<framework>-<model>` | `/stitch` | Run-scoped publications, checkpoints, and logs |
| `sglang-cache` | `/root/.cache/sglang` | Compiled SGLang kernels |
| Configured draft Volume | `/draft` | Optional external speculative draft |

Prepared model layouts have stable paths. For example:

```text
/checkpoints/
├── glm4-7-flash-bf16/
└── glm4-7-flash-torch-dist/
```

Each experiment Volume contains only run-scoped state:

```text
/stitch/
└── <run-id>/
    ├── latest
    ├── updates/weight_vNNNNNN/
    ├── checkpoints/
    └── train.log
```

The training framework owns `updates/`; Stitch owns the `latest` commit
pointer. A resumed checkpoint starts a new run and reads the previous run's
`checkpoints/` rather than appending to its update chain.

Datasets are independent of models and runs:

```text
/data/
└── <dataset>/
    ├── manifest.json
    ├── <trainer-input>
    └── <dataset-specific-assets>/
```

## External speculative drafts

Set `modal.draft_volume` and, when needed, `modal.draft_volume_env`. The server
mounts the volume at `/draft`; set `--speculative-draft-model-path` to the
checkpoint below it.

Draft weights remain fixed while target weights update. Their acceptance rate
may change as the target evolves. Updating both atomically requires restarting
the replica. Bundled MTP heads do not need a separate volume.

## Profile a weight update

The model profilers prepare their pinned base checkpoint and synthetic delta,
then run with `--update-mode disk|cpu`. CPU runs also select
`--canonical-storage memory|disk`; `disk` uses host-local NVMe.

```bash
uv run --extra modal modal run -d \
  tools/profiling/glm45_air_fp8_delta_weight_update.py \
  --update-mode cpu \
  --canonical-storage memory
```

Prepared artifacts are reused. The profilers generate during staging, pause
the engine to activate the target, validate the new version, and report timing
and resource use. The pinned SGLang runtime returns aligned verifier logprobs
for DFlash. DSpark still rejects logprob-returning requests, so its profiles
compare repeated deterministic text instead of token IDs and logprobs.

The K3 profiler downloads the pinned public checkpoint and constructs a
checksummed XOR publication over mutable, rollout-visible values. The fixed
vision tower and projector are excluded.

```bash
uv run --extra modal modal run -d \
  tools/profiling/kimi_k3_mxfp4_delta_weight_update.py \
  --update-mode cpu \
  --canonical-storage disk
```
