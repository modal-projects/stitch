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

### 3. Choose a checkpoint store

Modal Volumes are the default checkpoint store. Select S3 when policy
publications should live in object storage instead; the trainer and rollout
protocol otherwise stays the same. See the [S3 store appendix](#s3-store-appendix)
for cookbook setup, both authentication options, and integration with other
trainers.

### 4. Prepare immutable inputs

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

### 5. Launch an isolated run

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

#### Resume a Miles run

Use the same recipe and Modal environment as the source run. A resumable point
is the latest saved Megatron checkpoint with a matching complete Hugging Face
export. Resume it once with:

```bash
export EXPERIMENT_CONFIG=your_config_name
export MODAL_ENVIRONMENT=your_environment

uv run --extra modal python -m cookbook.miles_disagg.launch \
  --resume-from source_run_id
```

The launcher creates a new run ID; do not reuse or pass the source run ID as
`RUN_ID`. If the source checkpoint is v120, the new run boots at v120 and
publishes v121 next.

Add automatic resume to a fresh or resumed launch with:

```bash
uv run --extra modal python -m cookbook.miles_disagg.launch \
  --resume-from source_run_id \
  --auto-resume
```

`--auto-resume` stays attached to the trainer. An unexpected exit starts a new
run from the newest complete checkpoint and retries until the trainer returns
normally. Stop it with Ctrl-C. It is off by default and requires
`save_interval`, `save_hf`, and optimizer/RNG checkpointing. A fresh run becomes
resumable after its first complete Megatron/Hugging Face checkpoint pair.

With the Modal Volume store, complete Hugging Face checkpoints also accelerate
elastic rollout startup. When a Miles recipe saves checkpoints and updates
weights every step, a new replica loads the current run's newest complete
checkpoint no newer than `latest`, then applies only the remaining deltas.
Before the first save, it catches up from the run's configured boot checkpoint.

### 6. Inspect or change a live run

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

Prepared inputs, caches, logs, and trainer checkpoints use Modal Volume v2.
Run publications use either the experiment Volume or S3, according to
`STITCH_STORE_BACKEND`:

| Volume | Mount | Contents |
| --- | --- | --- |
| `huggingface-cache` | `/root/.cache/huggingface` | Pinned source-model downloads |
| `miles-checkpoints` or `slime-checkpoints` | `/checkpoints` | Immutable prepared model layouts |
| `miles-data` or `slime-data` | `/data` | Pinned datasets |
| `stitch-<framework>-<model>` | `/stitch` | Run-scoped checkpoints and logs; publications when using the Volume backend |
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
    ├── hf_checkpoints/weight_vNNNNNN/
    │   └── .complete
    └── train.log
```

The training framework owns `updates/` and the saved checkpoints; Stitch owns
`latest`. A resumed checkpoint starts a new run and reads the previous run's
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

## S3 store appendix

### Use S3 with the cookbook

The cookbook gives each run an isolated prefix below `S3_ROOT`. The AWS
identity used by the trainer and rollout functions needs these permissions,
scoped to that root:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::your-bucket",
      "Condition": {
        "StringLike": {"s3:prefix": ["stitch", "stitch/*"]}
      }
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::your-bucket/stitch/*"
    }
  ]
}
```

Choose one of the following authentication methods. Both use a Modal Secret
named `stitch-s3`; only its credential fields differ.

#### Static AWS access keys

Store the access key and S3 root together. Add `AWS_SESSION_TOKEN` when using
temporary credentials.

```bash
uv run --extra modal modal secret create -e "$MODAL_ENVIRONMENT" stitch-s3 \
  S3_ROOT=s3://your-bucket/stitch \
  AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  AWS_REGION="$AWS_REGION"
```

#### Modal OIDC

OIDC exchanges each Function's short-lived Modal identity token for AWS
credentials, so no AWS access key is stored in Modal. Register Modal's provider
once in the AWS account:

```bash
aws iam create-open-id-connect-provider \
  --url https://oidc.modal.com \
  --client-id-list oidc.modal.com
```

Create an IAM role with the S3 policy above and a trust policy scoped to the
Modal workspace and environment that run the cookbook:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<aws-account-id>:oidc-provider/oidc.modal.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {"oidc.modal.com:aud": "oidc.modal.com"},
        "StringLike": {
          "oidc.modal.com:sub": "modal:workspace_id:<workspace-id>:environment_name:<environment-name>:*"
        }
      }
    }
  ]
}
```

The workspace ID is available from `modal token info` or the Modal workspace
settings. The subject can be narrowed further by app or function name. See
Modal's [OIDC integration guide](https://modal.com/docs/guide/oidc-integration)
for the identity claims and trust-policy options.

Put the role and root in the Secret:

```bash
uv run --extra modal modal secret create -e "$MODAL_ENVIRONMENT" stitch-s3 \
  S3_ROOT=s3://your-bucket/stitch \
  AWS_ROLE_ARN=arn:aws:iam::<aws-account-id>:role/<role-name> \
  AWS_REGION="$AWS_REGION"
```

The cookbook exposes Modal's injected identity token through boto3's standard
web-identity credential chain. Do not copy `MODAL_IDENTITY_TOKEN` into the
Secret.

For either authentication method, select the backend before launching or
redeploying a run:

```bash
export STITCH_STORE_BACKEND=s3
export STITCH_S3_SECRET_NAME=stitch-s3
```

Trainers then write publications to node-local disk. One process per trainer
host uploads its files directly to the final immutable
`weight_vNNNNNN/` prefix, and rank 0 verifies the gathered receipts, index,
sizes, and checksums before conditionally advancing `latest`. Partially
uploaded versions remain invisible because replicas only follow `latest`.
S3's [strong consistency](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html#ConsistencyModel)
and [conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)
make the pointer the publication boundary; there is no shared publication
staging Volume or S3 copy step.

Rollout replicas download versions into their ephemeral local caches. The
experiment Volume remains available for logs and trainer checkpoints.

### Publish from a non-cookbook trainer

`S3Store` does not depend on Modal, Miles, or Slime. Install the optional boto3
dependency and let boto3 resolve credentials from environment variables, an
AWS profile, web identity, or the compute environment's IAM role:

```bash
pip install 'stitch[s3]'
```

Each version directory must be named `weight_vNNNNNN` and contain
`model.safetensors.index.json`. Its `metadata.version` must be `NNNNNN`, and
its `weight_map` must name every checkpoint or delta shard. A single-host
trainer can claim a new run and use the high-level publisher:

```python
from stitch.publish import claim_run, publish_version
from stitch.stores.s3 import S3Store

run_id = "run-2026-08-11"
store = S3Store(
    f"s3://my-bucket/stitch-runs/{run_id}",
    cache_dir=f"/tmp/stitch/{run_id}",
    run_id=run_id,
)

# Call once when establishing the run's base checkpoint.
claim_run(store, None, run_id, boot_version=0)

publish_version(
    store,
    None,
    "/local/updates/weight_v000001",
    run_id=run_id,
)
```

`publish_version` uploads and verifies the complete directory, advances
`latest` with an ETag precondition, and returns the published `VersionRef`.
Passing `None` for the pool is sufficient when replicas poll the store; a
custom `Pool` can be passed to provide an immediate wake-up hint.

For a distributed trainer, snapshot `latest` before upload, have one leader per
host upload only that host's local files, gather the receipts through the
trainer's communicator, and let rank 0 commit the pointer last:

```python
from stitch.types import VersionRef, decide_pointer_move

target = VersionRef(run_id, 1)

if global_rank == 0:
    expected = store.read_pointer()
    decide_pointer_move(expected, target)
else:
    expected = None
expected = broadcast_object(expected, src=0)  # your trainer's communicator

receipt = None
if is_host_leader:
    receipt = store.upload_version_files(target, local_version_dir)
receipts = gather_objects(receipt, dst=0)  # your trainer's communicator

if global_rank == 0:
    receipts = [receipt for receipt in receipts if receipt is not None]
    store.verify_version(target, receipts)
    store.compare_and_advance_pointer(expected, target)
```

All ranks should treat an upload, verification, or pointer conflict as a
failed publication and synchronize before continuing. The version prefix is
immutable: do not repair a failed publication by silently overwriting a
version that consumers may already have observed.
