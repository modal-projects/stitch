# Cookbook

The cookbook contains runnable Modal deployments that connect Miles
trainers to elastic SGLang rollout pools through Stitch. Recipes define the
model, trainer, rollout fleet, data, and weight-update policy; the shared
infrastructure handles preparation, isolated runs, and pool lifecycle.
`standalone` deploys the same rollout pool without a trainer: an external
trainer or harness publishes weight updates through the configured checkpoint
store and sends rollout traffic directly to the pool's Modal Server. Its launcher
claims the run's boot pointer at v0 before the pool enters rotation.

Every rollout `Server` enables `experimental_options={"kv_aware_routing": True}`.
Modal handles KV-aware routing. Rollout clients use
`ModalFlashPool(app_name, "Server")`.

## Reference recipes

| Recipe | Scenario |
| --- | --- |
| [`qwen3_6_35b_a3b_nvfp4`](miles_disagg/configs/qwen3_6_35b_a3b_nvfp4.py) | Agentic SWE-bench Pro training with humans& NVFP4 routed experts |
| [`glm5_3_nvfp4`](miles_disagg/configs/glm5_3_nvfp4.py) | Large-scale agentic NVFP4 training with an external speculative draft |
| [`qwen3_4b_math`](miles_disagg/configs/qwen3_4b_math.py) | Small synchronous BF16 GRPO starter on GSM8K |
| [`glm5_3_fp8`](standalone/configs/glm5_3_fp8.py) | Standalone FP8 rollout pool for an external trainer |

The [weight-update profiles](../tools/README.md#weight-update-validation) cover
additional architectures independently of this recipe catalog.

## Common workflow

### 1. Select a recipe

Set `EXPERIMENT_CONFIG` to a module in `miles_disagg/configs`,
or `standalone/configs`:

```bash
export EXPERIMENT_CONFIG=qwen3_6_35b_a3b_nvfp4
export MODAL_ENVIRONMENT=your_environment
uv run --extra modal modal profile current
```

The selected config is the authority for model revisions, Volume names,
hardware, scaling, and trainer arguments.

### 2. Create credentials

Create the secrets required by the selected recipe once in its Modal
environment:

```bash
export HF_TOKEN=your_hugging_face_token
export WANDB_API_KEY=your_wandb_api_key

uv run --extra modal modal secret create -e "$MODAL_ENVIRONMENT" \
  huggingface-secret HF_TOKEN="$HF_TOKEN"
uv run --extra modal modal secret create -e "$MODAL_ENVIRONMENT" \
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
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.miles_disagg.prep_app::prepare_checkpoints
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.miles_disagg.prep_app::prepare_torch_dist
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.miles_disagg.prep_app::prepare_dataset
```

Standalone recipes serve a quantized release checkpoint as published, so
download is the whole preparation:

```bash
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.standalone.prep_app::download_base
```

Preparation is idempotent. A complete artifact is reused; an incomplete one
fails rather than becoming a launch input. Model preparation must finish before
dependent format conversion. Dataset preparation is independent.

Miles preparation records the source revision, converter settings, and completed
shards. Reuse requires matching records; for changed inputs, choose a new
checkpoint path. Custom converters require a full `MILES_REPO_REF` commit hash;
preparation does not support a mutable `MILES_LOCAL_DIR` overlay.

### 5. Launch an isolated run

Use the launcher for the selected trainer:

```bash
# Miles
uv run --extra modal python -m cookbook.miles_disagg.launch

# Standalone pool (no trainer)
uv run --extra modal python -m cookbook.standalone.launch
```

The launcher creates an eight-character run ID unless `RUN_ID` is set explicitly,
deploys a run-scoped rollout pool, and waits for its gateway. Miles then
start the trainer; standalone claims `latest` at v0 and stops after the pool is
ready. Repeating the command creates a separate run and checkpoint lineage.

#### Resume a Miles run

Recovery is automatic: the Miles trainer runs under Modal retries, and every
attempt re-derives its resume state from the run volume — the newest saved
Megatron checkpoint whose Hugging Face export is complete and published. A
preempted or crashed trainer resumes on its own, without a launcher attached
and without redeploying the pool; replicas serving abandoned versions exit and
their replacements boot from the restored checkpoint. If the checkpoint is
v120, the run resumes at v120 and publishes a replacement v121 next. Resume
requires the Modal Volume store, `update_weights_interval = 1`,
`save_interval`, `save_hf`, and optimizer/RNG checkpointing — a fresh launch
warns when its config is not resumable, and a fresh run becomes resumable
after its first complete Megatron/Hugging Face checkpoint pair from an iteration
greater than zero. Miles treats iteration zero as a fresh actor.

For a run past its retry budget, or a manual takeover, spawn a successor
trainer with the same recipe and Modal environment as the source run:

```bash
export EXPERIMENT_CONFIG=your_config_name
export MODAL_ENVIRONMENT=your_environment

uv run --extra modal python -m cookbook.miles_disagg.launch \
  --resume-from existing_run_id
```

This cancels the run's recorded trainer call and spawns a new one against the
deployed pool, which resumes like any retry. The pool itself is never
redeployed here; if it was stopped (or its infra must change first), deploy it
explicitly under the same run ID and then resume:

```bash
EXPERIMENT_CONFIG=your_config_name RUN_ID=existing_run_id \
  uv run --extra modal modal deploy -e "$MODAL_ENVIRONMENT" -m cookbook.miles_disagg.app
```

With the Modal Volume store, complete Hugging Face checkpoints also accelerate
elastic rollout startup. When a Miles recipe saves checkpoints and updates
weights every step, a new replica loads the current run's newest complete
checkpoint no newer than `latest`, then applies only the remaining deltas.
Before the first save, it catches up from the run's configured boot checkpoint.

### 6. Inspect or change a live run

Follow logs using the app name printed by the launcher:

```bash
export APP_NAME=your_app_name
uv run --extra modal modal app logs -e "$MODAL_ENVIRONMENT" -f "$APP_NAME"
```

Verify that the gateway and every live replica serve an expected version:

```bash
export MODEL_NAME=/checkpoints/your_model

uv run --extra modal python -m cookbook.common.smoke \
  --app-name "$APP_NAME" \
  --model-name "$MODEL_NAME" \
  --weight-version 10
```

Rollout capacity is controlled by `rollout_min_containers` and
`rollout_target_inputs`. Leave `rollout_max_containers` unset to allow autoscaling
and replacement containers during blue-green deployment. Engine concurrency and
backpressure are controlled by `--max-running-requests` and
`--max-queued-requests` in the recipe.

Configure the sidecar's HTTP connection pool with `proxy_max_connections` (default
100) and `proxy_max_keepalive_connections` (default 20 idle connections) in
`serve_startup`, or with `--proxy-max-connections` / `--proxy-max-keepalive-connections`.

After changing fleet or SGLang settings, redeploy the active run with the same
experiment and run ID:

```bash
EXPERIMENT_CONFIG=your_config_name RUN_ID=your_run_id \
  uv run --extra modal modal deploy -e "$MODAL_ENVIRONMENT" -m cookbook.miles_disagg.app
```

Modal rolls replicas to the new configuration without changing the run's
checkpoint lineage.

## Qwen3.6 SWE-bench Pro training

[`qwen3_6_35b_a3b_nvfp4.py`](miles_disagg/configs/qwen3_6_35b_a3b_nvfp4.py)
runs fully asynchronous GRPO on SWE-bench Pro, using a pinned
`Qwen/Qwen3.6-35B-A3B` source and the humans& NVFP4 recipe for routed experts.

| Component | Configuration |
| --- | --- |
| Trainer | 2 nodes × 8 B300 GPUs |
| Rollouts | 16 warm replicas × 1 B200-or-better GPU; autoscale without a cap |
| Training | 500 batches; 32 prompts × 8 samples = 256 trajectories per batch |
| Checkpoints | Every 20 batches and at the end of training |
| Concurrency | 256 agent episodes; up to two completed batches buffered |
| Episode limits | 256 agent steps, 2 hours, 64K total tokens |
| Per-response limit | 8,192 tokens |

Preparation exposes fused experts individually without changing their weight
bytes, matching Miles' checkpoint-delta export layout. It also prepares the
NVFP4 serving checkpoint; trainer and serving quantization settings must match.
Shared experts, attention, and the last six layers remain BF16.

After creating the shared secrets, run each preparation command to completion
before launching:

```bash
export EXPERIMENT_CONFIG=qwen3_6_35b_a3b_nvfp4
export MODAL_ENVIRONMENT=your_environment

uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.miles_disagg.prep_app::prepare_checkpoints
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.miles_disagg.prep_app::prepare_torch_dist
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.miles_disagg.prep_app::prepare_dataset
uv run --extra modal python -m cookbook.miles_disagg.launch
```

The trainer stops after 500 batches. Stop its deployed app when finished
to release the rollout fleet:

```bash
uv run --extra modal modal app stop -e "$MODAL_ENVIRONMENT" "$APP_NAME"
```

## GLM-4.7 Flash example

This retired recipe provides historical end-to-end training measurements. Its
[archived configuration](https://github.com/modal-projects/stitch/blob/0d1f769a725cd1133fd36d812e7a093c7b71af87/cookbook/miles_disagg/configs/glm47_flash_swebench_pro.py)
is available for reference; use the maintained recipes above for new runs.

| Component | Configuration |
| --- | --- |
| Trainer | 4 nodes × 8 H200 GPUs |
| Rollout | 48 replicas × 1 H200 GPU |
| Model | `zai-org/GLM-4.7-Flash`, pinned BF16 revision |
| Dataset | SWE-bench Pro, including task environments and verifiers |
| Weight sync | Checksummed XOR deltas with CPU preparation and in-place activation |

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

| Recipe settings | Prepared state | During engine pause | Use when |
| --- | --- | --- | --- |
| `disk` + `LOCAL_CHECKPOINT_PATH` | Complete checkpoint on local storage | Reload from disk | Host RAM is constrained or the trainer publishes full checkpoints |
| `cpu` + `LOCAL_CHECKPOINT_PATH=None` | Canonical checkpoint and rank-ready images in RAM | Copy the images to GPU | Host RAM can hold both representations and the shortest preparation is preferred |
| `cpu` + `LOCAL_CHECKPOINT_PATH` | Canonical checkpoint on local NVMe; rank-ready images in RAM | Copy the images to GPU | Host RAM can hold the rank images but not both representations |

Both modes reconstruct and checksum the complete target in canonical checkpoint
space. CPU mode accepts deltas only and requires a new replica for a new
lineage. Disk mode accepts full checkpoints and deltas and can reset a live
replica.

The bundled recipes currently use CPU staging with the canonical checkpoint in
RAM. The profiling scripts exercise all three storage layouts. Memory sizing
and SGLang details are in [`SGLANG_FORK.md`](common/SGLANG_FORK.md).

## Persistent storage

Prepared inputs, caches, logs, and trainer checkpoints use Modal Volume v2.
Run publications use either the experiment Volume or S3, according to
`STITCH_STORE_BACKEND`:

| Volume | Mount | Contents |
| --- | --- | --- |
| `huggingface-cache` | `/root/.cache/huggingface` | Pinned source-model downloads |
| `miles-checkpoints` | `/checkpoints` | Immutable prepared model layouts |
| `miles-data` | `/data` | Pinned datasets |
| `stitch-<framework>-<model>` | `/stitch` | Run-scoped checkpoints and logs; publications when using the Volume backend |
| `sglang-cache` | `/root/.cache/sglang` | Compiled SGLang kernels |
| `miles-kernel-cache` | `/kernel-cache` | Compiled Triton and TorchInductor kernels for the Miles trainer |
| Configured draft Volume | `/draft` | Optional external speculative draft |

Prepared model layouts have stable paths. For example:

```text
/checkpoints/
├── qwen3-4b-1cfa9a72-bf16/
└── qwen3-4b-1cfa9a72-torch-dist-tp1/
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
    └── logs/train-from-vNNNNNN-<UTC>-<id>.log
```

The training framework owns `updates/` and the saved checkpoints; Stitch owns
`latest`. For standalone pools, the launcher establishes `latest` at v0 and the
external publisher owns subsequent updates. Resume keeps the same run ID,
restores `latest` to the checkpoint, and replaces the abandoned update suffix as
training continues. Each finished trainer attempt writes a separate log; use
Modal app logs while it is running.

Datasets are independent of models and runs:

```text
/data/
└── <dataset>/
    ├── manifest.json
    ├── <trainer-input>
    └── <dataset-specific-assets>/
```

The Miles trainer points `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` at
`/kernel-cache/<sm_cc>/torch-<ver>-triton-<ver>/{triton,inductor}`, so compiled
kernels survive across attempts and runs on the same GPU arch and toolchain.
Set `modal.kernel_cache_volume` to use another Volume, or `None` to disable.

## External speculative drafts

Set `modal.draft_volume` and, when needed, `modal.draft_volume_env`. The server
mounts the volume at `/draft`; set `--speculative-draft-model-path` to the
checkpoint below it. External draft Volumes may use either Volume version.
The `glm5_3_nvfp4` recipe requires its declared draft artifact to be available in
that Volume and environment before launch.

Draft weights remain fixed while target weights update. Their acceptance rate
may change as the target evolves. Updating both atomically requires restarting
the replica. Bundled MTP heads do not need a separate volume.

## Profile an update

The model profilers prepare their pinned base checkpoint and synthetic delta,
then run with `--update-mode disk|cpu`. CPU runs also select
`--canonical-storage memory|disk`; `disk` uses host-local NVMe.

```bash
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" -d \
  tools/weight_update/profiles/glm45_air_fp8.py \
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
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" -d \
  tools/weight_update/profiles/kimi_k3_mxfp4.py \
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

`S3Store` does not depend on Modal or Miles. Install the optional boto3
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

For a distributed trainer, `stitch.publisher.Publisher` runs the same protocol
the cookbook's trainers use — rank 0 snapshots `latest`, one leader per host
uploads only that host's local files, rank 0 verifies the gathered receipts and
commits the pointer last. Provide your trainer's communicator as a
`TrainerComms` (rank identity, an object gather, and one host leader election);
the defaults are the single-process case:

```python
from stitch.publisher import Publisher, TrainerComms


class TorchComms(TrainerComms):
    def rank(self):
        import torch.distributed as dist

        return dist.get_rank() if dist.is_initialized() else None

    def all_gather_object(self, value):
        import torch.distributed as dist

        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, value)
        return values

    def is_host_leader(self):
        return torch.distributed.get_rank() % ranks_per_host == 0


publisher = Publisher(store, None, run_id=run_id, comms=TorchComms())
publisher.claim(boot_version=0)  # once, when establishing the run's base
publisher.publish("/local/updates/weight_v000001")
```

`Publisher.publish` also dispatches on the backend: a shared mounted store is
committed by each host leader before rank 0 publishes from the refreshed view,
which is the flow the cookbook's Miles hooks use with Modal Volumes.

All ranks should treat an upload, verification, or pointer conflict as a
failed publication and synchronize before continuing. The version prefix is
immutable: do not repair a failed publication by silently overwriting a
version that consumers may already have observed.
