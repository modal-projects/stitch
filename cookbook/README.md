# Cookbook

These recipes connect Miles training to elastic SGLang rollout pools through
Stitch. Agentic, quantized training is the main supported workflow. The small
math recipe teaches the same publication and recovery path; standalone serving
lets an external trainer supply the updates.

## Supported scenarios

| Recipe | Choose it for | Trainer | Warm rollout fleet |
| --- | --- | --- | --- |
| [`qwen3_6_35b_a3b_nvfp4`](miles_disagg/configs/qwen3_6_35b_a3b_nvfp4.py) | Agentic training with humans& NVFP4 routed experts | 2 × 8 B300 | 16 × 1 B200 or better |
| [`glm5_3_nvfp4`](miles_disagg/configs/glm5_3_nvfp4.py) | Large-scale agentic training with a speculative draft | 32 × 8 B300 | 16 × 4 B300 |
| [`qwen3_4b_math`](miles_disagg/configs/qwen3_4b_math.py) | A small BF16 GRPO starter with evaluation and recovery | 1 × 2 H200 | 1 × 1 H200 |
| [`glm5_3_fp8`](standalone/configs/glm5_3_fp8.py) | A standalone policy-versioned pool for an external trainer | External | 2 × 8 B300 |

Each module pins the source revision and declares its prepared checkpoint
paths, hardware, parallelism, concurrency, and runtime settings. Training uses
the pinned [Miles runtime](miles_disagg/MILES_FORK.md); all pools use the pinned
[SGLang runtime](common/SGLANG_FORK.md). GPU counts above are deployment floors,
not preparation-job sizes. Pools leave `max_containers` unset so Modal can
autoscale and bring up replacement containers during blue-green deployments.

### Agentic NVFP4 training

Both agentic recipes run fully asynchronous GRPO on SWE-bench Pro. They use the
humans& row-scaled NVFP4 recipe with dequantized backward, four-over-six scaling,
MSE candidate selection, and matching preparation/training/serving settings.
Changing the quantizer changes the canonical weight bytes: prepare a new
checkpoint lineage before training with different precision settings.

Qwen3.6 uses 40 hybrid-attention layers. Routed experts use NVFP4; shared experts,
attention, and the last six layers remain BF16. Preparation exposes the fused
source experts individually to match the trainer's exported checkpoint layout.
It runs 500 batches of 32 prompts × 8 samples, with 256 concurrent agent episodes
and at most two completed batches buffered. Checkpoints are saved every 20
batches and at the end.

GLM-5.3 uses the same architecture preset as GLM-5.2. Its first three dense
layers, last 12 layers, and shared experts remain BF16. Training uses
TP4/PP4/CP8/EP32 over 256 GPUs, with 64 prompts × 8 samples per batch and 256
concurrent episodes. Checkpoints are saved every 10 batches and at the end.
Its BF16 training source is `zai-org/GLM-5.3-BF16`; the repository named
`zai-org/GLM-5.3` contains the FP8 release used by standalone serving.

The GLM trainer recipe requires its external GLM-5.3 NVFP4 DFlash checkpoint.
The module declares the Volume, source environment, and checkpoint path. Make
that artifact available in your workspace and set `modal.draft_volume`,
`modal.draft_volume_env`, and `DFLASH_CHECKPOINT_PATH` accordingly. The draft
remains fixed while the target model trains; acceptance can change as the
policy evolves. The standalone FP8 recipe has no draft dependency.

Both agentic recipes allow 64K total tokens, 8,192 tokens per response, 256 agent
steps, and two hours per episode. They use the published SWE-bench Pro test tasks
as training prompts; the resulting rewards are training metrics, not held-out
benchmark scores.

### Small math starter

Qwen3-4B runs synchronous GRPO on a pinned GSM8K training split with exact-version
requests and zero allowed weight lag. It trains for 120 batches, saving paired
Megatron/Hugging Face checkpoints and evaluating on the held-out test split every
20 batches. The same recipe serves short validation and longer training runs:
change the duration and save/evaluation intervals instead of copying the module.

## Prepare and launch

Choose a recipe and a Modal environment:

```bash
export EXPERIMENT_CONFIG=qwen3_6_35b_a3b_nvfp4
export MODAL_ENVIRONMENT=your_environment
modal profile current
```

Create `huggingface-secret` with `HF_TOKEN` in that environment. The agentic
recipes also enable W&B and require `wandb-secret` with `WANDB_API_KEY`:

```bash
uv run --extra modal modal secret create -e "$MODAL_ENVIRONMENT" \
  huggingface-secret HF_TOKEN="$HF_TOKEN"
uv run --extra modal modal secret create -e "$MODAL_ENVIRONMENT" \
  wandb-secret WANDB_API_KEY="$WANDB_API_KEY"
```

Prepare Miles inputs in order. These commands run in a separate app and do not
start the rollout fleet. Dataset preparation can run independently.

```bash
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.miles_disagg.prep_app::prepare_checkpoints
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.miles_disagg.prep_app::prepare_torch_dist
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.miles_disagg.prep_app::prepare_dataset
uv run --extra modal python -m cookbook.miles_disagg.launch
```

Preparation records the source revision, conversion settings, and completed
shards. Matching artifacts are reused. An incomplete artifact or a path prepared
under different settings fails with its path; inspect it before removal, or
choose a new prepared path. Never point a new model or precision recipe at an
unrelated prepared checkpoint merely because the architecture matches.
Preparation also requires an immutable Miles revision: set `MILES_REPO_REF` to a
full commit hash when using a custom fork.

For standalone serving, download the published FP8 checkpoint and launch the
pool:

```bash
export EXPERIMENT_CONFIG=glm5_3_fp8
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m cookbook.standalone.prep_app::download_base
uv run --extra modal python -m cookbook.standalone.launch
```

Launchers create an isolated run ID, deploy its pool, and wait for the gateway.
Miles then starts training; standalone claims the run's initial version at v0
and waits for an external publisher. Set `RUN_ID` only when deliberately
reusing a run identity. Every rollout server enables Modal's KV-aware routing;
clients address it through `ModalFlashPool(app_name, "Server")`.

Stop the deployed app after a run finishes to release its rollout fleet:

```bash
uv run --extra modal modal app stop -e "$MODAL_ENVIRONMENT" "$APP_NAME"
```

## Checkpoints and recovery

The Miles trainer retries under Modal. Each attempt finds the newest complete
Megatron/Hugging Face checkpoint pair that was published, restores optimizer and
RNG state, and resumes the existing run. Replicas serving abandoned versions
exit; replacements load the restored checkpoint and catch up from there.

Recovery requires the Modal Volume store, `update_weights_interval = 1`, a
positive `save_interval`, `save_hf`, and optimizer/RNG checkpointing. All three
training recipes provide these settings. Recovery requires a complete, published
checkpoint pair from an iteration greater than zero: Miles also uses iteration
zero for a fresh actor, so a checkpoint saved there is not resumable. S3
publication is supported, but cookbook trainer recovery currently requires
Modal Volumes.

For a run past its retry budget, start a successor trainer with the same recipe
and environment:

```bash
uv run --extra modal python -m cookbook.miles_disagg.launch \
  --resume-from existing_run_id
```

This cancels the recorded trainer call and resumes against its deployed pool.
If the pool was stopped, deploy it under the existing identity first:

```bash
RUN_ID=existing_run_id uv run --extra modal modal deploy \
  -e "$MODAL_ENVIRONMENT" -m cookbook.miles_disagg.app
```

Complete Hugging Face exports also accelerate elastic startup: new replicas
load the newest complete run checkpoint no newer than `latest`, then apply the
remaining deltas. Before the first save, they start from the prepared base.

## Observe and scale a run

Use the app name printed by the launcher:

```bash
uv run --extra modal modal app logs -f -e "$MODAL_ENVIRONMENT" "$APP_NAME"
uv run --extra modal python -m cookbook.common.smoke \
  --app-name "$APP_NAME" --model-name "$MODEL_NAME" --weight-version 10
```

`MODEL_NAME` is the recipe's served model name; absent an explicit
`--served-model-name`, it is the prepared checkpoint path. The smoke check
verifies gateway generation and the sampled replicas' reported versions.

`rollout_min_containers` sets the warm fleet floor, and `rollout_target_inputs`
sets the autoscaling target per replica. `--max-running-requests` and `--max-queued-requests`
control engine admission and backpressure. Change these in the recipe and
redeploy with the existing run ID to keep its checkpoint lineage:

```bash
RUN_ID=existing_run_id uv run --extra modal modal deploy \
  -e "$MODAL_ENVIRONMENT" -m cookbook.miles_disagg.app
```

## Storage and weight updates

Cookbook-owned volumes use Modal Volume v2. External draft checkpoints can use
an existing Volume of either version:

| Volume | Mount | Contents |
| --- | --- | --- |
| `huggingface-cache` | `/root/.cache/huggingface` | Pinned source downloads |
| `miles-checkpoints` | `/checkpoints` | Immutable prepared model layouts |
| `miles-data` | `/data` | Pinned prompts and task assets |
| Recipe experiment Volume | `/stitch` | Run checkpoints, publications, and logs |
| `sglang-cache` | `/root/.cache/sglang` | Compiled kernels |
| Configured draft Volume | `/draft` | Fixed speculative draft, when required |

Each experiment Volume keeps runs separate:

```text
/stitch/<run-id>/
├── latest
├── updates/weight_vNNNNNN/
├── checkpoints/
├── hf_checkpoints/weight_vNNNNNN/.complete
└── logs/train-from-vNNNNNN-<UTC>-<id>.log
```

The trainer owns updates and saved checkpoints; Stitch owns the publication
pointer. Standalone publishers claim v0 at launch and publish subsequent
versions through the same store protocol.

All maintained recipes use CPU staging: the engine reconstructs and checksums
the target checkpoint while serving, then pauses briefly to activate prepared
rank images. The alternative disk mode reconstructs a complete local checkpoint
and reloads it during activation. CPU mode needs room for canonical weights and
rank images; the [SGLang runtime contract](common/SGLANG_FORK.md) describes
memory and local-storage choices.

## Profile a weight update

[Delta-update profiling](../tools/profiling/README.md) has a broader architecture
and checkpoint-format catalog than the maintained recipes, including Kimi K3
MXFP4, Kimi K2.6 NVFP4, GLM-4.5-Air FP8, and GLM-5.2 FP8/NVFP4. Those scripts
retain their own model settings without implying a maintained training recipe.

After preparing a maintained GLM checkpoint, run its profiler in the same Modal
environment:

```bash
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  tools/profiling/glm5_3_fp8_delta_weight_update.py \
  --update-mode cpu --canonical-storage memory
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  tools/profiling/glm5_3_nvfp4_delta_weight_update.py \
  --update-mode cpu --canonical-storage memory
```

Each uses its recipe's serving image, model identity, server arguments, and
hardware. It constructs a checksummed synthetic XOR update, generates during
staging, activates the target, and checks version attribution and repeated
finite logprobs against an explicit tolerance. The default requires exact
repeats. These are one-replica diagnostics; they do not establish trainer export
correctness or fleet convergence. Results belong with the measured runtime and
input identity in the change's validation evidence.

CPU staging can also keep canonical weights on host-local NVMe with
`--canonical-storage disk`. To measure full disk reloads, pass
`--update-mode disk`.

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
