# SGLang fork

Stitch overlays a small SGLang fork onto the matching upstream image. The fork
adds general, asynchronous checkpoint staging and correct complete-weight
loading for quantized rollout models.

## Pin

[`serving_image.py`](serving_image.py) is the executable source of truth:

```python
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.16"
SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "stitch-sglang-v0.5.16"
SGLANG_FORK_COMMIT = "a8e66a00f3023a600bb8d92851454d9fd760ccc3"
```

The branch is upstream v0.5.16 plus:

| Commit | Responsibility |
| --- | --- |
| `49031cb24c` | Configure fastsafetensors with or without GDS and honor post-load cache release. |
| `5f6e1e613f` | Materialize and verify complete targets on host-local disk. |
| `a7e20596ba` | Restore checkpoint-facing quantized layouts for complete weight loading. |
| `c867782f3e` | Build verified, rank-ready CPU weight images from canonical targets. |
| `a8e66a00f3` | Expose asynchronous disk/CPU staging and CPU-to-GPU commit APIs. |

The image and branch must use the same SGLang release because Stitch overlays
Python code onto the image’s existing CUDA and C++ extensions.

## API

`POST /stage_weight_update` prepares a target while inference continues:

```json
{
  "base_checkpoint_dir": "/prep/model/format",
  "checkpoint_source_dir": "/delta-bulletin/run",
  "local_checkpoint_dir": "/local-checkpoint",
  "target_version": 7,
  "destination": "disk"
}
```

- `base_checkpoint_dir` is the immutable v0 checkpoint. It defaults to the
  engine’s boot model path.
- `checkpoint_source_dir` contains `weight_vNNNNNN` publications. It is omitted
  when initializing version 0.
- `destination="disk"` requires `local_checkpoint_dir` and accepts FULL or
  DELTA targets.
- `destination="cpu"` does not use `local_checkpoint_dir`. Version 0 initializes
  the cache; later targets must be DELTAs.

Commit remains a separate operation:

- `POST /update_weights_from_disk` runs SGLang’s complete checkpoint loader.
- `POST /update_weights_from_cpu` copies already-prepared rank images into the
  existing CUDA storages.

The separation is the pause boundary: staging may overlap rollout generation,
while commit is the short operation coordinated by Stitch.

## Disk destination

Disk mode is the general lower-RAM path. SGLang keeps one mutable host-local
checkpoint, seeds it from the immutable base or newest FULL anchor, applies the
required delta chain, verifies target checksums, and publishes its applied
version only after the files are durable. A backward target automatically
reseeds from an immutable anchor.

The rollout engine initially loads v0 directly from `base_checkpoint_dir`.
Stitch initializes the mutable local checkpoint in the background after v0 is
serving, so initial rollout readiness is not delayed by a second model-sized
copy. Local storage must hold the mutable checkpoint plus filesystem headroom;
the immutable base remains in its configured source.

The commit RPC still reads and transforms the complete target checkpoint. On
each commit, Stitch forwards the load format selected for the initial server
load. On hosts without GDS, recipes select fastsafetensors’ supported no-GDS
mode:

```python
"--load-format": "fastsafetensors",
"--model-loader-extra-config": '{"enable_gds":false}',
"--weight-loader-drop-cache-after-load": "",
```

## CPU destination

CPU mode trades host RAM for the shortest commit:

1. After v0 begins serving, SGLang allocates one canonical checkpoint snapshot
   per host and one complete rank-ready image per local TP rank.
2. It builds v0 through the model’s ordinary weight loader and quantization
   hooks and verifies that the prepared runtime storages match the active model.
3. For every delta, it reconstructs and checksums the canonical target, then
   builds every next rank image while inference continues.
4. `/update_weights_from_cpu` performs distributed preflight and copies the
   complete images into the existing CUDA storages without replacing storage
   pointers.

Stitch starts step 1 in the background. A published delta waits for that same
initialization task; it does not start a second cache build. An initialization
or staging failure is reported and the GPU remains on its prior weights.

CPU mode is delta-only. It rejects FULL publications and cannot reset a patched
live replica to another run’s v0; the controller must use disk mode or replace
that replica. This keeps a single canonical snapshot rather than retaining a
second rollback-sized CPU checkpoint.

Enable it explicitly:

```python
"--enable-cpu-weight-cache": "",
"--cpu-weight-cache-max-compile-group-gb": "8",
```

The group bound limits transient loader work; it does not tune correctness or
assume a model architecture. An indivisible module larger than the requested
bound remains intact and is reported.

Persistent host RAM is approximately:

```text
one canonical checkpoint per host
+ one rank-local runtime image per local TP rank
```

All runtime storages are prepared and committed. Element-wise sparsity only
reduces the compressed delta transport and XOR work.

## Correctness

Delta application and checksum verification happen in canonical checkpoint
tensor space, before TP sharding and runtime-layout conversion. Missing bytes,
invalid lineage, size mismatches, and checksum mismatches fail staging without
mutating the live model.

Verified tensors then pass through the same model loader and quantization hooks
used for initial loading. The implementation does not special-case changed
tensor sets and applies to dense element-wise deltas. Checkpoint layout support is
selected by each quantization method; FP8, ModelOpt NVFP4, and Blackwell MXFP4
therefore use their native SGLang transforms rather than checkpoint-specific
workarounds.

Every TP rank must pass preflight before commit starts. A rank-local copy failure
after that point is fatal because continuing with mixed rank versions would be
incorrect. Speculative draft models remain rejected until target and draft
weights can be committed atomically.

## Re-porting

For a new SGLang release:

1. create `stitch-sglang-vX` from the exact upstream tag;
2. omit the fastsafetensors commit if upstream already contains PR #31859;
3. reapply the remaining responsibilities as separate commits;
4. audit the release’s loader, quantization, scheduler, process-group, and
   CUDA-graph primitives and delete fork code superseded upstream;
5. run SGLang’s own pre-commit hooks and focused unit tests;
6. validate generation before, during, and after one complete delta update on
   FP8 and ModelOpt NVFP4, and validate MXFP4 transforms on Blackwell; and
7. update the image, branch, immutable commit, and this file.

Upstream fastsafetensors reference:
<https://github.com/sgl-project/sglang/pull/31859>.
