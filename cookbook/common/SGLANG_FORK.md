# SGLang fork

Stitch overlays a small SGLang fork onto the matching upstream image. The fork
adds general, asynchronous checkpoint staging and correct complete-weight
loading for quantized rollout models.

## Pin

[`serving_image.py`](serving_image.py) is the executable source of truth for the
default runtime:

```python
DEFAULT_SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.17",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-v0.5.17",
    commit="d050d06437d96196fc68d5b4e5c246408790d537",
)
```

The branch is upstream v0.5.17 plus four independently reviewable layers:

| Layer | Responsibility |
| --- | --- |
| Reload lifecycle | Restore checkpoint-facing layouts, run each quantization method's native loader and post-load hooks, and fail closed if a partially mutated model cannot be rolled back. |
| Verified materialization | Apply and fold complete XOR delta lineages in canonical checkpoint space, verify the published checksum, and durably materialize disk targets. |
| CPU staging | Build bounded rank-ready host images while serving, optionally keep the canonical checkpoint on local NVMe, then commit every runtime storage in place. |
| Serving correctness | Preserve routed-expert state and sampling masks across data-parallel and speculative paths, classify client cancellations, and surface scheduler-process failures. |

The branch history keeps these physical responsibilities in separate commits;
the immutable pin above is the executable definition of the stack.

The image and immutable source pin stay together so the Python overlay remains
ABI-compatible with the image's CUDA and C++ extensions. SGLang v0.5.17 includes
Kimi K3, so all cookbook recipes now use this one runtime line. The fork's MXFP4
staging path transforms runtime layouts on GPU before caching rank-ready host
images.

## API

`POST /stage_weight_update` prepares a target without changing live weights:

```json
{
  "base_checkpoint_dir": "/checkpoints/<artifact-id>",
  "base_version": 0,
  "checkpoint_source_dir": "/stitch/<run-id>/updates",
  "local_checkpoint_dir": "/local-checkpoint",
  "target_version": 7,
  "destination": "disk"
}
```

- `base_checkpoint_dir` is the immutable checkpoint already loaded by the
  engine. `base_version` records its logical version and defaults to 0.
- `checkpoint_source_dir` contains `weight_vNNNNNN` publications. It is omitted
  when initializing the base version.
- `destination="disk"` requires `local_checkpoint_dir` and accepts FULL or
  DELTA targets.
- `destination="cpu"` does not use `local_checkpoint_dir`. The base version
  initializes the cache; later targets must be DELTAs.

Commit remains a separate operation:

- `POST /update_weights_from_disk` runs SGLang’s complete checkpoint loader.
- `POST /update_weights_from_cpu` copies already-prepared rank images into the
  existing target-model CUDA storages. A speculative draft model remains fixed;
  target verification preserves generation correctness while its acceptance
  rate may change as the target evolves.

The separation is the pause boundary: after startup, staging may overlap rollout
generation, while commit is the short operation coordinated by Stitch.

## Disk destination

Disk mode is the general lower-RAM path. SGLang keeps one mutable host-local
checkpoint, seeds it from the immutable base or newest FULL anchor, applies the
required delta chain, verifies target checksums, and publishes its applied
version only after the files are durable. A backward target automatically
reseeds from an immutable anchor.

When catch-up spans consecutive XOR deltas, SGLang validates the complete
published lineage first, then range-streams the compressed fragments while
reading and writing each changed target tensor once. The folded representation
is ephemeral: no aggregate delta or additional checkpoint is persisted. The
final published target checksum remains the commit boundary.

The rollout engine initially loads its boot checkpoint directly from
`base_checkpoint_dir`. Its logical version is normally v0 and may be a saved
version for a resumed run. Stitch initializes the mutable local checkpoint
before the replica enters rotation, then reconciles it to the visible version.
Local storage must hold the mutable checkpoint plus filesystem headroom; the
immutable boot checkpoint remains in its configured source.

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

CPU mode keeps rank-ready images in RAM for the shortest commit:

1. After loading the boot checkpoint, SGLang allocates one complete rank-ready
   image per local TP rank and either caches one canonical checkpoint per host in
   RAM or materializes it on host-local storage. It also builds the model's
   native loader views once against each rank image.
2. When the base is the boot checkpoint, it captures the already-realized active
   runtime storages into the rank images instead of repeating a model-sized
   load. A different base goes through the model's ordinary loader and
   quantization hooks and must match the requested checkpoint before use.
3. For every delta lineage, it reconstructs and checksums the canonical target,
   then builds every next rank image while inference continues. The in-memory
   path streams deltas through a bounded work budget; the storage-backed path
   reuses the transactional disk materializer.
4. `/update_weights_from_cpu` performs distributed preflight and copies the
   complete images into the existing CUDA storages without replacing storage
   pointers.

Stitch completes step 1 before the replica enters rotation. An initialization
failure replaces the replica; a later staging failure is reported and the GPU
remains on its prior weights. Registering a model-sized rank image as CUDA host
memory and capturing active GPU weights are one-time startup work.

CPU mode is delta-only. It rejects FULL publications and cannot reset a patched
live replica to another run; the controller must use disk mode or replace that
replica. This keeps a single canonical snapshot rather than retaining a second
rollback-sized CPU checkpoint. Compressed deltas are not retained in a
lineage-sized CPU arena after reconstruction.

Enable it explicitly:

```python
"--enable-cpu-weight-cache": "",
"--cpu-weight-cache-max-compile-group-gb": "8",
```

By default, both the canonical checkpoint and rank images remain in RAM. Keep
only the rank images in RAM by placing the canonical checkpoint on writable
host-local storage:

```python
"--cpu-weight-cache-canonical-checkpoint-dir": "/local-checkpoint/canonical",
```

Use local NVMe rather than a network or shared filesystem. This path pays a
complete checkpoint copy during initialization and adds NVMe read/write work
during preparation; it does not change the CPU-to-GPU engine pause. Set
`--weight-loader-drop-cache-after-load` to release clean checkpoint pages after
every local TP rank finishes compiling; otherwise the kernel may retain those
reclaimable pages for later reads.

The group bound limits CUDA staging work; it does not tune correctness or assume
a model architecture. An indivisible module larger than the requested bound
remains intact and is reported. Loader views and their checkpoint-layout state
remain in RAM so each update can reuse the native loader graph. Temporary GPU
staging clones are reclaimed at their group boundary and cannot accumulate a
second model-sized device copy.

With the default in-memory canonical checkpoint, persistent host RAM is:

```text
one canonical checkpoint per host
+ one rank-local runtime image per local TP rank
+ native-loader state per local TP rank
```

The canonical checkpoint is interleaved across the host's allowed NUMA nodes so
it cannot exhaust one GPU-local node while capacity remains elsewhere. Rank
images remain GPU-local because they are the source of the latency-sensitive
CPU-to-GPU commit.

With a storage-backed canonical checkpoint, persistent host RAM is the rank
images and native-loader state; local storage holds one canonical checkpoint.
File-cache pages used during preparation are reclaimable.

Measured component sizes are:

| Model | TP | Canonical checkpoint per host | Rank image | Loader state per rank |
| --- | ---: | ---: | ---: | ---: |
| GLM-4.5-Air FP8 | 4 | 112.6 GB | 27.2 GB × 4 | 1.64 GB |
| Kimi K2.6 NVFP4 | 4 | about 595 GB | about 151 GB × 4 | 4.09 GB |
| GLM-5.2 mixed NVFP4/BF16 | 4 | 617.6 GB | 179.3 GB × 4 | 20.94 GB |
| GLM-5.2 FP8 | 4 | 755.6 GB | 189.4 GB × 4 | 20.94 GB |
| Kimi K3 MXFP4 | 8 | 1.561 TB | 207.5 GB × 8 | 0.14 GB maximum |

Allow additional memory for the engine process, delta decoding, and bounded
loader staging. The supplied GLM-4.5 recipe requests `(512 GiB, 2 TiB)`;
GLM-5.2, Kimi K2.6, and Kimi K3 request `(1 TiB, 3 TiB)`, expressed as
`(request, limit)`. GLM-5.2 FP8 reached 1.63 TB with both the canonical and rank
images in RAM, and 1.19 TB with the canonical checkpoint on NVMe.

All runtime storages are prepared and committed. Element-wise sparsity reduces
the compressed delta transport and storage, but not the full-target checksum,
sharding, runtime-layout conversion, or CPU-to-GPU commit.

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
incorrect. With a fixed speculative draft, CPU staging and commit cover the
target model only. Updating target and draft weights together is unsupported
because they cannot yet be committed atomically.

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
