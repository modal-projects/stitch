# SGLang fork

Stitch overlays an SGLang fork onto the matching upstream image. The fork
adds general, asynchronous checkpoint staging and correct complete-weight
loading for quantized rollout models.

## Pin

[`serving_image.py`](serving_image.py) is the executable source of truth for the
default runtime:

```python
DEFAULT_SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.20",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-v0.5.20",
    commit="18df9cb22e5b7c0b3dc5303376bf61c4a4090db4",
    patches=(str(_PATCHES_DIR / "sglang-gemma-rmsnorm-staged-load.patch"),),
)
```

The branch is upstream v0.5.20 plus four independently reviewable layers:

| Layer | Responsibility |
| --- | --- |
| Reload lifecycle | Restore checkpoint-facing layouts, run each quantization method's native loader and post-load hooks, and fail closed if a partially mutated model cannot be rolled back. |
| Verified materialization | Apply and fold complete XOR delta lineages in canonical checkpoint space, verify the published checksum, and durably materialize disk targets. |
| CPU staging | Build bounded rank-ready host images while serving, optionally keep the canonical checkpoint on local NVMe, then commit every runtime storage in place. |
| Serving correctness | Preserve request aborts, routed-expert state, and sampling masks across data-parallel and speculative paths, and surface scheduler-process failures. |

The branch history keeps these physical responsibilities in separate commits;
the immutable pin above is the executable definition of the stack.

The image and immutable source pin stay together so the Python overlay remains
ABI-compatible with the image's CUDA and C++ extensions. SGLang v0.5.20 includes
Kimi K3, so all cookbook recipes now use this one runtime line. The fork's MXFP4
staging path transforms runtime layouts on GPU before caching rank-ready host
images.

## API

Enable one inactive destination when starting the server:

```python
"--weight-update-staging": "disk",  # or "cpu"
"--weight-version": "0",
```

Disk staging requires `--weight-update-local-checkpoint-dir`. With CPU staging,
the same option keeps the canonical checkpoint on local NVMe; omit it to keep
the canonical checkpoint in RAM. Server startup initializes the destination
from the checkpoint already loaded by SGLang before the server becomes ready.

`POST /prepare_weight_update` reconstructs and verifies a complete inactive
target without changing live weights:

```json
{
  "checkpoint_source_dir": "/stitch/<run-id>/updates",
  "target_version": 7
}
```

`checkpoint_source_dir` contains the immutable `weight_vNNNNNN` publications.
Preparation follows and folds the verified lineage from the currently served
version to `target_version`. A second request for the same prepared version is
idempotent. Before commit, a newer target in the same lineage may supersede the
inactive preparation; stale targets are rejected and the served weights remain
unchanged.

`POST /commit_weight_update` exposes the prepared target:

```json
{
  "target_version": 7,
  "abort_all_requests": false,
  "torch_empty_cache": false,
  "flush_cache": false
}
```

`flush_cache` defaults to `true`, which requires an idle scheduler. Stitch sets
it to `false` for in-place commits because every request is keyed by its served
weight version, so cached prefixes from different versions cannot alias.

Disk commit runs SGLang's native complete-checkpoint loader and verifies that
all live tensor layouts and storage addresses are preserved. CPU commit copies
the complete rank images into the existing target-model CUDA storages. A
speculative draft model remains fixed in either mode; target verification
preserves generation correctness while its acceptance rate may change as the
target evolves.

The separation is the pause boundary: after startup, preparation may overlap
rollout generation, while commit is coordinated with engine quiescence. CPU
commit is the short H2D-only path; disk commit performs a complete native
checkpoint reload and therefore holds the pause longer.

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

The rollout engine initially loads its boot checkpoint directly from the
configured model path. Its logical version is normally v0 and may be a saved
version for a resumed run. SGLang initializes the mutable local checkpoint
before declaring the server ready, then Stitch reconciles it to the visible
version before the replica enters rotation. Local storage must hold the mutable
checkpoint plus filesystem headroom; the immutable boot checkpoint remains in
its configured source.

The commit RPC still reads and transforms the complete target checkpoint using
the load format configured for the initial server load. On hosts without GDS,
recipes select fastsafetensors’ supported no-GDS mode:

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
4. `/commit_weight_update` performs distributed preflight and copies the
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
"--weight-update-staging": "cpu",
"--weight-update-max-compile-group-gb": "8",
```

By default, both the canonical checkpoint and rank images remain in RAM. Keep
only the rank images in RAM by placing the canonical checkpoint on writable
host-local storage:

```python
"--weight-update-local-checkpoint-dir": "/local-checkpoint/canonical",
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
| GLM-5.2 mixed NVFP4/BF16 | 4 | 617.63 GB | 155.83 GB × 4 | 0.69 MB |
| GLM-5.2 FP8 | 4 | 755.63 GB | 188.30 GB × 4 | 0 |
| GLM-5.3-Flash FP8 | 8 | 328.34 GB | 40.73 GB × 8 | 0.05 MB |
| Kimi K2.6 NVFP4 | 4 | 595.19 GB | 151.17 GB × 4 | 0 |
| Kimi K3 MXFP4 | 8 | 1.561 TB | 207.47 GB × 8 | 8.04 MB |

Allow additional memory for the engine process, delta decoding, and bounded
loader staging. Modal memory requests use `(request, limit)`. K3 all-RAM
validation requires a 4 TiB limit. Exact-final GLM-5.2 FP8 validation reached
1.58 TB of cgroup memory with the canonical checkpoint in RAM and 1.54 TB with
it on NVMe. The NVMe path's file-cache pages are reclaimable; its persistent
allocation is the rank images and loader state.

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

## Stitch-side patches

`SGLangRuntime.patches` lists `git diff` files under [`patches/`](patches/) that
`build_serving_image()` applies to the fork checkout at image build time, after
`git checkout --detach <commit>` and before `python/` is copied over the base
image. Every patch goes through `git apply --check` first, so a patch that no
longer applies (or is already part of the pin) fails the image build rather than
the running replica. Patches are a staging area for fixes that belong in the
fork but have not yet advanced the pin.

| Patch | What | Why |
| --- | --- | --- |
| `sglang-gemma-rmsnorm-staged-load.patch` | `GemmaRMSNorm` (the zero-centered `weight + 1` norm used by Qwen3.5/3.6 and Qwen3-Next) exposes its precomputed `gemma_weight` buffer via `get_derived_weight_tensors()` and refreshes it in `process_weights_after_weight_commit()`. | Its `weight_loader` wrote `weight + 1` into `self.gemma_weight`. Under CPU staging the loader runs against a same-device *shadow* module, but a non-persistent buffer that is not declared derived stays shared with the live module, so the loader crashed with a CUDA/CPU device mismatch (`rank weight compilation failed ... Expected all tensors to be on the same device`), and had it not crashed it would have mutated live serving state before commit. Declaring the buffer derived gives the shadow its own copy and puts it in the rank image, so preparation never touches live GPU state; the post-commit hook then rederives `gemma_weight = weight + 1` in place on the live module (stable storage for CUDA graphs and fused paths), matching what the disk path's `_rebind_parameter_aliases` already does. |

To drop a patch once it is upstreamed into the fork: advance `commit` in
`DEFAULT_SGLANG_RUNTIME`, remove the entry from `patches`, and delete the file.
`git apply --check` fails on the already-patched tree, so forgetting the second
step is caught at image build time.

## Re-porting

For a new SGLang release:

1. create `stitch-sglang-vX` from the exact upstream tag;
2. audit which fork responsibilities the release already provides and omit
   superseded code;
3. reapply the remaining responsibilities as separate commits;
4. audit the release's loader, quantization, scheduler, process-group, and
   CUDA-graph primitives and delete fork code superseded upstream;
5. run SGLang’s own pre-commit hooks and focused unit tests;
6. validate generation before, during, and after one complete delta update on
   FP8 and ModelOpt NVFP4, and validate MXFP4 transforms on Blackwell; and
7. update the image, branch, immutable commit, and this file.
