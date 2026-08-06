# SGLang fork

Stitch overlays a small SGLang fork onto the matching upstream image. The fork
adds general, asynchronous checkpoint staging and correct complete-weight
loading for quantized rollout models.

## Pin

[`serving_image.py`](serving_image.py) is the executable source of truth for the
default runtime:

```python
DEFAULT_SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.16",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-v0.5.16",
    commit="7b09ce9f77c2290d5271c545d5b8b988c0479584",
)
```

The branch is upstream v0.5.16 plus:

| Commit | Responsibility |
| --- | --- |
| `49031cb24c` | Configure fastsafetensors with or without GDS and honor post-load cache release. |
| `5f6e1e613f` | Materialize and verify complete targets on host-local disk. |
| `a7e20596ba` | Restore checkpoint-facing quantized layouts for complete weight loading. |
| `c867782f3e` | Build verified, rank-ready CPU weight images from canonical targets. |
| `111a804d2e` | Expose asynchronous disk/CPU staging and CPU-to-GPU target-model commit APIs. |
| `e0859b7390` | Stream CPU delta lineages through bounded memory. |
| `77eca472e6` | Fold disk XOR lineages with bounded positional I/O. |
| `526e0ddca2` | Fail cache-flushing CPU commits before GPU mutation when the engine is busy. |
| `a562908a10` | Normalize native ModelOpt FP4 expert tensors through their existing loader path. |
| `1a4a4fd6b5` | Return aligned verifier logprobs for DFlash rollout tokens. |
| `607e107b44` | Store the canonical CPU-cache checkpoint on NVMe and overlap verified persistence with bounded rank-image compilation. |
| `1051a95a6a` | Balance CPU delta transforms across persistent worker tasks. |
| `0094b725b9` | Support top-p-only sampling masks through the native generation and chat-completions APIs. |
| `7b09ce9f77` | Return aligned top-p sampling masks for tokens accepted by DFlash SpecV2. |

The default Modal image also applies
[`sglang-dflash-prefill-attention-tp-padding.patch`](patches/sglang-dflash-prefill-attention-tp-padding.patch)
after checking out the immutable commit. DFlash captures the target model's
full prefill hidden states, which include up to `attention_tp_size - 1`
trailing alignment rows. The patch removes only that exact alignment suffix
before writing the unpadded draft KV locations and continues to reject every
other shape mismatch. Remove the image patch once the fork pin contains the
equivalent fix.

The image and branch must use the same SGLang release because Stitch overlays
Python code onto the image’s existing CUDA and C++ extensions.

Models that require another upstream SGLang line set `SGLANG_RUNTIME` in their
configuration. The image, fork branch, and immutable commit stay together so
the Python overlay remains ABI-compatible with the image.

[`kimi_k3_mxfp4.py`](../miles_disagg/configs/kimi_k3_mxfp4.py) pins the public
K3 image and `stitch-sglang-kimi-k3` fork. That fork ports the
same weight-sync responsibilities onto SGLang’s public `kimi-k3` branch; other
recipes continue to use the v0.5.16 default. Its K3-native loader narrows expert
lookup, batches safe copies, scopes post-load work to loaded modules, and
transforms Blackwell MXFP4 runtime layouts on GPU before caching rank-ready host
images.

## API

`POST /stage_weight_update` prepares a target while inference continues:

```json
{
  "base_checkpoint_dir": "/checkpoints/<artifact-id>",
  "checkpoint_source_dir": "/stitch/<run-id>/updates",
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
  existing target-model CUDA storages. A speculative draft model remains fixed;
  target verification preserves generation correctness while its acceptance
  rate may change as the target evolves.

The separation is the pause boundary: staging may overlap rollout generation,
while commit is the short operation coordinated by Stitch.

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

CPU mode keeps rank-ready images in RAM for the shortest commit:

1. After v0 begins serving, SGLang allocates one complete rank-ready image per
   local TP rank and either caches one canonical checkpoint per host in RAM or
   materializes it on host-local storage.
2. It builds v0 through the model’s ordinary weight loader and quantization
   hooks and verifies that the prepared runtime storages match the active model.
3. For every delta lineage, it reconstructs and checksums the canonical target,
   then builds every next rank image while inference continues. The in-memory
   path streams deltas through a bounded work budget; the storage-backed path
   reuses the transactional disk materializer.
4. `/update_weights_from_cpu` performs distributed preflight and copies the
   complete images into the existing CUDA storages without replacing storage
   pointers.

Stitch starts step 1 in the background. A published delta waits for that same
initialization task; it does not start a second cache build. An initialization
or staging failure is reported and the GPU remains on its prior weights.
Registering a model-sized rank image as CUDA host memory is also one-time work,
but can delay active inference on very large models. Initialize the CPU
destination before routing latency-sensitive traffic when warmup latency
matters.

CPU mode is delta-only. It rejects FULL publications and cannot reset a patched
live replica to another run’s v0; the controller must use disk mode or replace
that replica. This keeps a single canonical snapshot rather than retaining a
second rollback-sized CPU checkpoint. Compressed deltas are not retained in a
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
complete checkpoint copy during background initialization and adds NVMe
read/write work during preparation; it does not change the CPU-to-GPU engine
pause. Set `--weight-loader-drop-cache-after-load` to release clean checkpoint
pages after every local TP rank finishes compiling; otherwise the kernel may
retain those reclaimable pages for later reads.

The group bound limits transient loader work; it does not tune correctness or
assume a model architecture. An indivisible module larger than the requested
bound remains intact and is reported. The K3 recipe uses a 16 GiB bound to
balance its module granularity against transient RAM.

With the default in-memory canonical checkpoint, persistent host RAM is:

```text
one canonical checkpoint per host
+ one rank-local runtime image per local TP rank
```

With a storage-backed canonical checkpoint, persistent host RAM is the rank
images; local storage holds one canonical checkpoint. File-cache pages used
during preparation are reclaimable.

Measured component sizes are:

| Model | TP | Canonical checkpoint per host | Rank image |
| --- | ---: | ---: | ---: |
| GLM-4.5-Air FP8 | 4 | 112.6 GB | 27.2 GB × 4 |
| Kimi K2.6 NVFP4 | 4 | about 595 GB | about 151 GB × 4 |
| GLM-5.2 mixed NVFP4/BF16 | 4 | 617.6 GB | 156.3 GB × 4 |
| Kimi K3 MXFP4 | 8 | 1.561 TB | 207.5 GB × 8 |

Allow additional memory for the engine process, delta decoding, and bounded
loader staging. The supplied GLM-4.5 recipe requests `(512 GiB, 2 TiB)`;
GLM-5.2, Kimi K2.6, and Kimi K3 request `(1 TiB, 3 TiB)`, expressed as
`(request, limit)`.

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
