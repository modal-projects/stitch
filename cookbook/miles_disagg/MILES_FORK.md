# Miles integration

Stitch installs an immutable Miles revision over a dated trainer image:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202609290439"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "4aa180e9502d057cf3733593a82a513cea25f645"
```

The image supplies the compiled CUDA, Transformer Engine, and Megatron-LM
environment. The source pin belongs to
`modal-projects/miles:stitch-miles`: upstream Miles main at
`3439ec7513`, followed by twenty reviewed integration changes.
Each upstream PR is represented by one commit; the branch carries no additional
runtime patches. Stitch no longer patches Miles at container startup.

## Carried behavior

| Area | Upstream review | Branch commit | Responsibility |
| --- | --- | --- | --- |
| Checkpoint selection | [#2688](https://github.com/radixark/miles/pull/2688) | `34514fd61c` | Preserve explicit resume selection and isolate role-specific checkpoint selectors. |
| Disk delta | [#3237](https://github.com/radixark/miles/pull/3237) | `2a42496dc2` | Match emitted tensor names, shapes, dtypes, and raw checkpoint layouts before XOR encoding. |
| External fleet | [#3236](https://github.com/radixark/miles/pull/3236) | `b3af689af3` | Treat one independently managed rollout service as an opaque endpoint, publish its deltas, and overlap first-rollout generation with baseline capture. |
| Request policy | [#3344](https://github.com/radixark/miles/pull/3344) | `29b172b4d3` | Attach request-scoped admission and routing metadata at the session-server boundary. |
| Session collection | [#3736](https://github.com/radixark/miles/pull/3736) | `4153ed2d9d` | Keep sample materialization off the shared event loop, bound nested native threading, and make the collection deadline configurable. |
| Agent failure isolation | [#3863](https://github.com/radixark/miles/pull/3863) | `d7116370f5` | Retire failed agent sessions while returning an aborted sample instead of a partial trajectory. |
| Partial rollout groups | [#3702](https://github.com/radixark/miles/pull/3702) | `001ad989cc` | Optionally retain completed trajectories from aborted groups when at least two survive, using dynamic global batch sizing. |
| Ray placement | [#3640](https://github.com/radixark/miles/pull/3640) | `a4d3d2d004` | Resolve hard head affinity from Ray's GCS-backed node table without depending on dashboard reachability. |
| Canonical FP8 encoding | Not yet upstreamed | `d82fae3678` | Use the checkpoint's FP8 block representation for live exports and delta generation. |
| FP8 quantization scope | Not yet upstreamed | `ddc30d48ae` | Keep visual, non-matrix, and block-untileable weights in their checkpoint-defined high precision. |
| Rollout weight views | Not yet upstreamed | `46f5ec0004` | Publish independent checkpoint-defined rollout precisions, attribute samples to their source view, and preserve checksum-verified per-view baselines across resume. |
| Rollout checkpoint views | Not yet upstreamed | `5ba5e1cab6` | Save one complete HF checkpoint per rollout precision, preserving checkpoint-owned tensors that are absent from the trainer model. |
| Score-centered policy gradients | Not yet upstreamed | `871c7a1892` | Preserve sampler candidate probabilities through native and session rollouts and train with score centering plus optional truncated or masked importance weights. |
| Parallel score-centering validation | Not yet upstreamed | `b2656a4059` | Run the unchanged per-sample candidate checks on a bounded thread pool, raising the same first-in-order error as the serial loop. |
| Drain during weight publication | Not yet upstreamed | `c802bb390e` | Optionally convert the next fully-async batch while the weight update publishes, measuring its staleness against the version that update publishes. |
| Per-view mismatch diagnostics | Not yet upstreamed | `8448a2896b` | Report train-vs-rollout logprob difference and KL per rollout weight view, leaving the loss and existing metrics unchanged. |
| Mismatch diagnostics by staleness | Not yet upstreamed | `19459cde3c` | Split train-vs-rollout mismatch by the version lag between a sample's generation and the batch's training version. |
| Mismatch diagnostics per rollout pool | Not yet upstreamed | `d09c5900b7` | Split the train-vs-rollout mismatch diagnostics by the rollout source (pool and view) each sample came from, listed by `--rollout-sources`, all detached from the loss. |
| Prompt-mean loss and router freezing | Not yet upstreamed | `255ecedc2b` | Add `--prompt-mean-loss` (a token mean within each prompt's rollouts, then an equal-weight mean over prompts) and `--freeze-moe-router` (MoE router weights stay at their initial values). |
| Core mismatch split | Not yet upstreamed | `4aa180e950` | Report the mismatch split as KL and the share of tokens outside a [1/5, 5] trainer/sampler ratio, token-weighted whatever the loss aggregation, with no per-sample loop or host sync. |

The integration boundary is intentionally small:

- Miles owns trainer weights, monotonically increasing publication versions,
  delta encoding, and the session-server request-hook contract.
- Stitch supplies the external endpoint, the version already served at trainer
  startup, run/store coordinates for the publication hook, and the rollout
  request policy.
- SGLang owns checkpoint materialization, staged application, checksums, and
  admission while an update commits.

Megatron compatibility fixes are upstreamed to the Megatron fork and baked into
the dated trainer image; Stitch does not patch Megatron at runtime.

## Updating the pin

1. Rebase the integration branch onto the intended upstream Miles revision.
2. Drop changes already present upstream and retain only independently justified
   behavior.
3. Use a dated image whose Megatron-LM, Transformer Engine, and CUDA libraries
   match that Miles revision.
4. Run focused Miles tests for checkpoint selection, the external endpoint,
   request policy, disk delta, session collection, partial groups, and Ray head
   placement.
5. Validate fresh start, weight publication, mid-run replica join, and checkpoint
   resume end to end before advancing this immutable SHA.
