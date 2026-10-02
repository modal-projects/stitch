# Miles integration

Stitch installs an immutable Miles revision over a dated trainer image:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202609290439"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "1a0f6a9cd8dfac14e358dd26ac2dccb86f82832e"
```

The image supplies the compiled CUDA, Transformer Engine, and Megatron-LM
environment. The source pin belongs to
`modal-projects/miles:stitch-miles-hetero`, the heterogeneous-RL experiment branch:
upstream Miles main at `3439ec7513`, followed by twenty-four reviewed integration changes.
`stitch-miles` tracks the pin on Stitch main.
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
| Prompt-mean loss and router freezing | Not yet upstreamed | `9f40167078` | Add `--prompt-mean-loss` (a token mean within each prompt's rollouts, then an equal-weight mean over prompts) and `--freeze-moe-router` (MoE router weights stay at their initial values). |
| Mismatch split by source and staleness | Not yet upstreamed | `e5d495e581` | Report the mismatch as KL and the share of tokens outside a [1/5, 5] trainer/sampler ratio, for all tokens and per view, per rollout source (`--rollout-sources`) and per version-lag bucket; token-weighted whatever the loss aggregation, with no per-sample loop or host sync. |
| Canonical view exports | Not yet upstreamed | `f9559086ee` | Export each rollout view in its canonical checkpoint layout, as its deltas are, so a replica booted from an export can apply the deltas that follow it. |
| Joint view publication | Not yet upstreamed | `0f09a4da66` | Publish rollout views that finish encoding together with one `--custom-update-weight-post-write-views-path` call, so one commit round covers them; no view waits for another. |
| Weight-sync timing | Not yet upstreamed | `c96c327f32` | Log where each weight sync's time goes (bucket production, conversion wait, pinned buffers, GPU-to-CPU copy, diff/compress workers); log lines only. |
| Signed mismatch direction | Not yet upstreamed | `90c3e94a08` | Report the mean signed trainer-minus-rollout log-ratio next to KL and the ratio tail, for all tokens and per view, rollout source and staleness bucket; one more row in the same pass. |
| Per-view mismatch size | Not yet upstreamed | `931466431e` | Report the mean absolute trainer-minus-rollout log-ratio per view, rollout source and staleness bucket again, next to its signed mean; one more row in the same pass. |
| Client reply without training candidates | Not yet upstreamed | `1a0f6a9cd8` | Cut the candidate logprobs the session server requests for training back to the client's own `top_logprobs` in the reply, in both the OpenAI logprobs and `meta_info`; the session record keeps them all. |

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
