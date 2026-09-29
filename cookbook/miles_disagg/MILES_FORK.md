# Miles integration

Stitch installs an immutable Miles revision over a dated trainer image:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202609290439"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "b0f5e085e411ed5da06979715ad4236a6250b810"
```

The image supplies the compiled CUDA, Transformer Engine, and Megatron-LM
environment. The source pin belongs to
`modal-projects/miles:stitch-miles`: upstream Miles main at
`9e4260de04`, followed by twelve reviewed integration changes.
Each upstream PR is represented by one commit; the branch carries no additional
runtime patches. Stitch no longer patches Miles at container startup.

## Carried behavior

| Area | Upstream review | Branch commit | Responsibility |
| --- | --- | --- | --- |
| Checkpoint selection | [#2688](https://github.com/radixark/miles/pull/2688) | `0323ce6aa1` | Preserve explicit resume selection and isolate role-specific checkpoint selectors. |
| Disk delta | [#3237](https://github.com/radixark/miles/pull/3237) | `01cff40953` | Match emitted tensor names, shapes, dtypes, and raw checkpoint layouts before XOR encoding. |
| External fleet | [#3236](https://github.com/radixark/miles/pull/3236) | `ac3bb0b510` | Treat one independently managed rollout service as an opaque endpoint, publish its deltas, and overlap first-rollout generation with baseline capture. |
| Request policy | [#3344](https://github.com/radixark/miles/pull/3344) | `9810f4e3f6` | Attach request-scoped admission and routing metadata at the session-server boundary. |
| Session collection | [#3736](https://github.com/radixark/miles/pull/3736) | `0f58b0ceab` | Keep sample materialization off the shared event loop and make the collection deadline configurable. |
| Partial rollout groups | [#3702](https://github.com/radixark/miles/pull/3702) | `97d69458dc` | Optionally retain completed trajectories from aborted groups when at least two survive, using dynamic global batch sizing. |
| Ray placement | [#3640](https://github.com/radixark/miles/pull/3640) | `573c12cc91` | Resolve hard head affinity from Ray's GCS-backed node table without depending on dashboard reachability. |
| Canonical FP8 encoding | Not yet upstreamed | `9b5357a899` | Use the checkpoint's FP8 block representation for live exports and delta generation. |
| FP8 quantization scope | Not yet upstreamed | `fe40e526a8` | Keep visual, non-matrix, and block-untileable weights in their checkpoint-defined high precision. |
| Rollout weight views | Not yet upstreamed | `05cdb526d9` | Publish independent checkpoint-defined rollout precisions, attribute samples to their source view, and preserve checksum-verified per-view baselines across resume. |
| Rollout checkpoint views | Not yet upstreamed | `d714082fb7` | Save one complete HF checkpoint per rollout precision, preserving checkpoint-owned tensors that are absent from the trainer model. |
| Score-centered policy gradients | Not yet upstreamed | `b0f5e085e4` | Preserve sampler candidate probabilities through native and session rollouts and train with score centering plus optional truncated or masked importance weights. |

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
