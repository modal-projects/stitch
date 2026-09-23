# Miles integration

Stitch installs an immutable Miles revision over a dated trainer image:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202609231228"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "4aa480cf44a7447d96b1fb2fe5f7049c2411fcba"
```

The image supplies the compiled CUDA, Transformer Engine, and Megatron-LM
environment. The source pin belongs to
`modal-projects/miles:stitch-miles`: upstream Miles main at
`fbc511100c`, followed by reviewed changes that remain open upstream. Stitch no
longer patches Miles at container startup.

## Carried behavior

| Area | Upstream review | Responsibility |
| --- | --- | --- |
| Async rollout | [#3062](https://github.com/radixark/miles/pull/3062), [#3339](https://github.com/radixark/miles/pull/3339) | Dispose blocked producers and decode session samples off the event loop. |
| Sampling replay | [#3354](https://github.com/radixark/miles/pull/3354) | Replay bounded top-p support during training. |
| Resume and saving | [#2688](https://github.com/radixark/miles/pull/2688), [#3342](https://github.com/radixark/miles/pull/3342), [#3616](https://github.com/radixark/miles/pull/3616) | Preserve explicit resume selection, overlap final HF export, and retain declared source-owned tensors. |
| External fleet | [#3236](https://github.com/radixark/miles/pull/3236), [#3344](https://github.com/radixark/miles/pull/3344), [#3637](https://github.com/radixark/miles/pull/3637) | Treat one opaque URL as the rollout fleet, expose request policy, and give external agents routable session URLs while Miles retains private control-plane addresses. |
| Disk delta | [#3237](https://github.com/radixark/miles/pull/3237) | Match emitted tensor names, shapes, dtypes, and raw checkpoint layouts before XOR encoding. |
| NVFP4 | [#3592](https://github.com/radixark/miles/pull/3592), [#3601](https://github.com/radixark/miles/pull/3601) | Preserve nested BF16 carve-outs and adapt Qwen3.6 rollout checkpoints. |
| Data and agents | [#2801](https://github.com/radixark/miles/pull/2801), [#2802](https://github.com/radixark/miles/pull/2802), [#3600](https://github.com/radixark/miles/pull/3600), [#3617](https://github.com/radixark/miles/pull/3617) | Support mixed text/multimodal datasets and discard infrastructure-only Harbor failures rather than training on them. |
| Ray placement | Branch-only | Resolve the Ray head node before worker launch so head-pinned control actors do not depend on worker-side dashboard access. |

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
4. Run focused Miles tests for the external endpoint, request policy, disk
   delta, HF export, data processing, and NVFP4 planning.
5. Validate fresh start, weight publication, mid-run replica join, and checkpoint
   resume end to end before advancing this immutable SHA.
