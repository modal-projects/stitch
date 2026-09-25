# Miles integration

Stitch installs an immutable Miles revision over a dated trainer image:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202609250800"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "3082b60e69e0528c9d4c7092a2514c00dcde4e59"
```

The image supplies the compiled CUDA, Transformer Engine, and Megatron-LM
environment. The source pin belongs to
`modal-projects/miles:stitch-miles`: upstream Miles main at
`41c5e38b94`, followed by reviewed changes that remain open upstream and the
branch-only Modal SWE adapter. Stitch no longer patches Miles at container
startup.

## Carried behavior

| Area | Upstream review | Responsibility |
| --- | --- | --- |
| Partial rollout groups | [#3702](https://github.com/radixark/miles/pull/3702) | Optionally retain completed trajectories from aborted groups when at least two survive, using dynamic global batch sizing. |
| Resume and saving | [#2688](https://github.com/radixark/miles/pull/2688), [#3616](https://github.com/radixark/miles/pull/3616) | Preserve explicit resume selection and retain declared source-owned tensors. |
| External fleet | [#3236](https://github.com/radixark/miles/pull/3236), [#3344](https://github.com/radixark/miles/pull/3344) | Treat one opaque URL as the rollout fleet and expose a request-policy hook with explicit arguments. |
| Disk delta | [#3237](https://github.com/radixark/miles/pull/3237) | Match emitted tensor names, shapes, dtypes, and raw checkpoint layouts before XOR encoding. |
| NVFP4 | [#3601](https://github.com/radixark/miles/pull/3601) | Adapt Qwen3.6 NVFP4 rollout checkpoints. |
| Modal SWE | Branch-only | Provide the Modal Sandbox transport and verified mini-SWE agent adapter used by these recipes; upstream Miles does not ship this provider-specific example. |
| Ray placement | [#3640](https://github.com/radixark/miles/pull/3640) | Resolve the Ray head once in the driver so head-pinned workers do not depend on worker-side dashboard access. |

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
