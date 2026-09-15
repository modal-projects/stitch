# Miles fork

Stitch’s Miles recipes install an immutable Miles revision over a dated trainer
image. [`trainer_image.py`](trainer_image.py) defines the shared default:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202607290235"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "13fa3952ffe682955806cef8e9343b9d7d891af8"
```

The pin belongs to `modal-projects/miles:stitch-miles-fully-async-swe`. The fork
contains a Miles fully-async stack based on upstream main at `5330aadc0`,
followed by the Stitch external-fleet integration.

## Miles fully-async stack

These commits are useful independently of Stitch:

| Commit | Responsibility |
| --- | --- |
| `c190aca57` | Distinguish explicit checkpoint resume from a fresh run. |
| `21feb8664` | Load optional P2P weight-sync dependencies lazily. |
| `a1cbe0a10` | Preserve in-flight requests while changing their weight version. |
| `9953ae5da` | Bound SGLang control-plane health requests. |
| `10fca8606` | Resume rollout health monitoring after weight updates. |
| `814dce696` | Close persistent asynchronous rollout producers cleanly. |
| `2e6a9f30b` | Report generation and queue staleness separately. |
| `b1a84e852` | Replay exact truncated-sampling support during training. |
| `eb869236c` | Compact routing-replay expert IDs for transport. |
| `079e49695` | Shard routing-replay transport by pipeline stage. |
| `0324040fb` | Start session-server pools against one deadline. |
| `615b5f48d` | Make session endpoint affinity and lifecycle explicit. |
| `9f596da2b` | Decode session samples off the event loop and report timing. |
| `99ea71afa` | Suppress per-request session access logs. |
| `e04e617d3` | Reject aborted generations at the session protocol boundary. |
| `ace083efc` | Keep session and agent failures local to their trajectories without accepting partial samples. |
| `def5ac37c` | Add the Modal Sandbox v2 transport for SWE rollouts. |
| `f1fcf842d` | Add and validate the mini-SWE rollout adapter, including policy-limit terminals. |
| `1854d1c14` | Replay exact truncated-sampling support with DFlash. |
| `073e537f3` | Read current speculative-decoding counters in metrics. |
| `d598798c5` | Adapt NVFP4 rollout checkpoints for Qwen3.6. |
| `2da7d0955` | Validate canonical disk-delta tensor layouts. |
| `e917930bc` | Buffer distributed HF export and overlap shard writes with Megatron checkpoint saves. |
| `6bb57f7eb` | Restore CPU test isolation, rollout fixtures, and Megatron tokenizer compatibility. |
| `360090909` | Format the maintained Miles stack. |

## Stitch integration

| Commit | Responsibility |
| --- | --- |
| `4313a589c` | Route fully-async generation through an external fleet with version constraints and finite timeouts. |
| `9f96381eb` | Publish disk deltas without Miles-managed rollout-engine handles. |
| `13fa3952f` | Overlap external weight updates with active rollout. |

The dated image supplies Megatron-LM, TransformerEngine, CUDA, and other
compiled dependencies. Miles is installed over it with `--no-deps`, so the
image and Miles revision must remain compatible.

The [GLM-5.3-Flash recipe](configs/glm5_3_flash_nvfp4.py) selects its own immutable
image and Miles revision, including [Miles #7](https://github.com/modal-projects/miles/pull/7).
Its model overlay is applied before the cookbook runtime patches. Miles owns
native checkpoint completion and session transport; the cookbook owns Volume
durability and provider-specific retry policy.

## Responsibilities

Miles owns trainer-side state:

- gather training weights into canonical Hugging Face tensor names;
- retain the previous canonical bytes and publish element-wise XOR or overwrite
  deltas with final-state checksums;
- atomically expose a completed version to Stitch; and
- attach required weight-version constraints to rollout requests.

Stitch and SGLang own rollout-host state: materializing publications, verifying
lineage and checksums, compiling disk or CPU destinations, gating admission for
the short engine pause, and reporting the version that served each request.

Canonical checkpoint bytes are required only for `disk-delta`. Other Miles
weight transports retain their established engine-runtime layouts. Delta
encoding is dtype- and quantization-agnostic; model-specific converters are
responsible only for producing the same tensor names, dtypes, shapes, and byte
layouts as the base checkpoint.

## Updating the pin

1. Start from the intended upstream Miles revision.
2. Reapply only behavior still missing upstream, keeping general Miles changes
   below the Stitch-specific integration commits.
3. Keep canonical checkpoint layout changes scoped to `disk-delta`.
4. Use a dated image whose Megatron-LM and TransformerEngine match the target
   Miles revision.
5. Run Miles pre-commit and the focused tests for export, endpoint routing, and
   staged engine requests.
6. Run a multi-step rollout test with a replica joining mid-run.
7. Update the immutable SHA and this file.
