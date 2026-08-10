# Miles fork

Stitch’s Miles recipes install an immutable Miles revision over a dated trainer
image. [`trainer_image.py`](trainer_image.py) defines the shared default:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202607290235"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "b21dc7df8dcb0d4fd3fc07f6a6be1ee9fa35d369"
```

The pin is `modal-projects/miles:stitch-miles-fully-async-swe`. It contains a
Miles fully-async stack based on upstream main at `6bc45ad39`, followed by the
Stitch external-fleet integration.

## Miles fully-async stack

These commits are useful independently of Stitch:

| Commit | Responsibility |
| --- | --- |
| `ad0411c85` | Assemble additional routing-replay rows during in-place updates. |
| `35c78a78b` | Preserve in-flight requests while changing their weight version. |
| `6d638ba8a` | Make distributed service startup and teardown deterministic. |
| `9be219f58` | Report generation and queue staleness separately. |
| `63da214e1` | Distinguish explicit checkpoint resume from a fresh run. |
| `bd0216740` | Load optional P2P weight-sync dependencies lazily. |
| `717802e21` | Validate canonical disk-delta tensor layouts. |
| `13dd38252` | Report trainer-rollout policy drift accurately. |
| `a658a505c` | Make session endpoint affinity and lifecycle explicit. |
| `6dbc5ab45` | Add the Modal Sandbox v2 transport for SWE rollouts. |
| `080e58555` | Honor rollout-only LoRA configuration. |
| `4e98074fa` | Add and validate the mini-SWE rollout adapter. |
| `113271f5e` | Add CISPO policy optimization. |
| `d688df471` | Replay exact truncated-sampling support during training. |
| `53c402487` | Compact routing-replay expert IDs for transport. |
| `719e2844d` | Suppress per-request session access logs. |
| `45a2c5b25` | Reject aborted generations at the session protocol boundary. |
| `55b69d37a` | Continue disk-delta versions from a loaded checkpoint. |

## Stitch integration

| Commit | Responsibility |
| --- | --- |
| `b82540407` | Route fully-async generation through an external fleet with version constraints and finite timeouts. |
| `3b51b9894` | Publish disk deltas without Miles-managed rollout-engine handles. |
| `b21dc7df8` | Overlap external weight updates with active rollout. |

The dated image supplies Megatron-LM, TransformerEngine, CUDA, and other
compiled dependencies. Miles is installed over it with `--no-deps`, so the
image and Miles revision must remain compatible.

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
