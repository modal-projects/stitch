# Miles fork

Stitch’s Miles recipes install an immutable Miles revision over a dated trainer
image. [`trainer_image.py`](trainer_image.py) defines the shared default:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202607290235"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "6e2db0b80a7252743e37cad7b73fd8a9ee4b1fee"
```

The pin is `modal-projects/miles:stitch-miles-fully-async-swe`. It contains a
Miles fully-async stack based on upstream main at `0113aa5ab`, followed by the
Stitch external-fleet integration.

## Miles fully-async stack

These commits are useful independently of Stitch:

| Commit | Responsibility |
| --- | --- |
| `b6627fbcf` | Assemble additional routing-replay rows during in-place updates. |
| `3ca48bb28` | Make distributed service startup and teardown deterministic. |
| `6b792d8a0` | Report generation and queue staleness separately. |
| `8f812030e` | Distinguish explicit checkpoint resume from a fresh run. |
| `15ab881ca` | Load optional P2P weight-sync dependencies lazily. |
| `84a09f66b` | Validate canonical disk-delta tensor layouts. |
| `deaf1af70` | Report trainer-rollout policy drift accurately. |
| `679845a10` | Add the Modal Sandbox v2 transport for SWE rollouts. |
| `d2afa6639` | Add and validate the mini-SWE rollout adapter. |
| `0788a3cfc` | Add CISPO policy optimization. |
| `56116e18b` | Replay exact top-p sampling masks during training. |
| `a59b628bd` | Compact routing-replay expert IDs for transport. |
| `4920cce83` | Suppress per-request session access logs. |
| `fac618950` | Reject aborted generations at the session protocol boundary. |

## Stitch integration

| Commit | Responsibility |
| --- | --- |
| `59931a089` | Route fully-async generation through an external fleet with version constraints and finite timeouts. |
| `5aa513895` | Publish disk deltas without Miles-managed rollout-engine handles. |
| `6e2db0b80` | Overlap external weight updates with active rollout. |

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
