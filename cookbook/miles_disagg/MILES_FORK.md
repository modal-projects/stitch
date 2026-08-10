# Miles forks

Stitch’s Miles recipes use a dated trainer image and an immutable Miles commit.
[`trainer_image.py`](trainer_image.py) defines the default:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202607290235"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "1eb7520018446cb94b7406715f66dff1a271b53b"
```

An experiment config may override `MILES_REPO_REF` without changing other
recipes. GLM-5.2 does this because it runs on Miles’ fully-async SWE branch.

## Standard weight-sync branch

The branch `stitch-weight-sync-v0516` is upstream `radixark/miles` main at
`52403d0e3` plus six commits:

| Commit | Responsibility |
| --- | --- |
| `7995ec27c` | Route generation through one opaque fleet endpoint, publish versions without per-engine handles, gate individual requests through a generic hook, and use a finite rollout read timeout. |
| `4572f97a6` | Emit canonical quantized checkpoint layouts for disk deltas, support GLM-Air channel FP8 and plain-language-model Kimi names, and preserve SGLang runtime layouts for P2P/broadcast. |
| `cf56d32b0` | Encode zero-dimensional quantization scalars by flattening before their byte view. |
| `748f1724a` | Use SGLang’s staged checkpoint API for Miles-managed disk-delta engines. |
| `d814b87c3` | Replace the unused delta progress bar with an explicit baseline-snapshot log. |
| `1eb752001` | Honor ModelOpt glob exclusions when exporting NVFP4 weights. |

## GLM-5.2 fully-async SWE branch

The GLM-5.2 and GLM-4.7 fully-async SWE recipes pin
`stitch-miles-fully-async-swe` at `b1020b596`. The Stitch branch is stacked on
`modal/feat/modal-swe-fully-async` at `d0c11a412`.

The fully-async branch owns behavior that is useful without Stitch:

| Commit | Responsibility |
| --- | --- |
| `de10d68d1` | Format the fully-async rollout files. |
| `ffcad9557` | Match GLM router tensors to the canonical checkpoint dtypes. |
| `c0661aa6a` | Validate canonical disk-delta layouts and encode scalar tensors safely. |
| `8857d8114` | Shard routing replay by trainer topology with a lossless compact wire dtype. |
| `8e140dbd9` | Apply the configured log-prob token budget when routing replay is enabled. |
| `791ef9593` | Bound Modal agent submissions before they enter Ray actor mailboxes. |
| `d0c11a412` | Format the added fully-async code. |

The Stitch branch adds only the external-fleet integration:

| Commit | Responsibility |
| --- | --- |
| `2fa28cde4` | Route session rollouts through an external fleet with request gating, version constraints, and finite timeouts. |
| `7bfb4a69a` | Publish disk deltas without Miles-managed rollout-engine handles. |
| `23b262bca` | Overlap the external fleet's initial delta snapshot with the first rollout. |
| `b1020b596` | Sort imports in the external-fleet adapter. |

This branch is additive and only backs fully-async SWE experiments. It does not
replace the standard recipe pin.

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

## Re-porting

When updating a Miles pin:

1. start from the intended upstream branch;
2. reapply only the responsibilities still missing upstream;
3. keep canonical checkpoint layout changes scoped to `disk-delta`;
4. use a dated Miles image whose Megatron and TransformerEngine match that
   upstream revision;
5. run Miles pre-commit plus the focused GPU tests for export, endpoint routing,
   and staged engine requests;
6. run a multi-step rollout test with a replica joining mid-run; and
7. update the immutable repository SHA and this file.
