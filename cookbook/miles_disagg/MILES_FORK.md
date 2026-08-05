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

`glm5_2_nvfp4.py` pins `stitch-miles-fully-async-swe` at `6280a2196ba3`. The
branch is `modal/feat/modal-swe-fully-async` at `a999ec511` plus:

| Commit | Responsibility |
| --- | --- |
| `bee4c4b61` | Format the fully-async files touched by the integration. |
| `7529d01d0` | Route session rollouts through one external fleet endpoint with request gating and finite timeouts. |
| `88b81a82f` | Publish disk deltas without Miles-managed rollout-engine handles. |
| `7e91d93ad` | Match GLM-5.2 router tensor dtypes to the canonical checkpoint. |
| `098e4dc78` | Encode zero-dimensional checkpoint tensors safely. |
| `6280a2196` | Read scalar and version-constraint model-info responses from an external fleet. |

This branch is additive and only backs the GLM-5.2 fully-async experiment. It
does not replace the standard recipe pin.

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
