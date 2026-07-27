# Miles fork

Stitch’s Miles recipes pin a small fork for opaque rollout-fleet routing and
canonical disk-delta publication. [`trainer_image.py`](trainer_image.py) is the
executable source of truth:

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202607260602"
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "1eb7520018446cb94b7406715f66dff1a271b53b"
```

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

The dated base image supplies Megatron-LM, TransformerEngine, CUDA, and the
other compiled dependencies. The fork is installed over it with `--no-deps`;
therefore the image and the upstream Miles revision must remain compatible.

## Responsibilities

Miles owns trainer-side state:

- gather training weights into canonical Hugging Face tensor names;
- retain the previous canonical bytes and publish element-wise XOR or overwrite
  deltas with final-state checksums;
- atomically expose a completed version to Stitch; and
- attach required weight-version constraints to rollout requests.

Stitch and SGLang own rollout-host state: materializing publications, verifying
lineage and checksums, compiling disk or CPU destinations, gating admission for
the short commit, and reporting the version that served each request.

Canonical checkpoint bytes are required only for `disk-delta`. Other Miles
weight transports retain their established engine-runtime layouts. Delta
encoding is dtype- and quantization-agnostic; model-specific converters are
responsible only for producing the same tensor names, dtypes, shapes, and byte
layouts as the base checkpoint.

## Re-porting

When updating Miles:

1. create a new branch from the desired `radixark/miles` main;
2. reapply the six responsibilities separately, dropping code already
   available upstream;
3. keep canonical checkpoint layout changes scoped to `disk-delta`;
4. use a dated Miles image whose Megatron and TransformerEngine match that
   upstream revision;
5. run Miles pre-commit plus the focused GPU tests for export, endpoint routing,
   and staged engine requests;
6. run a multi-step rollout test with a replica joining mid-run; and
7. update the immutable repository SHA and this file.
