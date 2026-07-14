# The sglang fork stitch pins

stitch's rollout engines run a **patched sglang**, not upstream: the disaggregated
weight-sync path (engine-side `/pull_weights`, correct quantized reloads, and the
O(delta) partial reload) is not in upstream sglang yet. This file is the source of
truth for *what* we patch, *why*, and *how to move the patch stack onto a newer
sglang release*.

## The pin

`cookbook/common/serving_image.py` builds the serving image by overlaying the fork's
`python/` onto a stock sglang container:

```python
SGLANG_IMAGE_TAG   = "lmsysorg/sglang:v0.5.15"          # base kernels/CUDA
SGLANG_FORK_BRANCH = "stitch-sglang-v0.5.15"            # modal-projects/sglang
SGLANG_FORK_COMMIT = "43fccb0cf6be72ecb0096d010eeaf8507cc302d0"
```

The branch is **`v0.5.15` + the patch stack below, nothing else** — every commit
`git log v0.5.15..stitch-sglang-v0.5.15` is one of ours. Because only `python/` is
overlaid, `SGLANG_IMAGE_TAG` **must** be the same sglang version the branch is based
on (`v0.5.15`), or the baked C++/CUDA ops will be ABI-mismatched with the python.

## The patch stack

Two tiers. The **base case** makes disaggregated weight sync work at all; the
**optimization** tier makes a reload O(delta) instead of O(full checkpoint). Read the
commit bodies for the details — this is the map.

### Base case — weight sync works

1. **`[RL] /pull_weights: engine-side pull ... into a host-local checkpoint`**
   The disaggregated receiver: `POST /pull_weights` walks the published
   `weight_v{N}/` chain from the nearest full anchor, replays deltas (xor + zstd,
   xxh3 checksum) into a host-local checkpoint that `/update_weights_from_disk` then
   reloads, while the engine keeps serving. Hardened for an eventually-consistent
   volume mount (whole-file in-memory read + size-verify before the xor, one reload
   per host, reseed from the pristine boot checkpoint).
   *Upstreaming:* https://github.com/sgl-project/sglang/pull/30367

2. **`[RL] update_weights_from_disk: load quantized checkpoints like initial loading`**
   Makes an in-place reload of a quantized checkpoint reproduce `init(checkpoint)`
   (fp8 blockwise + compressed-tensors block/channel): restore latched quant scale
   state before loading, fix the rollback path, keep weights refillable, and fix the
   UE8M0 scale inverse for row counts not divisible by 128. Without this, fp8
   reloads silently diverge from the served kernel format.
   *Upstreaming:* https://github.com/sgl-project/sglang/pull/30761

### Optimization — O(delta) reload

3. **`[RL] reload: record/replay load plans for repeated reloads`**
   Record the model's first-reload weight dispatch once, replay it directly after
   (skip the per-tensor routing scan, parallelize the load). Opt in per model
   (`supports_load_plan_replay`); gated by `SGLANG_ENABLE_RELOAD_LOAD_PLAN=1`. Falls
   back to the legacy loader on any failure.

4. **`[RL] reload: O(delta) partial reload via touched checkpoint names`**
   Given the touched tensor names (`weight_names`, from a delta apply), reload only
   those tensors + their fused/expert closures and re-post-process only the touched
   modules. Pre-flights every touched module's quant method for incremental support;
   any gap falls back to a full reload.

5. **`[RL] modelopt fp4: incremental post-loading for partial reloads`**
   The NVFP4 model-side enabler for (4): `process_weights_after_partial_loading`
   re-derives kernel state for only the touched experts — CUTLASS per-expert
   re-swizzle and TRT-LLM per-expert re-alignment — declining marlin/cutedsl and any
   padded/whole-layer layout to a safe full reload. Also restores NVFP4 raw bit-exact
   compare in the weight checker (v0.5.15 replaced it with a NotImplementedError) so a
   partial reload is byte-verifiable against a full one via `/weights_checker`. Without
   this, NVFP4 partial reload declines and pays a full reload.
   *Upstreaming:* not yet filed.

6. **`[RL] load plan: record during the initial load so reloads start already-replaying`**
   Record the load plan during the model's initial boot load, so the first
   `update_weights_from_disk` already replays / goes O(delta) partial instead of paying
   a full record-reload — the full reload is eliminated from steady state (matters most
   for elastic joiners that boot then immediately catch up via deltas). Gated on the
   same flag; drops the plan and falls back to a plain load on any failure.

`SGLANG_ENABLE_RELOAD_LOAD_PLAN` is opt-in per recipe (off unless set): the NVFP4 configs
enable it via `SGLANG_ENV` — their native load is single-threaded, so replay is a large win —
while a recipe whose native load is already multithreaded+fast leaves it off (see
`cookbook/miles_disagg/configs/glm45_air_fp8.py`).

## Re-porting to a newer sglang release (`stitch-sglang-vX`)

When bumping the base (e.g. to `v0.5.16`):

1. Branch `stitch-sglang-vX` off the new tag on `modal-projects/sglang`.
2. Re-apply the five commits **in the order above** (cherry-pick from
   `stitch-sglang-v0.5.15`, or from the source branches `weight-sync-miles` /
   `fp8-reload-main` / `weight-sync-upstream`). Squash-preserving is fine; keep the
   two tiers legible.
3. **Audit before trusting a clean cherry-pick** — a clean apply does not prove the
   patch is still needed or correct. In particular:
   - `[RL] modelopt fp4 ...` (5) is tightly coupled to sglang's NVFP4 MoE post-load
     (`ModelOptNvFp4FusedMoEMethod.process_weights_after_loading`, the swizzle/alias
     helpers, and the MoE runner backends). This function is restructured often; the
     partial pass usually needs a genuine **rewrite**, not a port. The v0.5.15
     rewrite scopes the fast path to CUTLASS and declines other backends to a safe
     full reload.
   - Check whether the new base already upstreamed any patch (then drop it).
4. **Validate** a quantized partial reload is byte-identical to a full reload of the
   same bytes via `/weights_checker` (per-tensor checksums) before shipping — this is
   how the NVFP4 partial pass is proven correct.
5. Update the three constants in `cookbook/common/serving_image.py` and this file.

## Upstreaming

We want these in upstream sglang so the fork shrinks to zero. Open PRs:
`/pull_weights` → sgl-project/sglang#30367; quantized reload → sgl-project/sglang#30761.
The load-plan / partial-reload tier is not yet filed.
