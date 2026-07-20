# The miles fork stitch pins

stitch's miles recipes run a **forked miles**, not upstream: the disaggregated disk-delta
publish path, FP8/NVFP4 HF export, and opaque-rollout routing are not in upstream miles
(`radixark/miles`) yet. This file is the source of truth for *what* we carry on top of
upstream, *why*, and *how to re-rebase onto a newer upstream main*.

## The pin

`cookbook/miles_disagg/trainer_image.py` clones the fork over a stock radixark base image
(which bakes Megatron-LM + TransformerEngine); miles is installed `--no-deps`, so the base
image's Megatron/TE is what actually runs.

```python
MILES_IMAGE_TAG = "radixark/miles:dev-202607182122"   # base: Megatron + TE 2.12.0 (torch 2.11+cu130)
MILES_REPO_URL  = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF  = "15cf7ed0344850affa354b8b81ad3acbda11474b"   # branch stitch-miles
```

`stitch-miles` is **upstream `radixark/miles` main (`c35f0e58`) + the six commits below,
nothing else** — every commit in `git log c35f0e58..stitch-miles` is one of ours. Keep
`MILES_IMAGE_TAG` on a base image whose Megatron matches the upstream `main` the branch is
rebased on (currently Jul-18), or miles will call Megatron APIs the baked Megatron lacks.

## The commit stack (6 commits on upstream main)

Grouped by concern; read the commit bodies for the details.

1. **Add NVFP4 RL support** — the upstream NVFP4 quantized-training support (routed-expert
   NVFP4 GEMMs, `te_precision` recipe, HF NVFP4 export), imported as one squashed commit.
   Kept isolated so it **drops cleanly once the equivalent lands in upstream main** (then
   re-rebase without it).
2. **fix(convert): exclude shared_experts from NVFP4** — routed-experts-only NVFP4 to match
   the trainer's `te_precision` recipe (reload-clean).
3. **FP8 disk-delta export** — publish fp8 updates in plain HF layout (no UE8M0 pre-transform),
   GLM-Air FP8 disk-delta export, and plain-LM export names.
4. **delta: flatten before the byte-view** — so 0-dim scalar tensors encode in the delta.
5. **disagg rollout** — opaque endpoint routing + request hook + publish-only mode
   (`--rollout-endpoint-url`) + a finite read timeout on the disagg `/generate` client.
6. **NVFP4 delta-encode: pass only the quantizer kwargs the installed TE supports** — a compat
   shim. The delta-encode passes 4over6 / row-scaled `NVFP4Quantizer` kwargs (all disabled)
   that exist only on TE ≥ 2.13; filter them against the signature so the encode runs on the
   base image's TE 2.12.0. **A no-op — and can be dropped — once the base image ships TE ≥ 2.13.**

Commits 1 and 6 are the two we expect to retire: (1) when upstream absorbs the NVFP4 support,
(6) when the base image's TE catches up to what the NVFP4 code calls.

## Re-rebasing onto a newer upstream main

1. Fetch `radixark/miles` main; `git rebase --onto <new-main> <old-base> stitch-miles`, or
   cherry-pick the six commits onto a fresh branch off the new main.
2. If upstream has **absorbed the NVFP4 support**, drop commit 1 and rebase the rest onto it.
3. If the base image moved to **TE ≥ 2.13**, drop commit 6 (the kwarg filter becomes a no-op).
4. Bump `MILES_IMAGE_TAG` to a base image whose Megatron/TE matches the new upstream main, then
   update the two constants here and in `trainer_image.py`.
5. Keep the grouping legible (≤ ~6 commits) and keep commit 1 a single squashed import.
