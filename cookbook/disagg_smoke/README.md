# disagg_smoke

A small Modal app that validates the pool-control work end to end on a minimal
Qwen model (`Qwen/Qwen2.5-0.5B-Instruct`) — no Kimi/large config, no Megatron
trainer. It exercises both:

1. **PR #5** — explicit pool claim/reset, fresh-run-id ownership, monotonic
   pointer, 1:1 trainer-call ↔ pool epoch.
2. **PR #6** — the cookbook consolidation (shared `cookbook.sidecar` /
   `serving` / `trainer_helpers` / `rollout_control` spine + thin per-trainer
   adapters, and the whole-cookbook image mount).

Instead of running a real trainer, it synthesizes **real** slime disk-delta
versions (xor + zstd + xxh3-128) from the base checkpoint (`delta.py`) and
publishes them through the real bulletin board, so the actual host-side delta
apply + reconcile path runs.

## Entrypoints

Invoke by **module path** (`-m`), not file path — like the other cookbook apps
(`modal deploy -m cookbook.slime_disagg.modal_train`). A bare file path names the
entrypoint module `app`, which the container can't import (`ModuleNotFoundError:
No module named 'app'`); `-m` resolves it to `cookbook.disagg_smoke.app`, which is
importable from the mounted cookbook package.

```bash
# GPU-free control-plane test (primary). Asserts everything itself; raises on failure.
modal run -m cookbook.disagg_smoke.app::control_plane_test

# 1x GPU live confirmation: real SGLang reload + version-pinned completion.
modal run -m cookbook.disagg_smoke.app::serving_smoke
```

### `control_plane_test` (no GPU)

Against a fake SGLang upstream, on the consolidated image:

- `claim` writes the empty pointer `<run_id>/weight_v000000` and resets the pool
  to base; startup converges there.
- Publishing a 2-version delta chain and reconciling patches the **local
  checkpoint** to the exact trainer-intended bytes (slime's real `apply_deltas`),
  and reloads the engine.
- A same-run rewind and a reused `run_id` both raise `PointerRewind`
  (fresh-run-id enforcement).
- A fresh run re-claims, resets the engine to base, and replays its own chain.
- Every consolidated module + thin adapter imports off the whole-cookbook mount
  (the PR #6 regression we fixed: with `include_source=False` a subdir-only mount
  would `ImportError` here).

### `serving_smoke` (1 GPU)

Runs the **real** consolidated sidecar (`python3 -m cookbook.slime_disagg.sidecar`)
in front of a real SGLang server on tiny Qwen, publishes one delta, triggers a
reconcile, and asserts the engine reloaded to v1 (real `update_weights_from_disk`)
and serves a completion pinned to `min_required_version: 1`.

It uses a vanilla single-GPU SGLang image (not the cookbook's Blackwell fa4 fork
build, which targets B200s) so it runs on a common 1×GPU; the whole-cookbook
mount + `--no-deps` decoder layers are identical, so the consolidation/reload
path is validated the same way.

## Config

Override via env vars when launching:

- `SLIME_SMOKE_REPO` / `SLIME_SMOKE_REF` — the slime checkout for the host-side
  decoder. Pin these to the **same** ref the trainer encodes with (encoder ==
  decoder).

## Tests

`delta_test.py` covers the encoder structurally (skips if numpy/zstandard/xxhash
are absent). The full byte-for-byte round-trip against slime's real decoder is
asserted by `control_plane_test` on Modal.
