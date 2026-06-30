# local_disagg — minimal pool-claim harness

The smallest possible disaggregated-rollout setup: **no Modal, no slime/miles,
no GPUs.** A local-filesystem bulletin board, an in-memory rollout pool, and a
trainer-as-writer, wired with the *same* claim / advance / reconcile primitives
the production cookbooks use. It exists so the pool-ownership invariants can be
exercised and iterated on in milliseconds.

## The model

One **trainer** call owns one **run**, which owns one **pool epoch**:

```
trainer.claim()      -> board.claim(run_id)      # write <run_id>/weight_v000000 (empty)
trainer.publish()    -> board.advance(run_id, N) # write <run_id>/weight_vN, monotonic
replica.reconcile()  -> WeightSyncManager.sync_to()  # converge to latest (reset on run switch)
```

- The **bulletin board** is the single source of truth: a self-identifying
  `latest` pointer `<run_id>/weight_v{N}`.
- The **trainer** is the single writer. `claim` resets the pool to base for a
  fresh run; `publish` advances monotonically within the run. Both go through
  the board's guarded writers (`stitch.protocol.decide_pointer_move`), so a
  reused `run_id` or a non-monotonic publish raises `PointerRewind` instead of
  serving stale weights.
- Each **replica** is a pure reconciler — it reads `latest` and converges
  (replay the chain forward, or reset-to-base then replay on a run switch).

`run_id` is a per-launch epoch/fence token (default: a fresh `uuid4`). A restart
is just a new epoch that claims and resets the pool — there is no special restart
path, and a crash-restart can never reuse a `run_id` to resurrect a dead pointer.

## Invariants (see `harness_test.py`)

- A claim resets every replica to base (v0) under the new run's id.
- Within a run the pool converges to each published version, in order.
- A new run forks at base: the pool resets even from a higher prior version.
- Re-claiming a run already at the pointer (a reused `run_id`) is rejected as a
  rewind; the correct restart mints a fresh `run_id`.
- A non-monotonic publish within a run is rejected.
- A late-joining (cold / scaled-up) replica reconciles to the *current* run.

## Run it

```bash
uv run python -m pytest cookbook/local_disagg/harness_test.py -q
```

To iterate by hand, build a board + trainer + pool and step through claim /
publish / reconcile; see `harness.py` for the (tiny) surface.
