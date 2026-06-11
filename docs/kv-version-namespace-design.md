# Cross-request weight sync: version-namespaced KV cache

Status: design (verified against sglang v0.5.12.post1 + slime patch, and sglang master).

## Goal

Allow a rollout server to commit a new weight version without aborting, retracting, or
draining in-flight requests:

1. In-flight requests pause briefly during the commit, then resume decoding on their
   existing (stale) KV — no abort, no retraction, no re-prefill.
2. Requests admitted after the commit must miss every KV entry produced under older
   weight versions.
3. Stale KV drains naturally via LRU once unreachable, with an optional deterministic
   sweep once the last request from an old version finishes.

Non-goals: LoRA, PD-disaggregation, hierarchical cache (HiCache/L3), cpp radix tree.
These either need separate treatment (noted below) or are not used by the rollout fleet.

## Foundation: what sglang already provides

All anchors below were verified in the pristine v0.5.12.post1 wheel, re-checked against
the slime patch (`docker/patch/latest/sglang.patch`), and exist near-identically on
master (so the work upstreams cleanly).

- **`RadixKey.extra_key` is a first-class cache namespace.** Every children-dict key in
  the radix tree embeds `extra_key` (`radix_cache.py` `child_key`: `(extra_key, plain)`
  tuples), so different `extra_key` values form fully disjoint subtrees for any
  `page_size`, including the in-batch dedupe tree in `schedule_policy.py`. This is the
  exact mechanism LoRA uses to isolate KV across adapters (`lora_id` is concatenated
  into `extra_key` at `Req` construction). The plumbing
  `GenerateReqInput.extra_key → TokenizedGenerateReqInput → Req.extra_key` exists in
  0.5.12.post1 and is untouched by the slime patch.
- **Running decodes never consult the tree.** A running request's KV addressing is its
  `req_to_token_pool` row plus directly allocated `token_to_kv_pool` slots; the tree
  only pins the matched prefix via `lock_ref` (never evictable) and is consulted at
  admission and at `cache_finished_req`/`cache_unfinished_req` boundaries. Namespacing
  old entries away therefore cannot break in-flight requests.
- **`pause_generation(mode="in_place")` + `update_weights_from_disk(flush_cache=false)`
  + `continue_generation`** already implements pause-swap-resume on stale KV: the
  scheduler short-circuits its loop while still servicing control messages, the
  tokenizer blocks new admissions (`is_pause`) and bypasses the writer lock for the
  update. What it does *not* do today is prevent cross-version KV reuse (the tree is
  fully reusable across the commit), report start/end versions, or synchronize with the
  in-flight overlap forward (see "Mandatory engine fixes").
- **`cache_finished_req(req, is_insert=False)`** (used by retract/abort) frees a
  finishing request's computed KV instead of inserting it — the primitive for the
  strictness knob below.
- **`SGLANG_RADIX_FORCE_MISS`** provides a global force-miss kill switch (covers both
  the main tree and the in-batch dedupe tree).

## Design

### 1. Scheduler-side version state and admission stamping

- The scheduler keeps an integer `current_weight_version`, updated from the existing
  (currently unused there) `recv_req.weight_version` field when an update succeeds.
  Update handlers run inline in the scheduler thread, totally ordered with
  `handle_generate_request`, so stamping is deterministic. Contract: monotonic integer
  rendered as string on the wire; bump only on full success (target *and* draft).
- Every request is stamped at admission, in `Req.__init__` (one chokepoint — covers the
  session-controller construction path too):
  - `req.weight_version_start = current_weight_version`
  - `req.extra_key = compose(user_extra_key, version)` with an unambiguous,
    delimiter-terminated encoding. Put the version segment in a fixed position and
    parse from the module that composes it — `lora_id` is appended delimiter-free after
    `extra_key` today, so the composition must be collision-proof against that.
- TP/DP consistency holds for free: requests and update messages travel one ordered
  ZMQ channel and are broadcast as one ordered list per iteration to all ranks, so all
  ranks stamp identically. (Under dp attention, control messages are processed after
  work messages within one poll batch — rank-consistent, just documented behavior.)

### 2. Commit sequence

```
sidecar: prefetch artifacts to local NVMe/page cache; zstd-decompress + checksum there
sidecar: POST /pause_generation {"mode": "in_place"}
sidecar: POST /update_weights_from_disk {load_format: delta, model_path: <local copy>,
         files: [...], flush_cache: false, weight_version: "<v+1>"}
sidecar: POST /continue_generation        # in try/finally with the pause
```

- `flush_cache: false` is mandatory and must be explicit (default is `true`, and
  `flush_cache_after_weight_update` *asserts* flush success — with live requests that
  assert kills the scheduler process via SIGQUIT).
- The version bump, the namespace switch for new admissions, and the multimodal
  embedding-cache clear all happen in the same scheduler-thread handler.

### 3. Namespace semantics

- New requests (stamped v+1) structurally cannot match wv=v subtrees.
- In-flight requests keep their admission-time `extra_key` for all subsequent
  chunked-prefill inserts, so their KV stays self-consistently in the old namespace.
- A request that crosses a commit produces mixed-version KV inside its own sequence —
  the accepted PipelineRL-style impurity. Its inserts land in its old namespace:
  invisible to new-era requests, reusable only by same-era stragglers.
- Retraction of a stale request stays consistent: freed-without-insert KV, re-prefill
  re-matches whatever remains of its own namespace, recomputes the rest under new
  weights. Bookkeeping (`cache_protected_len`, `lock_ref`) holds; verified against the
  retract paths.

### 4. Draining

- **Natural**: old-namespace nodes are never matched again, so their access times go
  stale and global LRU evicts them first under pressure. Run rollout servers with the
  LRU eviction policy (priority-aware policies can keep stale nodes resident).
- **Deterministic sweep**: the scheduler keeps `live_request_count[version]`. On commit
  success and when a stale version's count reaches zero, sweep `root.children`
  (namespaced keys are `(extra_key, first_page)` tuples — one O(#children) dict scan)
  and evict unlocked subtrees of namespaces **with zero live requests** (not merely
  "older than current": that over-evicts shared prefixes of still-live older versions,
  e.g. GRPO same-prompt siblings). Implement by reusing the existing
  evict/`_delete_leaf` machinery (or biasing eviction priority per namespace) so
  `evictable_size_`/`protected_size_` accounting stays correct; skip any subtree
  containing `lock_ref > 0` and retry on the next trigger. Note the patch's
  `dec_lock_ref` assert→break hunk silences the crash that a buggy sweep would
  otherwise produce — add an explicit `lock_ref` guard + counter to keep the invariant
  observable.
- **Strictness knob (optional)**: when a finishing request has
  `weight_version_start != current`, call `cache_finished_req(is_insert=False)` so its
  newly computed KV is freed rather than inserted. Its previously inserted chunk
  prefixes remain until the sweep/LRU — the knob reduces, not eliminates, stale
  insertion.

### 5. Version metadata on responses

Stamp `weight_version_start` (admission stamp) and `weight_version_end` (scheduler
current at emission) per output batch: `Req.weight_version_start` + parallel lists on
`BatchTokenIDOutput`/`BatchStrOutput`, threaded through the detokenizer into
`meta_info`. Three dataclasses + two stamp sites. Streaming chunks become
self-describing (pre-commit chunks end=v, post-commit end=v+1; final chunk
authoritative — exact given the drain fix below). Abort-path outputs carry no stamp;
consumers must treat missing version metadata as discard. This replaces the sidecar's
response-rewrite stamping, which is approximate under cross-request commits.

### 6. Exact-version pinning interplay

Strict (`exact_version`) requests must not cross a commit. The gate is: commits wait
for `exact_inflight` (summed over all versions) to reach zero **and** atomically block
new exact admissions from gate-check until `continue_generation`; non-strict requests
cross freely. Since every commit drains exact traffic, at most one exact version is
live at a time. Hardening: decrement `exact_inflight` in a `finally` that also covers
client disconnects, and add a timeout/abort-and-retry policy so a long exact request
cannot head-of-line-block commits indefinitely.

## Mandatory engine fixes (ship first, independently useful)

1. **Overlap drain + stream sync before weight mutation.** 0.5.12.post1 uses the same
   single-thread two-stream overlap design as master. `pause(in_place)` returns without
   processing `result_queue`, and the delta apply mutates weights with no
   event/wait against `forward_stream` — a forward launched the iteration before the
   pause can still be executing when weights mutate, corrupting that batch's logits.
   Fix (~6 lines, in the scheduler update handlers, mirroring what
   `pause(mode="retract")` already does): while `result_queue` non-empty,
   pop-and-process (this waits `copy_done` and flushes tokens to clients), then
   `forward_stream.synchronize()`; cover spec_v2 delayed-sample/`future_map` state.
   Without this, *both* V0 and V1 are unsound; the deployed quiesce flow is safe only
   because the sidecar drains everything first.
2. **O(delta) apply.** Today `_decode_delta_one_param` materializes a full-shape NaN
   tensor per param and the patched `copy_` masked-scatters over every param byte —
   apply is O(full model). Acceptable while servers are idle during commits; under
   in-place commits it becomes an ITL stall (~0.5–1s for a 4B model, tens of seconds
   for 355B-class). Rewrite: precompute once per model a HF-name → (local tensor,
   TP-shard slice, fusion/transpose transform) table; apply deltas as direct
   `index_copy_` scatters into param storage. Target pause: ~50–200ms for 250MB deltas,
   ~1–3s for 5GB. Keep the NaN/`load_weights` path as fallback for exotic layouts.
3. **Pause watchdog.** If the sidecar dies between pause and continue, the engine is
   paused forever while health stays green. Engine-side max-pause auto-continue (loud
   metric) + sidecar try/finally.
4. **Multimodal embedding cache.** `MultiModalStaticCache` keys vision-encoder
   embeddings by content hash only and survives commits — clear (or version-key) it in
   the update handler. Only matters when the vision tower trains; 3-line change.

## Staging

- **V0 (sidecar-stamped namespaces)**: sidecar injects composed
  `extra_key` into proxied `/generate` bodies and switches commits to
  pause(in_place)/no-flush/continue. Requires engine fix #1 (added to the existing
  sglang patch) but no other engine changes. Limitations (all verified): stamping at
  sidecar-admission creates a mislabel window around commits (old-era impurity/wasted
  cache only — never new-era contamination, given single tokenizer worker and the FIFO
  tokenizer→scheduler channel); start/end stamps approximate; draining is LRU-only;
  traffic bypassing the sidecar lives in the unversioned namespace (stamp or
  force-miss it).
- **V1 (engine-stamped)**: admission stamping in `Req.__init__`, scheduler version
  state, per-batch start/end metadata, deterministic sweep, strictness knob, exact-pin
  gate. This is the production design and the upstreamable patch; write it against
  master first and backport (anchors verified on both; 0.5.12 delta is mostly line
  offsets).

## What this buys, by rollout mode

- **Strict push (deployed today)**: in lockstep GRPO essentially no main-batch traffic
  crosses commits; the win is that commits stop being blocked behind over-generation
  zombies, partial-rollout stragglers, and eval traffic, and the full-tree flush
  disappears (a commit no longer costs every other request its prefix cache — though
  new-era requests still cold-miss by design).
- **Non-strict (min_required / null) PipelineRL-style**: the full payoff — requests
  cross commits freely with correct per-sample (start, end) metadata for buffer-side
  off-policy filtering. Requires slime additions: a `min-required` policy emission, a
  buffer filter on `(end - start)` and on trainer-version lag (the dynamic-filter hook
  exists), per-segment versions for partial rollouts (`Sample.weight_versions` already
  exists, currently unread).
- **KV-drift budget**: optionally abort requests with
  `current_version - weight_version_start >= K` (analogous to the protocol's rebase
  drift budget) so decode isn't spent on samples the buffer will drop.

## Observability (minimum bar)

- Engine: `current_weight_version` gauge; radix tokens by namespace (counted at
  insert/evict); sweep duration + evicted-bytes-per-namespace; stale in-flight request
  count by version lag; commit pause duration.
- Sidecar `/server_info`: last commit pause ms, prefetch ms/bytes, commits total, last
  error.
- Trainer: fleet version-lag percentiles (poll `/server_info`), rate of samples with
  `start != end` per rollout.

## Known constraints

- Plain Python `RadixCache` only. `RadixCacheCpp` drops `extra_key` entirely; HiCache
  host/L3 page hashes are token-only (cross-version aliasing) — keep both disabled on
  rollout servers, or fold the namespace into page hashes before enabling.
- EAGLE/spec decoding: draft + target update together in the same handler and draft KV
  shares radix-managed slots, so versioning composes; the drain fix must cover spec_v2
  state (incl. the draft-extend `plan_stream` sync in `eagle_worker_v2`).

### Spec/HiCache extension notes (verified against master, 2026-06)

- DFLASH (blockwise-parallel drafting, now upstream): composes — its draft KV is reused
  cross-request only via radix prefix hits, which the namespace partitions. But
  `DFlashWorker` has no `update_weights_from_disk`; `__getattr__` delegates to the
  target worker, so draft weights are silently never updated (upstream bug). Fix before
  any dflash rollout fleet; draft acceptance otherwise decays across commits.
- HiCache: host tier already namespaced (inherits python `RadixCache` child keys). Only
  the storage/L3 hash chain is token-only — seeding the chain root with `extra_key`
  (`get_hash_str`/`hash_page`, ~30 LOC) closes it, so enabling HiCache later is cheap.
- RadixCacheCpp stays disabled (moderate lift: per-namespace tree instances).
  PD-disagg unverified; it ships KV by page hash, so likely needs the same hash seeding.
- Memory: locked stale prefixes shrink the effective pool until their requests finish;
  prefill admission throttles (budget excludes protected size) rather than crashing,
  and retraction prefers least-progress (new) requests — bounded unfairness right
  after commits under high occupancy. Reserve headroom or cap concurrent stale tokens
  if this bites.
