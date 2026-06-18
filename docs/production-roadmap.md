# Disaggregated rollout weight sync: production roadmap

Findings below were produced by a full review of this package, the
`slime_delta_disagg` example, the slime delta branch (trainer side + the sglang
0.5.12.post1 patch), and sglang master, with each high-stakes claim
adversarially re-verified against the code. Companion doc:
[kv-version-namespace-design.md](kv-version-namespace-design.md) for the
cross-request weight-sync design.

Severity reflects impact on the deployed strict-mode configuration
(exact request-version pins, quiesce-all commits, publish-only trainer).

## P0 — verified correctness/liveness failures

1. **Commit-window admission hole** (fixed in this change). The quiesce check in
   `WeightSyncManager._sync_once` was evaluated once, after which the sync task
   suspended on the flush/apply network calls with `current_version` still
   stale and no admission gate. A strict `exact=N` request arriving in that
   multi-second window validated against N, was pinned, and was served by the
   engine on N+1 weights (sglang queues the generate behind the in-flight
   update). slime never checks `weight_version_end`, so wrong-version rollouts
   were silently accepted. Fix: `_committing` gate set atomically with the
   quiesce predicate; the authoritative policy check + version capture moved
   inside `request_context`, after the gate, under the same lock
   (`sync.py`). Note the previously-suspected validate→pin TOCTOU does *not*
   exist (asyncio no-yield fast paths make the old sequence atomic) — the gate
   closes the real window and removes the fragility.

2. **Zombie generations stall every commit** (mitigated in this change; engine
   abort still best-effort). slime's endpoint-mode abort is `cancel-only`
   (`slime/utils/arguments.py:1859-1862`): client tasks are cancelled, the TCP
   connection closes, but uvicorn/starlette never cancel a non-streaming
   handler on disconnect, so the sidecar held `active_requests` until sglang
   finished every abandoned generation (up to `max_tokens`). Every overlapping
   commit stalled at the quiesce point while the next rollout's pinned
   requests burned their 240×1s retry budget — stall > budget crashes the run.
   GPU is also wasted on every over-generated sample. Fix: the proxy now
   injects a `rid`, races the upstream call against `http.disconnect`, and on
   disconnect cancels the upstream call and POSTs `/abort_request {rid}`
   (`servers/sglang.py`). Remaining: slime-side, prefer an abort strategy that
   broadcasts `abort_request` to containers via tunnel URLs at rollout end.

3. **Gateway-reachable engine control routes** (fixed in this change). The
   catch-all proxy forwarded `/update_weights_from_disk`, `/flush_cache`,
   `/pause_generation`, etc. from the body-blind gateway to the engine,
   bypassing all manager bookkeeping; any stray/buggy caller could mutate
   weights or state behind the sync manager. Fix: `BLOCKED_ROUTES` 403 in the
   proxy.

4. **Engine-side post-apply flush assert can kill the container** (fixed in
   this change). `apply_manifest` omitted `flush_cache`, which defaults to
   `true`; `flush_cache_after_weight_update` *asserts* flush success, flush
   fails whenever the scheduler is not fully idle, and the AssertionError
   SIGQUITs the whole sglang server (verified in the 0.5.12 wheel:
   `scheduler_update_weights_mixin.py:45-50`, `scheduler.py:4043-4046`).
   Guarded today only by the sidecar quiesce + tokenizer RW-lock; reachable via
   narrow disconnect/abort races. Fix: send `flush_cache: false`; the manager's
   pre-apply `GET /flush_cache` (which returns 400 when busy and aborts the
   sync attempt cleanly) is the flush of record.

5. **Trainer-side `DeltaState` cross-stream use-after-free** (open; slime
   repo). `update_snapshot_async` enqueues D2H snapshot copies on `d2h_stream`
   reading TP/EP-gather outputs allocated on the default stream; the Python
   refs die at `_pipeline_pass` rebind and nothing orders the D2H read before
   allocator reuse (no `record_stream`; `flush_snapshot` syncs only at end of
   pass; `_seed_snapshot` has no host sync at all). Silent snapshot corruption
   → wrong diff baseline → density blowups and rare silently-stale receiver
   weights, defeating the lossless guarantee. Fix:
   `tensor.record_stream(d2h_stream)` per copied tensor, or hold chunk refs in
   a deferred-free list released on a per-chunk d2h event
   (`update_weight_from_distributed_delta.py:305-319,624-629,687`). The
   symmetric prefetch hazard is incidentally safe (host-blocking
   `searchsorted().tolist()` drains the default stream first) — keep that sync
   if encode changes.

6. **Resume/restart version collisions** (open; slime + package). The
   trainer's `weight_version` counter is in-memory only and resets to 0 on
   restart (`update_weight_from_distributed.py:46`), while
   `start_rollout_id` resumes from the checkpoint and the bulletin board still
   holds `latest=K`: republished `weight_v000001` overwrites immutable
   artifacts; warm sidecars at K ignore lower versions while pinned requests
   spin; manifest `run_id` is written but never validated. Modal preemption
   auto-retries the clustered `train()` input, so this fires *without operator
   action*. Fix: persist/restore the counter with the checkpoint (or
   fast-forward from `read_latest()` at startup and refuse to publish
   `<= latest`), put a run epoch in `latest.json`/manifests, and make sidecars
   treat an epoch change explicitly (exit and let Modal replace them).

7. **`reset_bulletin_board` poisons a running pool** (open). Sidecars have no
   downward path (`queue_sync` ignores targets ≤ current; `_sync_once` walks
   forward only), `min_containers` keeps the stale containers alive, and every
   pinned request then gets `WeightVersionTooOld` for the full retry budget.
   Same epoch mechanism as (6): on observing `latest < current` or an epoch
   change, the sidecar should exit(1) so Modal replaces the container.

8. **`startup_sync` busy-loops forever on persistent errors** (open).
   `sync_to` swallows all exceptions into sticky `ERROR`; `startup_sync`
   re-calls it in a tight loop (volume reload per iteration) while the FastAPI
   lifespan blocks, so uvicorn never binds the port and the container looks
   dead until the 40-minute startup ceiling, then crash-loops — one bad/missing
   manifest poisons the fleet. Fix: bounded retries with backoff, then raise
   (fail fast → container replacement); or serve immediately with sync state
   surfaced and let policy rejections gate traffic.

## P1 — protocol features the implementation is missing

- **Recovery anchors, retention, GC** (protocol §3/§12/§13; the largest gap).
  Nothing publishes or consumes anchors: cold start replays the *entire*
  transition chain from v0, so container cold-start time and volume size grow
  linearly with training steps, and the 40-minute Modal startup ceiling turns
  that into a guaranteed future fleet-wide crash loop. Publish a recovery
  anchor (delta_accum or checkpoint) every N versions, implement the sidecar
  recovery path (`manifest.recovery` is currently write-only), add a retention
  window + GC job, and alarm on replay duration.
- **Typed version-aware retry handling** (protocol §6). The slime endpoint
  branch now emits both `exact_version` and `min_required_version`, but `_post`
  still retries every error identically (including non-retryable
  `WeightVersionTooOld` and arbitrary 4xx) until the budget kills the whole
  rollout. Parse the typed 409 error, fail fast on too-old, back off on
  not-ready.
- **`weight_version_start/end` consumption** (protocol §6.3). The sidecar now
  reports them correctly, but nothing in slime reads
  `sample.weight_versions`; off-policy handling is version-blind. Needed for
  any non-strict mode (see the KV design doc's rollout-mode section).
- **Manifest integrity**: `Artifact.checksum`/`size_bytes` and
  `protocol_version` are written but never verified by readers; the engine
  receiver also ignores the safetensors `current_version` metadata (no
  base-version precondition on apply — protocol §14 invariant unenforced at
  the engine; today only the sidecar's chain check enforces it).
- **Elastic concurrency** (verified high): endpoint mode pins
  `rollout_num_engines=1`, so the client semaphore and connection pool cap at
  `sglang_server_concurrency` (64) total across the pool — the autoscaler
  never sees demand beyond ~1 container. Make client concurrency independent
  of engine count (explicit flag), and revisit `@modal.concurrent` /
  `--max-running-requests` sizing together.

## P2 — cross-request weight sync (KV versioning)

See [kv-version-namespace-design.md](kv-version-namespace-design.md). Ordered
steps:

1. Engine patch: overlap drain + `forward_stream.synchronize()` at the top of
   the update handlers (prerequisite for everything; fixes a latent race in
   the deployed flow too).
2. Sidecar V0: extra_key version stamping + pause(in_place)/no-flush/continue
   commit + artifact prefetch to local disk before the pause.
3. Engine V1: admission stamping, scheduler version state, per-batch
   start/end metadata, namespace sweep, pause watchdog, multimodal embedding
   cache clear.
4. O(delta) apply rewrite (precomputed name→shard-slice `index_copy_`),
   required before in-place commits on large models.
5. slime: buffer filtering / off-policy handling to actually exploit commits
   that cross requests.

## P3 — platform/ops hardening (Modal-specific)

- **Health is TCP-connect-only** (verified: Flash heartbeats only check the
  port). A sidecar in sticky `ERROR`, or one whose engine died, serves traffic
  forever. Map unrecoverable states to the only signal the platform watches:
  exit the sidecar process (container gets replaced); make `/health` reflect
  engine + sync state for humans.
- **Shutdown ordering defeats draining**: `@modal.exit` SIGTERMs sidecar and
  engine immediately, so the Flash `exit_grace_period` elapses with backends
  already dead — every scaledown/redeploy kills in-flight generations at t=0.
  Drain (bounded) before terminating.
- **Trainer cluster watchdogs**: rank!=0 nodes `sleep` forever; a dead Ray
  worker hangs rank 0 until the 24h timeout while the warm fleet idles.
  Followers should health-check the Ray head and exit on loss; rank 0 should
  enforce node-count and progress deadlines.
- **Trainer singleton**: nothing stops two concurrent `train()` calls writing
  the same board (last-write-wins on v2 volumes). Take a lease (modal.Dict or
  a lease file validated by publish hooks).
- **Best-effort wake hardening** (fixed in this change): `wake_targets` now
  fans out in parallel and the publish hook no longer dies on Modal
  control-plane/env errors after `latest.json` has advanced.

## P4 — simplification / code health (highest-signal smells)

- `trainers/slime.py` imports the Modal provider directly and reads env vars
  deep in the publish hot path — inject a provider/notifier object instead;
  validate config at trainer init, not first publish (`pyproject` extras don't
  even include `modal` for the `slime` extra).
- `delta_dir`/`delta_root`/`bulletin_root`/volume-name are plumbed through 4
  layers with the dir→root derivation duplicated 3×; resolve once in slime
  arg post-processing and pass explicitly.
- `trainers/slime.generate_rollout` is now only a compatibility wrapper; new
  configs should use upstream slime's rollout function plus
  `custom_rollout_request_hook_path`. Keep an eye on any remaining configs that
  still depend on the wrapper before deleting it.
- Dead surface: `volume_committer`, `wake_targets_aio`,
  `WeightVersionPolicy.to_payload`, the never-constructed `exact_only` commit
  policy, write-only manifest fields (`recovery`, `base_model`,
  `protocol_version` parsed-but-unvalidated), `discover_flash_targets`'s
  discarded `Cls.from_name`, `publish_delta_version`'s `return []`.
- `versioned_routes` is fake configurability (gating uses the set, metadata
  injection hardcodes the two paths) — drive both from one table.
- Proxy data-plane: per-request `httpx.AsyncClient` (no pooling), double JSON
  decode/encode of MB-scale logprob bodies on one event loop, upstream
  headers dropped, `stream=true` silently broken, `globals()["Request"]`
  import hack. Long-term the data plane should disappear into the engine
  (protocol fields in `/generate`); short-term: shared client, raw-bytes
  passthrough for non-versioned routes.
- Naming: pick one vocabulary across protocol/doc/code
  (PREFETCHING/PREPARING/COMMITTING vs prepare/commit vs
  flush_cache/apply_manifest); PREFETCHING currently prefetches nothing.
- modal_train: module-level config-from-env at import time (deploy-time
  defaults silently leak across experiments — e.g. an experiment omitting
  `DELTA_VOLUME_NAME` inherits the default experiment's volume while the
  Server always mounts the default's); 4 near-identical gateway resolvers;
  smoke-test DI theater.
- Tests assert mock call-shapes and implementation details while the
  concurrency core and proxy had no coverage (partially addressed in this
  change: commit-gate, pin/violation, blocked-route, and 409 tests added);
  still missing: wake-during-sync, ERROR recovery, chain catch-up with gaps,
  an ASGI test with a live fake upstream for meta injection and disconnect
  handling.

## Corrected/retired suspicions (verified false — don't chase)

- `torch.cat` mixed-dtype bucket break: refuted; cat type-promotes, offsets
  are element-based, bf16→fp32→bf16 is bit-exact. Only effect is wire-size
  inflation for chunks containing fp32 tensors (and `wire_bytes` undercounts
  promoted co-bucketed chunks).
- nccl+`deltas_zstd` misdecode: refuted; the NCCL path never zstd-decodes and
  treats DELTAS_ZSTD == DELTAS for positions. Cosmetic mislabel; forbid or
  warn on the combination.
- Sidecar validate→pin TOCTOU as originally stated: refuted (asyncio fast
  paths); the real hole was the commit-window gap (P0.1, fixed).
- Publish-path bulletin corruption on trainer crash: refuted; ordering
  (artifacts committed → manifest → latest → commit) plus atomic writes means
  partial states are invisible or self-healing. Residual risks are
  availability-level.
- Adapter "ignores flush failure": refuted; `/flush_cache` returns 400 on
  failure and the adapter raises on non-200/404.
