# Heterogeneous low-precision rollout plan

## Goal

Train one Qwen3.6-35B-A3B BF16 policy on the Xiaomi MiMo-V2.6 code data while
sampling from a heterogeneous rollout fleet:

- Hopper replicas serve block-wise FP8 weights.
- Blackwell replicas serve NVFP4 weights with W4A16 execution.
- Every serialized weight view is derived from the same BF16 trainer version.
- The trainer remains BF16; fake FP8 or NVFP4 QAT is not part of the base
  experiment.

The design is not specific to these two formats. A weight view is a named
serialized representation of the policy, such as BF16, FP8, MXFP8, or NVFP4.

The first two algorithm experiments are:

1. GRPO without R3 or importance-sampling correction.
2. GRPO with TIS and R3 as the established correction control.

A later experiment will reuse the same systems and configuration boundary.

## First-principle model

There is one trainer model and multiple independent rollout representations of
each trainer version. The trainer tensor is gathered and mapped to Hugging Face
semantics once. Each configured view then encodes those mapped tensors, compares
them with its own previous encoded snapshot, and publishes its own delta stream.

```text
trainer version N (BF16)
    |
    `-- gather and map each tensor once
          |-- BF16 encoder  -> updates/bf16/weight_vN
          |-- FP8 encoder   -> updates/fp8/weight_vN
          |-- MXFP8 encoder -> updates/mxfp8/weight_vN
          `-- NVFP4 encoder -> updates/nvfp4/weight_vN
```

Views do not form a distributed transaction. When one view finishes, it is
published and its rollout pool starts updating immediately. It does not wait
for another view. The overall Miles call may still wait for all configured work
to drain before releasing its source buffers and returning; that is resource
lifetime, not a publication barrier.

The implementation should compose existing primitives:

- Miles already gathers Megatron tensors, maps them into HF names, encodes
  supported checkpoint formats, and publishes disk deltas.
- Stitch already provides version pointers, monotonic reconciliation,
  stage/commit admission, checksums, pool readiness, and session affinity.
- SGLang already loads and applies each supported homogeneous representation.

Phase 1 should instantiate those primitives once per view. It should not add a
second weight-update protocol or modify SGLang.

The ownership boundary is deliberately small:

- Miles maps trainer tensors, encodes views, writes each view's normal delta,
  and invokes the existing post-write hook with the producing view.
- A Stitch Store scopes the existing artifact and pointer operations to one
  view; Publisher remains a single-view publisher.
- The post-write hook selects only the rollout pools for that view and uses the
  existing Publisher to advance and wake them.
- Each rollout replica's existing reconciler follows exactly one Store and
  reports the version it actually serves.
- The existing fleet router owns session-to-pool affinity and source
  attribution. It does not own weight publication.

No component coordinates a cross-view commit.

## Correctness invariants

1. BF16 trainer tensors are gathered once for each published trainer version.
2. Every view is encoded from the gathered BF16 tensor. A delta is never used
   as the input to another quantizer.
3. Each view owns its canonical checkpoint, previous encoded snapshot, delta
   directory, version pointer, and publication lifecycle.
4. A view advances only after its own artifact, index, checksums, and durable
   publication are complete. There is no shared latest pointer and no
   all-views-ready barrier.
5. `weight_vN` has the same logical trainer meaning in every view, but views may
   expose different latest versions at the same wall-clock time.
6. A rollout pool reads and applies only its configured view. It must never
   label version `N - 1` weights as version `N`.
7. A session remains on one pool and one view. The router must not silently
   switch precision during a session.
8. Every sample records the pool/view and the actual weight version that served
   it. Existing async staleness policy decides whether the trainer may consume
   it.
9. Failure behavior remains the existing single-view behavior. A failed view
   does not advance; an engine that fails to apply an otherwise valid update is
   removed or recycled by the existing lifecycle. A sibling view already
   published at that version is not rolled back.
10. Resume considers only the views selected by the resumed configuration. At
    resume version `N`, every selected view must have a complete lineage through
    `N`; historical but unselected views do not constrain the resume point.
11. The resumed trainer conversion for each selected view must match that
    view's durable checksum state at `N` before Miles may publish `N + 1`.
12. When trainer-versus-rollout mismatch metrics are enabled, they are
    observational and must not change the loss.

## Storage contract

Miles owns immutable update artifacts. Stitch owns mutable pointers:

```text
<run>/updates/fp8/weight_v000010/
<run>/updates/nvfp4/weight_v000010/

<run>/hf_checkpoints/weight_v000009/fp8/.complete
<run>/hf_checkpoints/weight_v000009/nvfp4/.complete

<run>/latest/fp8     # points to fp8 v10
<run>/latest/nvfp4   # may still point to nvfp4 v9
```

Each view directory is a normal instance of the existing single-view disk-delta
layout. There is no parent transaction manifest, projection directory, or
symlink lineage.

Periodic HF saves use the same view encoders. A view's nested `.complete`
marker is published only after the distributed checkpoint write succeeds, so a
new replica may load that exact view and apply only its remaining delta tail.
Without a complete saved view, the replica falls back to the static v0
checkpoint and the full delta lineage.

## Phase 1: complete the primitives

### 1. Miles: generic multi-view disk-delta publication

Add a generic mapping from view name to canonical rollout checkpoint. The view
name is an opaque identifier; it is not an enum of known precisions. Miles
derives the existing encoder and canonical tensor layout from that checkpoint's
configuration.

For each trainer tensor:

1. Gather it once with the existing placement logic.
2. Map it once into the model's HF tensor names.
3. Feed the mapped tensor to each configured view's existing encoder.
4. Feed each encoded result to an independent instance of the existing
   disk-delta protocol rooted at `updates/<view>`.

Each child protocol keeps its normal post-write hook. Consequently it publishes
`updates/<view>/weight_vN` as soon as that view is complete. The parent only
coordinates the shared input stream and resource lifetime; it writes no parent
manifest and owns no shared pointer.

Keep memory bounded by the existing bucketed iterator. At most the current
mapped bucket and the bounded per-view encode/diff work should be live. Do not
materialize a full BF16 export or every complete encoded model in memory.

Reuse the canonical encoders used by Miles' checkpoint-conversion tools for
disk artifacts. In particular, FP8 disk-delta bytes must match the canonical
FP8 checkpoint layout. Do not globally change the existing direct-transfer
representation merely to support disk views.

The existing homogeneous path must remain unchanged when no view mapping is
configured.

When HF saving is enabled, reuse the same view iterator to write one complete
HF checkpoint under `weight_vNNNNNN/<view>`. This is part of the multi-view
policy: ordinary single-view `save_hf` keeps using `hf_checkpoint` as its target
layout. Do not introduce a second generic checkpoint-layout option.

### 2. Miles: rollout-source attribution

Use the existing session response path to copy the rollout source into sample
metadata. Every pool replica's sidecar stamps its `<pool>:<view>` source into each
choice's `meta_info`, beside the weight version it served, so a one-pool fleet with
no router is attributed too. The source travels in the body because the Modal
gateway in front of a pool drops custom response headers. In a mixed fleet the
router's `X-Stitch-Rollout-Source` header names the same pool and takes precedence.
This is transport metadata only; it must not affect reward, filtering, correction,
or loss.

Do not add a new mismatch-metrics primitive. Pure GRPO disables TIS and mismatch
metrics. Corrected experiments use Miles' existing TIS/mismatch contract. R3
remains independent replay data for discrete MoE routing.

### 3. Stitch: select one view per rollout pool

Extend `RolloutPoolConfig` with an optional opaque `weight_view` name. A recipe
provides a generic mapping such as:

```python
ROLLOUT_WEIGHT_VIEWS = {
    "bf16": BF16_ROLLOUT_CHECKPOINT,
    "fp8": FP8_ROLLOUT_CHECKPOINT,
    "nvfp4": NVFP4_ROLLOUT_CHECKPOINT,
}
```

Every configured pool references one key from that mapping. Stitch uses the
selected checkpoint as that pool's boot checkpoint.

Make the existing store view-scoped:

- `latest()` reads `<run>/latest/<view>`;
- `publish()` advances only `<run>/latest/<view>`;
- `materialize(vN)` returns `<run>/updates/<view>/weight_vN` and its sibling
  lineage directly.

Apply the same contract to Modal Volume and S3 stores. Do not project,
duplicate, or symlink version directories.

The trainer's existing post-write hook receives the producing view, publishes
through that view's store, and wakes only rollout pools configured for that
view. Initial claim follows the same rule: every view pointer is claimed at v0
against only the pools booted from that view's canonical checkpoint.

The Miles request hook runs before the fleet router has selected a pool. It
therefore snapshots every view's independent latest pointer and sends that map
as internal routing metadata rather than synthesizing a global minimum or
maximum. After affinity selects a pool, the router injects that pool's latest as
the request's `min_version` and removes the internal metadata. A replica that
has already advanced remains eligible; a lagging replica returns the existing
retryable admission response. The response is still stamped with the version
actually served, and normal async staleness filtering remains authoritative.

### 4. Stitch: preserve pool identity and session affinity

Keep the existing heterogeneous router lifecycle. Add only the information
needed to:

- assign a session deterministically to one pool;
- keep the session on that pool without cross-view fallback; and
- return the selected pool/view as response metadata for Miles to record.

Do not replace the router with a subprocess or introduce a second server
lifecycle.

### 5. Minimal configuration contract

Phase 1 adds only two data fields:

- Miles receives `view name -> canonical checkpoint path`.
- Each Stitch `RolloutPoolConfig` optionally names one view.

There is no precision enum, view-specific class hierarchy, parent checkpoint,
shared manifest, readiness barrier, or new publication callback. A view name is
just a validated path component. Its checkpoint configuration determines the
encoder.

### 6. Phase 1 validation

Unit tests must prove:

1. Arbitrary view names and checkpoint-derived encoders work; the API is not
   restricted to FP8 and NVFP4.
2. A trainer tensor is gathered/mapped once and fanned out into distinct
   encoded tensor names, shapes, dtypes, and scales.
3. Every view maintains its own previous snapshot and computes its delta
   against that snapshot.
4. Completing FP8 v1 advances only `latest/fp8`; NVFP4 may complete and advance
   later. No parent artifact or shared pointer is required.
5. Failure before one view's publication leaves that pointer unchanged without
   rolling back an already published sibling.
6. Modal Volume and S3 stores resolve direct view-first paths and enforce
   monotonicity independently per view.
7. A stable session ID remains on one pool/view, and the selected source reaches
   sample metadata.
8. Mixed-view requests use the selected view's latest pointer as their minimum
   version without fabricating a global cross-view floor; the actual serving
   version still reaches sample metadata and normal staleness handling.
9. The legacy single-view path remains behaviorally unchanged, including its
   existing request-version gate.
10. Resume chooses the newest saved trainer version represented by every
    selected view, rewinds only those view pointers, preserves their delta
    lineages, and rejects a converted baseline that disagrees with the durable
    checksums. An unselected incomplete view does not block resume. A save at N
    is eligible only once some selected pointer reached vN+1, since that publish
    committed every trainer host; a view still at vN must hold every file of its
    vN+1 delta.
11. A replacement replica selects the newest complete HF checkpoint for its
    own view at or below that view's pointer, and applies only the later deltas.
    A nested marker is never visible before the distributed save commits.

Rollout validation must then:

1. Starting from the same Qwen3.6 BF16 checkpoint, build the FP8 and NVFP4 base
   checkpoints with Miles' conversion tools. Do not use externally published
   quantized HF checkpoints.
2. From deterministic BF16 v0 and v1 tensors, publish nonempty FP8 and NVFP4 v1
   deltas and verify each child's existing target checksums.
3. Cold-boot one Hopper FP8 engine and one Blackwell NVFP4 W4A16 engine on the
   maintained SGLang pin.
4. Apply each view's v1 through the ordinary Stitch/SGLang stage-and-commit
   lifecycle and confirm deterministic inference against a clean load of the
   corresponding v1 checkpoint.
5. Demonstrate staggered readiness: one pool serves v1 while the other still
   serves its correctly identified prior version, then independently catches
   up.

Phase 1 does not change SGLang or start a trainer. Resume reuses the existing
trainer checkpoint and each selected view's immutable delta lineage; per-view
HF checkpoints optimize replacement-replica catch-up without changing that
lineage. The short end-to-end phase validates the combined lifecycle.

## Phase 2: rollout-only correctness and tuning

Use fixed model weights and a representative MiMo prompt slice. Send identical
requests and sampling settings through BF16, Hopper FP8, and Blackwell W4A16
engines. Record separately for every view:

- successful episodes, aborts, rewards, response lengths, and tool turns;
- when mismatch metrics are enabled: behavior log-probs, BF16 rescored
  log-probs, mismatch distribution, and ESS;
- selected experts when R3 collection is enabled;
- prefill/decode throughput, time to first token, inter-token latency, KV-cache
  usage, queueing, and cache hit rate;
- cold start, destination preparation, update preparation, and engine pause.

Tune TP/EP, engine concurrency, admission targets, KV-cache allocation, and
pool size independently per hardware/precision view. Lower-precision rollout
does not imply the same optimal settings across Hopper and Blackwell, and it
does not accelerate the BF16 trainer.

## Phase 3: experiment configurations

Create two experiment configs initially:

1. A complete, standalone MiMo mixed-precision GRPO config. It must define its
   own model, data, preparation, fleet, trainer, checkpoint, and algorithm
   settings rather than inheriting from a SWE-bench recipe. It disables TIS,
   mismatch metrics, and R3.
2. A thin TIS + R3 control config that imports the standalone base and
   changes only the algorithm controls.

The later experiment should be another thin variant of the standalone base so
differences remain attributable to the algorithm rather than rollout topology
or precision setup.

## Phase 4: short end-to-end validation

Run a small fixed prompt set for 5-10 training steps per config. A run passes
only when:

- all configured precision pools generate usable trajectories;
- each live engine agrees with its own view pointer and reports its actual
  committed logical version;
- when enabled, per-view mismatch metrics are finite and correctly attributed;
- the pure-GRPO run has no correction-induced loss changes;
- TIS and R3 activate only in the control run;
- one mid-run replica boots from its latest complete view checkpoint and
  independently applies the remaining delta tail;
- a save and resume restores the BF16 trainer and restarts every rollout view
  from a correct version;
- reward, abort, queue, staleness, and throughput metrics show no unexplained
  fleet-specific failure.

## Phase 5: final runs

Freeze converter settings, model revisions, sampling parameters, fleet layout,
and checkpoint policy after the short validation. Run the pure-GRPO and TIS +
R3 experiments with identical systems settings and report results both
globally and split by FP8 versus W4A16 rollout source. Add the later experiment
only after these two controls are reproducible.
