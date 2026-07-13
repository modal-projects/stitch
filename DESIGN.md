# stitch — design & rewrite plan

## Purpose

stitch is a **general library for disaggregated, versioned rollout** in RL post-training: it turns an external inference endpoint into a **current-policy, version-correct, elastic rollout service**.

An RL framework (miles, slime) that disaggregates rollout exposes exactly three plug points — and nothing else:

- `custom_update_weight_post_write_path` — publish hook, called after each weight update (full *or* delta)
- `custom_rollout_request_hook_path` — request hook, called before each generation request
- `rollout_endpoint_url` — when set, the framework runs no local engines (`rollout_num_gpus=0`) and sends generation here

stitch fills those hooks and runs the service side. What a raw endpoint can't do — and stitch does:

1. get published weights into an elastic, multi-replica pool and reloaded (**weight sync**);
2. make every sample version-attributable and let the trainer bound staleness (**version correctness**);
3. let replicas join/leave mid-run and catch up (**elastic sync**);
4. coordinate full-vs-delta apply, session affinity, and MoE router replay across publish/serve.

stitch is separate from the *framework* (miles and slime present the identical seam → framework-general) and from the *engine* (sglang is one instance behind an interface → engine-general). Cognition's independent RL Rollout Spec uses the same primitives — evidence the domain model is real.

## The one rule: general core, customizable edges

> **Core** = an abstraction, an *implementation* of one (an instance), or provider/engine/framework-agnostic logic.
> **Non-core** = a concrete *deployment*, an experiment *config*, a customer *facade*, dev *tooling*, or *tests*.

The generality test both ways:
- **Different store / engine / pool?** add a new instance behind the abstraction — a core file, or a third-party package implementing the Protocol. Zero core-logic edits.
- **Different deployment** (k8s, another image, another provider, another model)? it's an **example**. Core never changes.

stitch is a **library, not a baked service** — every concrete deployment (including the Modal pool app) is an example.

## Domain model

A **version** is a published policy state `(run_id, version)`. It is an **anchor** (full — its files *are* the weights) or a **delta** (diffs a `base` that chains back to an anchor). Invariant concerns: the sync handshake (publish → load → readiness), staleness/version control, session affinity, router replay — identical regardless of where weights live, how the pool runs, or which engine serves.

## Abstractions (ports)

- **Store** — where versions live + the `latest` pointer (read/publish/pointer/claim/refresh/open_version).
- **Engine** — drive one inference engine: `stage`/`commit`/`reset`/`applied_version` + request version-`stamp`/read + `generate_url`.
- **Pool** — reach the replica set: `gateway_url`/`discover_replicas`/`wake`(opt)/`scale`(opt). A *client* to a running pool, not its deployment.

## Layout

```
src/stitch/                     # THE LIBRARY — general: abstractions + logic + instances
  contract.py                   # Store/Engine/Pool abstractions; VersionRef,
                                #   VersionManifest{kind,base,files,diff,compression,checksum},
                                #   VersionConstraint, SyncState; pointer rules
  sync.py                       # Reconciler (reconcile replica → latest) + AdmissionGate
  service.py                    # create_app (versioned proxy) + serve(store, engine, ...) + readiness()
  publish.py                    # publish_version() + publish_from_hf_layout() + stamp_request()
  stores/modal_volume.py        # ModalVolumeStore        — a Store instance
  engines/sglang.py             # SGLangEngine            — an Engine instance
  pools/modal_flash.py          # ModalFlashPool (client) — a Pool instance
examples/<experiment>/          # a config owning the WHOLE experiment (customizable, provider-specific):
                                #   the Modal app (image + Server cls running serve()+engine + Flash +
                                #   Trainer cls + entrypoints), model prep, run config, the ~2-line
                                #   framework hook shim, and any consumer facade (e.g. cognition /hot_load)
  _modal/                       # OPTIONAL shared example helper (Modal app factory) — keeps examples DRY,
                                #   lives in the example layer, NOT general core
tools/profiling/                # dev-only diagnostics (never imported by src/)
tests/                          # unit tests + the in-memory core harness (was local_disagg)
```

Only three files in the library are Modal/sglang-specific (the instances), each isolated behind its abstraction. Everything provider- or experiment-specific — including the deployable Modal pool app — is an example.

## Full vs delta — one mechanism, a manifest field

`kind = full` (anchor) or `delta`. `stage(target)` = walk back to the nearest anchor ≤ target, seed from it, replay deltas forward.
- **full-sync** = every version an anchor (slime `update_weight_from_disk.py`)
- **delta-sync** = one base anchor + deltas (slime `update_weight_from_disk_delta.py`, miles)
- **periodic-full** = anchors every K (bounds joiner catch-up, enables GC)

No codec component: encode lives in the framework, decode inside the engine; stitch carries the format as manifest data (`diff`/`compression`/`checksum`) so the two agree.

## Framework integration — agnostic helper + example shim

Both miles and slime write the same HF-safetensors + delta-metadata layout, so the substantive bridge is framework-agnostic and lives in core: **`publish_from_hf_layout()`** (parse standard layout → `VersionManifest` → publish + wake) and **`stamp_request()`**. The only framework-specific residue is the ~2-line hook shim that conforms to a framework's hook signature — it lives in the **example** (pinned with that framework's version). Core never imports a framework's evolving output format; a framework either conforms to the layout or its example translates.

## Correctness invariants (contracts, not emergent behavior)

1. **Applied-version flips atomically, only after `commit` succeeds** — the gate never advertises an unserved version. Under `in_place` commit, attribution comes from the request's own stamp, not the replica's live version.
2. **Publish writes files → then advances the pointer; apply verifies checksums; a missing source = retry, not reseed.**
3. **Retention: never GC an anchor or delta any replica's chain still needs** (keep newest-anchor ≤ `min(applied)` through `latest`).
4. **`Store.open_version` guarantees files are locally readable before returning** (hides mount vs download).

## Naming conventions

- Abstractions are single nouns (`Store`, `Engine`, `Pool`); instances are `<Concrete><Port>` (`ModalVolumeStore`, `SGLangEngine`, `ModalFlashPool`); functions are `verb_noun`.
- `Pool` (matches the existing "rollout pool" vocabulary).
- manifest `compression`/`checksum` map to Cognition's `compression_format`/`checksum_format`.

## What moves where (from today's tree)

| Today | New home |
|---|---|
| `protocol.py` | `contract.py` (+ the abstractions) |
| `sync.py` (`WeightSyncManager`, `RolloutAdmissionGate`) | `sync.py` (`Reconciler`, `AdmissionGate`) |
| `bulletin.py` + Volume half of `providers/modal.py` | `stores/modal_volume.py` (`ModalVolumeStore`) |
| `servers/sglang.py` | `service.py` (engine-agnostic; stamp behind `Engine`) |
| `engines/sglang.py` | `engines/sglang.py` (`SGLangEngine`) |
| Flash *client* half of `providers/modal.py` | `pools/modal_flash.py` (`ModalFlashPool`) |
| `cookbook/bulletin_hooks.py` | `publish.py` (`publish_from_hf_layout`, `stamp_request`) |
| `cookbook/sidecar.py` | `service.py` (`serve`) |
| `cookbook/{miles,slime}_disagg`, `standalone_rollouts`, `{ray_cluster,sidecar_process,serving,trainer_helpers}.py`, the Modal `Server`/`Trainer` skeletons | `examples/<experiment>/` (+ optional `_modal` helper) |
| `src/stitch/trainers/slime.py` (dead `publish_delta_version`) | deleted; live hook → an example shim |
| `cookbook/local_disagg` | `tests/` (in-memory core harness) |
| `profiling/` | `tools/profiling/` |
| `VersionManifest.from_slime_index`, `layout="slime"` | `from_hf_index`, `layout="hf_delta"` |

## Rewrite plan (branch `stitch-v2`) — extract & de-leak, not reinvent

Core logic is already proven (today's `local_disagg`). This is relocation + naming the abstractions + fixing the inward-dependency leaks.

- **Phase 0 — skeleton.** Lay out the tree; move `profiling/` → `tools/profiling/`; create `tests/` + `examples/`; wire `pyproject`. No logic.
- **Phase 1 — core, provable in-memory.** `contract.py` (abstractions + types + rules), `sync.py`, `service.py`, `publish.py` — all provider/engine/framework-agnostic — with the in-memory harness (port of `local_disagg`) as the gate. **Gate: core passes with fakes before any instance exists.**
- **Phase 2 — instances.** `stores/modal_volume`, `engines/sglang`, `pools/modal_flash` — port the proven code onto the abstractions and kill the leaks (proxy engine-agnostic via `Engine.stamp`; `publish` agnostic; Volume durability inside the store; Pool is client-only).
- **Phase 3 — first example.** One `examples/<experiment>/` assembling the Modal app + prep + config + hook shim (the copy-forked skeleton collapses into one example, optionally an `_modal` helper). Fold in the slime cleanup; delete the superseded old tree.
- **Phase 4 — new-version fixes.** Land the miles/sglang updates (TE 2.17, etc.) on the clean structure — "make NVFP4 work," now on solid ground.

Each phase is one reviewable PR. Phases 0–1 are pure and cheap; the leaks die in Phase 2; nothing gets a second instance or a stub until it is actually needed.
