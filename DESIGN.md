# stitch — design & rewrite plan

## Purpose

stitch is a **general library for disaggregated, versioned rollout** in RL post-training: it turns an external inference endpoint into a **current-policy, version-correct, elastic rollout service**.

An RL framework (miles, slime) that disaggregates rollout exposes exactly three plug points — and nothing else:

- `custom_update_weight_post_write_path` (miles) / `custom_delta_pre_push_path` (slime) — publish hook, called after each weight update (full *or* delta)
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
- **Different store / engine / pool?** add a new instance behind the abstraction — a core file, or a third-party package subclassing the base. Zero core-logic edits.
- **Different deployment** (k8s, another image, another provider, another model)? it's a **cookbook recipe**. Core never changes.

stitch is a **library, not a baked service** — every concrete deployment (including the Modal pool app) is a cookbook recipe.

## Domain model

A **version** is a published policy state `(run_id, version)`. It is an **anchor** (full — its files *are* the weights) or a **delta** (diffs a `base` that chains back to an anchor). Invariant concerns: the sync handshake (publish → load → readiness), staleness/version control, session affinity, router replay — identical regardless of where weights live, how the pool runs, or which engine serves.

## Abstractions (ports)

Each port is a plain base class; an instance subclasses it and overrides its methods (a missing override surfaces as `NotImplementedError` when called).

- **Store** — where versions live + the `latest` pointer: `refresh`/`read_pointer`/`advance_pointer`/`claim`/`read_manifest`/`publish`/`materialize`.
- **Engine** — drive one inference engine: `stage`/`commit`/`flush`/`pause`/`resume`/`reset` + `stamp_request`/`stamp_response` + `base_url`.
- **Pool** — reach the replica set: `gateway_url`/`discover_replicas` + `wake`(opt)/`scale`(opt, default no-op). A *client* to a running pool, not its deployment.

## Layout

```
src/stitch/                     # THE LIBRARY — general: abstractions + logic + instances
  versions.py                   # domain vocabulary: VersionRef, VersionManifest
                                #   {kind, base_version, files, delta_encoding, compression,
                                #   checksum}, VersionConstraint, ReplicaState/PoolState,
                                #   SyncState; the pure pointer rules (decide_pointer_move)
  sync.py                       # Reconciler (reconcile replica → latest) + AdmissionGate
  service.py                    # create_app (versioned proxy) + serve(store, engine, ...) + readiness()
  publish.py                    # publish_version() + claim_run() + constrain_request()
  stores/base.py  + modal_volume.py   # Store           + ModalVolumeStore
  engines/base.py + sglang.py         # Engine          + SGLangEngine
  pools/base.py   + modal_flash.py    # Pool            + ModalFlashPool (client)
cookbook/                       # NON-core: recipes (deployments), layered
  common/                       #   framework-agnostic, shared by every recipe:
                                #     config.py (ModalConfig + mount paths), serving_image.py,
                                #     server.py (register_server), sidecar.py (serve entrypoint),
                                #     hooks.py (publish/claim/request logic), launch.py, ray_cluster.py,
                                #     process.py, smoke.py
  miles/                        #   framework subdir: trainer_image.py (+ pins), config.py
                                #     (MilesConfig), app.py, prep.py, configs/<experiment>.py
  slime/                        #   symmetric (SlimeConfig; app.py; configs/<experiment>.py)
tools/profiling/                # dev-only diagnostics (never imported by src/)
tests/                          # unit tests + the in-memory core harness (was local_disagg)
```

Only the three instance files in the library are Modal/sglang-specific, each isolated behind its port. Everything provider- or experiment-specific lives in `cookbook/`.

## Full vs delta — one mechanism, a manifest field

`kind = full` (anchor) or `delta`. `stage(target)` = walk back to the nearest anchor ≤ target, seed from it, replay deltas forward.
- **full-sync** = every version an anchor
- **delta-sync** = one base anchor + deltas (miles, slime)
- **periodic-full** = anchors every K (bounds joiner catch-up, enables GC)

No codec component: encode lives in the framework, decode inside the engine; stitch carries the format as manifest data (`delta_encoding`/`compression`/`checksum`) so the two agree.

## Framework integration — agnostic helpers, config-referenced

Both miles and slime write the same HF-safetensors + delta-metadata layout, so the bridge is framework-agnostic: `publish_version()` (parse the standard HF index → `VersionManifest` → publish + wake), `claim_run()`, and `constrain_request()` in `publish.py`. The publish/claim/request **logic** is shared once in `cookbook/common/hooks.py`; a framework's run config simply points its dotted hook paths at `cookbook.common.hooks.*` (no per-framework re-export shim). The only framework-specific residue is which config key names the publish hook (`custom_update_weight_post_write_path` vs `custom_delta_pre_push_path`).

## Correctness invariants (contracts, not emergent behavior)

1. **Applied-version flips atomically, only after `commit` succeeds** — the gate never advertises an unserved version. Under `in_place` commit, attribution comes from the request's own stamp, not the replica's live version.
2. **Publish writes files → then advances the pointer; apply verifies checksums; a missing source = retry, not reseed.**
3. **Retention: never GC an anchor or delta any replica's chain still needs** (keep newest-anchor ≤ `min(applied)` through `latest`).
4. **`Store.materialize` guarantees files are locally readable before returning** (hides mount vs download).

## Naming conventions

- Ports are single nouns (`Store`, `Engine`, `Pool`); instances are `<Concrete><Port>` (`ModalVolumeStore`, `SGLangEngine`, `ModalFlashPool`); functions are `verb_noun`.
- One internal name per concept, translated to external wire spellings only at the boundary: `delta_encoding`/`compression`/`checksum` (manifest) map to Cognition's `compression_format`/`checksum_format`; the manifest reads the wire key `diff` but the field is `delta_encoding`.
- `base_url` (the engine's HTTP base), `materialize` (ensure a version is locally readable) — named for what they are, not proxy/handle metaphors.

## What moved (old tree → this structure)

| Old | New home |
|---|---|
| `protocol.py` | `versions.py` (types + pointer rules) + `{stores,engines,pools}/base.py` (the ports) |
| `sync.py` (`WeightSyncManager`, `RolloutAdmissionGate`) | `sync.py` (`Reconciler`, `AdmissionGate`) |
| `bulletin.py` + Volume half of `providers/modal.py` | `stores/modal_volume.py` (`ModalVolumeStore`) |
| `servers/sglang.py` | `service.py` (engine-agnostic; stamp behind `Engine`) |
| `engines/sglang.py` (adapter) | `engines/sglang.py` (`SGLangEngine`) |
| Flash *client* half of `providers/modal.py` | `pools/modal_flash.py` (`ModalFlashPool`) |
| `cookbook/bulletin_hooks.py` | `publish.py` helpers + `cookbook/common/hooks.py` (shared logic) |
| `cookbook/sidecar.py` | `service.py` (`serve`) + `cookbook/common/sidecar.py` (entrypoint) |
| `cookbook/{ray_cluster,sidecar_process,serving,trainer_helpers}.py` | `cookbook/common/{ray_cluster,process,serving_image,launch}.py` + `server.py`/`smoke.py` |
| `cookbook/{miles,slime}_disagg/` (copy-forked apps + configs) | `cookbook/{miles,slime}/` (shared Server; per-framework Trainer + pins + configs) |
| `src/stitch/trainers/slime.py` (dead) | deleted |
| `cookbook/local_disagg` | `tests/` (in-memory core harness) |
| `profiling/` | `tools/profiling/` |
| `VersionManifest.from_slime_index` | `from_hf_index` |

## Rewrite plan (branch `stitch-v2`) — extract & de-leak, not reinvent

- **Phase 0 — skeleton.** ✅ Tree; `profiling/` → `tools/profiling/`; `tests/`; `pyproject`.
- **Phase 1 — core, provable in-memory.** ✅ `versions.py`, `sync.py`, `service.py`, `publish.py`, the three port `base.py` — all provider/engine/framework-agnostic, with the in-memory harness (`tests/`) as the gate. Passes with fakes before any instance exists.
- **Phase 2 — instances.** ✅ `stores/modal_volume`, `engines/sglang`, `pools/modal_flash` — proven code ported onto the ports, leaks killed (proxy engine-agnostic via `Engine`; `publish` agnostic; Volume durability inside the store; Pool client-only).
- **Phase 3 — cookbook.** ✅ `common/` (shared, incl. the parameterized `register_server`) + `miles/` + `slime/`, each self-contained config, all experiments ported. The old `*_disagg` tree stays as reference **until e2e, then deleted**.
- **Phase 4 — new-version fixes.** Land the miles/sglang updates (TE 2.17, etc.) on the clean structure — "make NVFP4 work," now on solid ground.

Tests: `tests/` (core harnesses + the cookbook hook test) is the gate for Phases 0–3; the Modal-backed instances + recipes are validated e2e.
