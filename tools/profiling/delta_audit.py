"""Audit an experiment's published deltas: touched names, reload closure, density.

The O(delta) partial-reload cost scales with each version's *closure* (the
touched checkpoint tensors plus everything they drag in), so this audit is the
cheap leading indicator to run against a bulletin before spending GPU-hours:
it recomputes, from each version's ``model.safetensors.index.json`` alone, the
same closure the engine's load plan applies:

  - expert closure: one touched tensor of an expert pulls every tensor of that
    expert (the per-expert post-loading transform consumes weights AND scales
    together, and destroys the raw forms in place), and
  - fused-sibling closure: a touched input of a load-time fusion pulls its
    sibling (the model re-fuses only from a complete part set).

It also fingerprints WHICH (layer, expert) slots each version touches and
compares them across runs — identical sets across independent runs indicate a
structural effect (e.g. padding tokens routed to expert 0), not learning. The
2026-06 K2.6 bulletin audit found exactly that: every published delta touched
only expert 0 per layer, so its 0.44% density reflects smoke-test-scale
training plus NVFP4 quantization masking sub-quantum updates, NOT a property
to extrapolate to production-scale runs.

CPU-only, no GPUs, seconds per bulletin. Run:
    EXPERIMENT_CONFIG=kimi_k2_6_nvfp4_disagg \
      uv run --extra modal modal run -m profiling.delta_audit::audit
Optionally filter with --run-id <hex>.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

import modal

# Resolve the experiment's volumes/paths from EXPERIMENT_CONFIG and bake them into the
# image env so the in-container re-import resolves the same objects without the cookbook.
if modal.is_local():
    from cookbook.miles_disagg import modal_train as mt

    _DELTA_VOLUME_NAME = mt.exp.DELTA_VOLUME_NAME
    _BULLETIN_ROOT = mt.exp.DELTA_BULLETIN_ROOT
    # e.g. "kimi-k2-6-nvfp4/nvfp4" under the prep volume
    _BASE_REL = os.path.relpath(mt.MODEL_NAME, str(mt.PREP_PATH))
else:
    _DELTA_VOLUME_NAME = os.environ["AUDIT_DELTA_VOLUME"]
    _BULLETIN_ROOT = os.environ["AUDIT_BULLETIN_ROOT"]
    _BASE_REL = os.environ["AUDIT_BASE_REL"]

# Keep in sync with cookbook/miles_disagg/modal_train.py.
_PREP_VOLUME_NAME = "miles-prep-checkpoints"

app = modal.App("weight-sync-delta-audit")
image = modal.Image.debian_slim(python_version="3.12").env(
    {
        "AUDIT_DELTA_VOLUME": _DELTA_VOLUME_NAME,
        "AUDIT_BULLETIN_ROOT": _BULLETIN_ROOT,
        "AUDIT_BASE_REL": _BASE_REL,
    }
)
delta_volume = modal.Volume.from_name(_DELTA_VOLUME_NAME)
prep_volume = modal.Volume.from_name(_PREP_VOLUME_NAME)

EXPERT_RE = re.compile(r"\.layers\.(\d+)\..*?\.experts\.(\d+)\.")
# Load-time fusions whose siblings must reload together (DSv3/Kimi attention).
FUSED_SIBLINGS = [("self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa")]


def _closure(touched: set[str], base_names: list[str]) -> set[str]:
    expert_index: dict[str, list[str]] = {}
    for name in base_names:
        m = EXPERT_RE.search(name)
        if m:
            expert_index.setdefault(name[: m.end() - 1], []).append(name)

    out = set(touched)
    for name in touched:
        m = EXPERT_RE.search(name)
        if m:
            out.update(expert_index.get(name[: m.end() - 1], ()))
        for a, b in FUSED_SIBLINGS:
            for x, y in ((a, b), (b, a)):
                if f".{x}." in name:
                    prefix = name.split(f".{x}.")[0]
                    out.update(n for n in base_names if n.startswith(f"{prefix}.{y}."))
    return out


@app.function(
    image=image,
    volumes={_BULLETIN_ROOT: delta_volume, "/prep": prep_volume},
    timeout=1800,
)
def audit(run_id: str = "") -> None:
    base_index = os.path.join("/prep", os.environ["AUDIT_BASE_REL"], "model.safetensors.index.json")
    with open(base_index) as f:
        base_names = list(json.load(f)["weight_map"])
    expert_slots = {m.group(1, 2) for n in base_names if (m := EXPERT_RE.search(n))}
    root = os.environ["AUDIT_BULLETIN_ROOT"]
    print(f"base: {len(base_names)} tensors, {len(expert_slots)} expert slots  ({base_index})")
    print(f"{'run':>14} {'ver':>4} {'mtime':>12} {'touched':>8} {'experts':>8} {'closure':>8} {'density':>8}  payload")

    per_version_experts: dict[str, dict[str, set]] = {}
    for run in sorted(os.listdir(root)):
        if run_id and run != run_id:
            continue
        run_dir = os.path.join(root, run)
        if not os.path.isdir(run_dir):
            continue
        for ver in sorted(d for d in os.listdir(run_dir) if d.startswith("weight_v")):
            vdir = os.path.join(run_dir, ver)
            files = sorted(os.listdir(vdir)) if os.path.isdir(vdir) else []
            idx_path = os.path.join(vdir, "model.safetensors.index.json")
            if not os.path.isfile(idx_path):
                if files:
                    print(f"{run:>14} {ver[-4:]:>4}  (no index; files: {files[:4]})")
                continue
            with open(idx_path) as f:
                weight_map = json.load(f)["weight_map"]
            touched = set(weight_map)
            payload = sum(
                os.path.getsize(os.path.join(vdir, f))
                for f in files
                if f.endswith(".safetensors")
            )
            missing_shards = set(weight_map.values()) - set(files)
            mtime = max(os.path.getmtime(os.path.join(vdir, f)) for f in files)
            stamp = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%m-%d %H:%M")
            experts = {m.group(1, 2) for n in touched if (m := EXPERT_RE.search(n))}
            per_version_experts.setdefault(ver, {})[run] = experts
            clo = _closure(touched, base_names)
            print(
                f"{run:>14} {ver[-4:]:>4} {stamp:>12} {len(touched):8d} {len(experts):8d} "
                f"{len(clo):8d} {len(clo) / len(base_names):8.2%}  {payload / 1e6:8.1f} MB"
                + (f"  MISSING SHARDS: {sorted(missing_shards)}" if missing_shards else "")
            )
            kinds = Counter(
                "expert" if EXPERT_RE.search(n) else n.rsplit(".", 1)[-1] for n in touched
            )
            print(f"{'':>14}      breakdown: {dict(kinds.most_common(5))}")
            if experts and len(experts) <= 6:
                print(f"{'':>14}      touched experts (layer, id): {sorted(experts, key=lambda t: (int(t[0]), int(t[1])))}")

    print("\n===== cross-run expert-slot comparison (identical sets => structural, not learning) =====")
    for ver, by_run in sorted(per_version_experts.items()):
        sets = {r: e for r, e in by_run.items() if e}
        if len(sets) < 2:
            continue
        inter = set.intersection(*sets.values())
        union = set.union(*sets.values())
        print(f"{ver}: runs={len(sets)} intersection={len(inter)} union={len(union)}")
    print("\nAUDIT DONE")
