"""Materialize BFCL multi-turn episodes into a teacher-forced replay file.

For each episode in a BFCL multi-turn category, emit the exact chat-message
prefix at every ground-truth step (K = 0..total_steps-1) plus the OpenAI
``tools=`` schema list. The replay probe (``traffic.run_bfcl_replay``) then
drives an episode by sending step K's prefix, recording latency, and advancing
on ground truth — the workload is deterministic and bit-identical across
routing-policy arms, so episodes pair exactly for A/B statistics. Tool results
in the prefixes are REAL: ground-truth calls replayed against BFCL's in-process
state machines (GorillaFileSystem, TwitterAPI, ...).

Requires ``bfcl-eval`` (PyPI). The helpers below are vendored from
modal-projects/training-gym branch ``alessio/bfcl-environment``
(modal_training_gym/common/environments/bfcl.py) so this script has no
dependency on that private repo.

    uv run --with bfcl-eval python -m tools.probes.bfcl_prep --out bfcl_replay.jsonl
    uv run --extra modal modal volume put stitch-probe-results bfcl_replay.jsonl /bfcl/bfcl_replay.jsonl -e alessio-dev
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import statistics
from copy import deepcopy
from typing import Any

# ── Vendored from training-gym bfcl.py (alessio/bfcl-environment) ────────────

_JSON_TYPE_MAP = {
    "dict": "object", "list": "array", "tuple": "array",
    "float": "number", "integer": "integer", "string": "string", "boolean": "boolean",
}

DEFAULT_SYSTEM_PROMPT = """\
You are a tool-using agent completing a user's request with the function tools provided to you. Work one step at a time.

Rules:
- Make EXACTLY ONE tool call per turn. Emit only the tool call — no extra prose, narration, or markdown fences around it.
- Use only the tools provided to you, with their exact names. Do not invent tools, arguments, or file paths.
- Each user message may require several tool calls before the request is satisfied; keep calling tools until the request is complete, then stop calling tools.
- After each tool result, check whether it succeeded before continuing; do not blindly repeat a failed call with the same arguments."""


def _data_dir() -> str:
    import bfcl_eval

    return os.path.join(os.path.dirname(bfcl_eval.__file__), "data")


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _backend_mappings():
    # Location moved across bfcl-eval releases; the training-gym branch imports
    # from constants.executable_backend_config, PyPI ships it in multi_turn_utils.
    try:
        from bfcl_eval.constants.executable_backend_config import (
            CLASS_FILE_PATH_MAPPING,
            MULTI_TURN_FUNC_DOC_FILE_MAPPING,
            STATELESS_CLASSES,
        )
    except ImportError:
        from bfcl_eval.constants.category_mapping import MULTI_TURN_FUNC_DOC_FILE_MAPPING
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            CLASS_FILE_PATH_MAPPING,
            STATELESS_CLASSES,
        )

    return CLASS_FILE_PATH_MAPPING, MULTI_TURN_FUNC_DOC_FILE_MAPPING, STATELESS_CLASSES


def parse_call_string(call: str) -> dict[str, Any]:
    node = ast.parse(call.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"Not a call expression: {call!r}")
    name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
    arguments: dict[str, Any] = {f"_pos{i}": ast.literal_eval(a) for i, a in enumerate(node.args)}
    for kw in node.keywords:
        arguments[kw.arg] = ast.literal_eval(kw.value)
    return {"name": name, "arguments": arguments}


def _normalize_arguments(owner: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    positional = sorted(
        ((k, v) for k, v in arguments.items() if k.startswith("_pos")), key=lambda kv: int(kv[0][4:])
    )
    if not positional:
        return arguments
    try:
        param_names = [p for p in inspect.signature(getattr(owner, name)).parameters if p != "self"]
    except (TypeError, ValueError):
        param_names = []
    normalized = dict(zip(param_names, (v for _, v in positional)))
    normalized.update({k: v for k, v in arguments.items() if not k.startswith("_pos")})
    return normalized


def _instantiate(class_name: str, initial_config: dict) -> Any:
    import importlib

    class_map, _, stateless = _backend_mappings()
    module = importlib.import_module(class_map[class_name])
    instance = getattr(module, class_name)()
    if class_name not in stateless:
        instance._load_scenario(deepcopy(initial_config.get(class_name, {})), long_context=False)
    return instance


def _method_owner(instances: dict[str, Any], method_name: str) -> Any | None:
    if method_name.startswith("_"):
        return None
    for instance in instances.values():
        if hasattr(type(instance), method_name):
            return instance
    return None


def _stringify(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except TypeError:
            return str(result)
    return str(result)


def replay(involved_classes: list[str], initial_config: dict, calls: list[dict]) -> list[str]:
    instances = {c: _instantiate(c, initial_config) for c in involved_classes}
    observations = []
    for call in calls:
        owner = _method_owner(instances, call["name"])
        if owner is not None:
            call["arguments"] = _normalize_arguments(owner, call["name"], call.get("arguments") or {})
            try:
                result = getattr(owner, call["name"])(**deepcopy(call["arguments"]))
                observations.append(_stringify(result))
                continue
            except Exception as e:  # mirrors upstream's catch-all
                observations.append(f"Error during execution: {e}")
                continue
        observations.append(f"Error during execution: unknown function {call['name']!r}")
    return observations


def to_json_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: to_json_schema(v) for k, v in node.items() if k != "default"}
        if out.get("type") in _JSON_TYPE_MAP:
            out["type"] = _JSON_TYPE_MAP[out["type"]]
        return out
    if isinstance(node, list):
        return [to_json_schema(v) for v in node]
    return node


def load_func_docs(involved_classes: list[str], excluded: list[str] | None = None) -> dict:
    _, doc_map, _ = _backend_mappings()
    excluded_set = set(excluded or [])
    schemas: dict[str, dict] = {}
    for class_name in involved_classes:
        doc_file = doc_map.get(class_name)
        if not doc_file:
            continue
        for doc in _load_jsonl(os.path.join(_data_dir(), "multi_turn_func_doc", doc_file)):
            if doc["name"] not in excluded_set:
                schemas[doc["name"]] = {
                    "description": doc.get("description", ""),
                    "parameters": doc.get("parameters", {}),
                }
    return schemas


def tool_schemas_to_openai(tool_schemas: dict) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.get("description", ""),
                "parameters": to_json_schema(spec.get("parameters", {})),
            },
        }
        for name, spec in (tool_schemas or {}).items()
    ]


def build_prefix_messages(turns: list[dict], observations: list[str], K: int, obs_limit: int) -> list[dict]:
    """Chat prefix after the first K ground-truth calls (user turns inserted in place)."""
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    shown = 0
    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})
        for call in turn["calls"]:
            if shown >= K:
                return messages
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{shown}",
                            "type": "function",
                            "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": f"call_{shown}", "content": str(observations[shown])[:obs_limit]}
            )
            shown += 1
    return messages


def _first_user_text(turn_messages: list[dict]) -> str:
    for m in turn_messages:
        if m.get("role") == "user" and str(m.get("content", "")).strip():
            return str(m["content"])
    return ""


# ── Prep driver ───────────────────────────────────────────────────────────────


def main() -> None:
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX

    p = argparse.ArgumentParser()
    p.add_argument("--category", default="multi_turn_base")
    p.add_argument("--out", default="bfcl_replay.jsonl")
    p.add_argument("--obs-limit", type=int, default=1500)
    args = p.parse_args()

    fname = f"{VERSION_PREFIX}_{args.category}.json"
    entries = {e["id"]: e for e in _load_jsonl(os.path.join(_data_dir(), fname))}
    gts = {e["id"]: e["ground_truth"] for e in _load_jsonl(os.path.join(_data_dir(), "possible_answer", fname))}

    rows, step_counts, prefix_chars = [], [], []
    for eid, entry in entries.items():
        gt = gts.get(eid)
        if not gt or not any(gt):
            continue
        turns = [
            {"user": _first_user_text(tm), "calls": [parse_call_string(c) for c in calls]}
            for tm, calls in zip(entry["question"], gt)
        ]
        flattened = [c for t in turns for c in t["calls"]]
        observations = replay(entry["involved_classes"], entry["initial_config"], flattened)
        tools = tool_schemas_to_openai(load_func_docs(entry["involved_classes"], entry.get("excluded_function")))
        steps = [
            build_prefix_messages(turns, observations, K, args.obs_limit) for K in range(len(flattened))
        ]
        rows.append({"episode_id": eid, "tools": tools, "steps": steps})
        step_counts.append(len(steps))
        prefix_chars.extend(len(json.dumps(s)) for s in steps)

    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    print(f"category={args.category} episodes={len(rows)} total_steps={sum(step_counts)}")
    print(f"steps/episode: min={min(step_counts)} p50={statistics.median(step_counts)} max={max(step_counts)}")
    print(
        "prefix size (chars): "
        f"p50={statistics.median(prefix_chars):.0f} p95={sorted(prefix_chars)[int(0.95*len(prefix_chars))]:.0f} "
        f"max={max(prefix_chars)} (~/4 = tokens, tools excluded)"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
