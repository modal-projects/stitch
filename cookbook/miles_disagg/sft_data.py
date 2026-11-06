"""SFT data for the warm start (qwen3_8_27b_sft_warmstart): public Terminus-2 trajectories as our RL harness shows
them. A CPU job; it downloads from Hugging Face straight into the Volume `sft-data`.

Sources, at pinned revisions:
- nvidia/Nemotron-Terminal-Corpus (CC BY 4.0, teacher DeepSeek-V3.2): the skill-based synthetic tasks without their
  security part, and the SWE adapter (not the math and code adapters).
- nvidia/Nemotron-SFT-SWE-v3 (CC BY 4.0): its Terminus-2 rows only.
- open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K (Apache-2.0): SWE-smith tasks.

Each Terminus-2 JSON trajectory becomes the conversation that Harbor's terminus-2 at harbor_config.HARBOR_REF (XML
parser, interleaved thinking) shows: its XML prompt, each reply as XML that its XML parser reads back to the same
analysis, plan, commands and completion, and each observation with the XML harness's warnings and completion
prompt. Parse-error turns (the reply and its error message) are cut out. Trajectories are dropped when the trial
errored, the last reply does not mark the task complete, a format is not recognized, a reply has no reasoning, a
reply is over 16,384 tokens or the whole is over 131,072 (counted as Miles' qwen3 loss mask renders them with
sft.CHAT_TEMPLATE), or the task overlaps our eval and test sets (overlaps() and benchmarks() of the eval
repository's envs/general_sets.py, plus a stricter containment cut). A seeded sample of 300M tokens, in fixed shares by
source, then goes to /<NAME>/train.jsonl, with /<NAME>/manifest.json (counts, drops by reason, tokens).

    EVAL_ENVS_DIR=<eval repo>/envs uv run --extra modal modal run -e <env> -m cookbook.miles_disagg.sft_data
"""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import re
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import modal

from cookbook.miles_disagg import harbor_config, sft
from cookbook.miles_disagg.configs import qwen3_8_27b_swebench as base

NAME, TOKENS, SEED = "warmstart-v1", 300_000_000, 0
MAX_REPLY, MAX_TOKENS = 16_384, 131_072
TERMINAL = ("nvidia/Nemotron-Terminal-Corpus", "a1667c4ffdadea02a89bffe4f1bb7ca2ff19f8d9")
SWE_V3 = ("nvidia/Nemotron-SFT-SWE-v3", "3f73de64c1fe928a8f538fe45ccc10c228cc4c6a")
COLDSTART = ("open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K", "261a70c35c0b16327c01a745ab65cc5651fb00d9")
SKILLS = {  # Terminal-Corpus skill-based tasks by difficulty, without security
    "easy": "data_processing data_querying data_science debugging dependency_management file_operations "
    "scientific_computing software_engineering",
    "medium": "data_processing data_querying data_science debugging dependency_management file_operations "
    "model_training scientific_computing software_engineering system_administration",
    "mixed": "data_processing data_science debugging file_operations scientific_computing",
}
FILES = [
    *(
        (*TERMINAL, f"synthetic_tasks/skill_based/{level}/{skill}/data_filtered.parquet", "terminal-corpus/skill-based")
        for level, skills in SKILLS.items()
        for skill in skills.split()
    ),
    (*TERMINAL, "dataset_adapters/swe.parquet", "terminal-corpus/swe"),
    *((*SWE_V3, f"data/train-{i:05d}-of-00096.parquet", "nemotron-sft-swe-v3") for i in range(96)),
    *((*COLDSTART, f"data/train-{i:05d}-of-00002.parquet", "openthoughts-coldstart") for i in range(2)),
]
LICENSES = {TERMINAL[0]: "cc-by-4.0", COLDSTART[0]: "apache-2.0"}  # SWE-v3 rows carry their own
# Token shares of the sample: half terminal tasks, half repository bug fixes.
SHARES = {
    "terminal-corpus/skill-based": 0.5,
    "openthoughts-coldstart": 0.2,
    "nemotron-sft-swe-v3": 0.15,
    "terminal-corpus/swe": 0.15,
}
# overlaps() keeps out near-duplicates at 5-gram containment >= 0.5. A benchmark prompt can be a rewrite of a
# training task (Harbor Index rewrote a SWE-smith issue of the OpenThoughts set: containment 0.33), so SFT data
# also drops tasks whose closest benchmark prompt has containment >= NEAR.
NEAR = 0.3

TERMINUS = "You are an AI assistant tasked with solving command-line tasks in a Linux environment."
PARSE_ERRORS = "Previous response had parsing errors:\n"
WARNINGS = "Previous response had warnings:\n"
STATE = "Current terminal state:\n"
CONFIRM = "\n\nAre you sure you want to mark the task as complete?"
OUTPUTS = ("New Terminal Output:", "Current Terminal Screen:", "Previous command:\n")

ENVS_DIR = os.environ.get("EVAL_ENVS_DIR")  # the eval repository's envs/ folder
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "pyarrow==26.0.1",
        "transformers==5.22.1",
        "jinja2==3.1.6",
        f"harbor @ git+https://github.com/harbor-framework/harbor@{harbor_config.HARBOR_REF}",
    )
    .env({"PYTHONPATH": "/root/envs", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("cookbook", ignore=["**/__pycache__"])
)
if ENVS_DIR:
    image = image.add_local_dir(ENVS_DIR, "/root/envs", ignore=["**/__pycache__"])
with image.imports():
    import general_sets
    import pyarrow.parquet as pq
    from harbor.agents.terminus_2 import (
        terminus_json_plain_parser,
        terminus_xml_plain_parser,
    )
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

app = modal.App("sft-data")
volume = modal.Volume.from_name("sft-data", create_if_missing=True, version=2)
DATA, BENCHMARKS = Path("/data"), Path("/harbor-envs")
_worker: dict = {}


def first_prompt(messages: list[dict]) -> tuple[str, str, int] | None:
    """(task, terminal state, index of the first reply) of a Terminus-2 JSON trajectory, else None."""
    head = messages[0]["content"] or ""
    if not head.startswith(TERMINUS) or "Format your response as JSON" not in head:
        return None
    if messages[0]["role"] == "user":
        _, found, rest = head.partition("\n\nTask Description:\n")
        task, found_state, state = rest.rpartition(f"\n\n{STATE}")
        return (task, state, 1) if found and found_state else None
    # Nemotron-SFT-SWE-v3: the template's instructions as the system message, then "Task:\n\n<task>\n\n<screen>".
    body = messages[1]["content"] if len(messages) > 1 else ""
    task, found, screen = body.removeprefix("Task:\n\n").rpartition("\n\nCurrent Terminal Screen:\n")
    return (task, f"Current Terminal Screen:\n{screen}", 2) if body.startswith("Task:\n\n") and found else None


def reply(message: dict) -> tuple[str, str]:
    """(reasoning, reply) of an assistant message; the reasoning is a field or leads the content in <think> tags."""
    text = (message["content"] or "").strip()
    if reasoning := (message.get("reasoning_content") or "").strip():
        return reasoning, text
    if text.startswith("<think>") and "</think>" in text:
        reasoning, _, text = text.removeprefix("<think>").partition("</think>")
    return reasoning.strip(), text.strip()


def terminal_output(observation: str) -> str | None:
    """The terminal output in a Terminus-2 observation, without its warnings or completion prompt."""
    if observation.startswith(WARNINGS):
        _, found, observation = observation.partition("\n\n")
        if not found:
            return None
    elif observation.startswith(STATE) and CONFIRM in observation:
        observation = observation[len(STATE) : observation.rindex(CONFIRM)]
    return observation if observation.startswith(OUTPUTS) else None


def render_xml(analysis: str, plan: str, commands: list[tuple[str, float]], complete: bool | None) -> str:
    """A reply in the layout of Harbor's terminus-xml-plain prompt; complete=None leaves out <task_complete>."""
    keys = "".join(f'<keystrokes duration="{duration}">{text}</keystrokes>\n' for text, duration in commands)
    done = "" if complete is None else f"<task_complete>{str(complete).lower()}</task_complete>\n"
    return (
        f"<response>\n<analysis>\n{analysis}\n</analysis>\n<plan>\n{plan}\n</plan>\n"
        f"<commands>\n{keys}</commands>\n{done}</response>"
    )


def observation(output: str, warning: str, complete: bool, pending: bool) -> tuple[str, bool]:
    """The next prompt of Harbor's terminus-2 (XML parser) after a reply, and whether a completion is pending."""
    if complete:
        return (output if pending else f"{STATE}{output}{CONFIRM} This will trigger your solution to be graded and "
                "you won't be able to make any further corrections. If so, include "
                "<task_complete>true</task_complete> again."), True
    return (f"{WARNINGS}WARNINGS: {warning}\n\n{output}" if warning else output), False


def convert(messages: list[dict], parse_json, parse_xml, prompt: str) -> tuple[list[dict] | None, str, int]:
    """(XML conversation, drop reason, parse-error turns cut) of a Terminus-2 JSON trajectory."""
    start = first_prompt(messages)
    if start is None:
        return None, "unrecognized", 0
    task, state, first = start
    turns = messages[first:]
    roles = ["assistant", "user"] * (len(turns) // 2) + ["assistant"] * (len(turns) % 2)
    if not turns or [m["role"] for m in turns] != roles:
        return None, "unrecognized", 0
    last = parse_json(reply(turns[-1])[1])
    if turns[-1]["role"] != "assistant" or last.error or not last.is_task_complete:
        return None, "unfinished", 0
    out = [{"role": "user", "content": prompt.format(instruction=task, terminal_state=state)}]
    pending, cut = False, 0
    for i in range(0, len(turns), 2):
        after = (turns[i + 1]["content"] or "") if i + 1 < len(turns) else None
        if after is not None and after.startswith(PARSE_ERRORS):
            cut += 1
            continue
        reasoning, text = reply(turns[i])
        parsed = parse_json(text)
        if parsed.error:
            return None, "parse_mismatch", cut
        if not isinstance(parsed.analysis, str) or not isinstance(parsed.plan, str):
            return None, "xml_roundtrip", cut
        fields = (parsed.analysis.strip(), parsed.plan.strip(), [(c.keystrokes, c.duration) for c in parsed.commands])
        complete = parsed.is_task_complete if parsed.is_task_complete or '"task_complete"' in text else None
        xml = render_xml(*fields, complete)
        try:
            back = parse_xml(xml)
        except IndexError:  # Harbor's XML parser fails on a tag of only whitespace, e.g. "< >"
            return None, "xml_roundtrip", cut
        read_back = (back.analysis, back.plan, [(c.keystrokes, c.duration) for c in back.commands])
        if back.error or read_back != fields or back.is_task_complete != parsed.is_task_complete:
            return None, "xml_roundtrip", cut
        if not reasoning:
            return None, "no_reasoning", cut
        out.append({"role": "assistant", "content": xml, "reasoning_content": reasoning})
        if after is not None:
            output = terminal_output(after)
            if output is None:
                return None, "unrecognized", cut
            shown, pending = observation(output, back.warning, back.is_task_complete, pending)
            out.append({"role": "user", "content": shown})
    return out, "", cut


def _start_worker(tokenizer_dir: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.chat_template = sft.CHAT_TEMPLATE.read_text()
    prompt = Path(terminus_xml_plain_parser.__file__).with_name("templates") / "terminus-xml-plain.txt"
    _worker.update(
        tokenizer=tokenizer,
        parse_json=terminus_json_plain_parser.TerminusJSONPlainParser().parse_response,
        parse_xml=terminus_xml_plain_parser.TerminusXMLPlainParser().parse_response,
        prompt=prompt.read_text(),
    )


def _convert_row(row: dict) -> tuple[dict | None, str, Counter]:
    counts = Counter()
    try:
        if row["error"]:
            return None, "errored", counts
        w = _worker
        out, reason, counts["parse_error_turns_cut"] = convert(row["messages"], w["parse_json"], w["parse_xml"], w["prompt"])
        if out is None:
            return None, reason, counts
        ids, mask = sft.loss_mask(w["tokenizer"], out)
    except Exception as exc:  # noqa: BLE001 - one bad row must not stop its file; the manifest counts it
        return None, f"exception:{type(exc).__name__}", counts
    if max(len(run) for run in re.findall("1+", "".join(map(str, mask)))) > MAX_REPLY:
        return None, "reply_over_16k", counts
    if len(ids) > MAX_TOKENS:
        return None, "over_128k", counts
    # The teachers answered in JSON, so their reasoning may still plan a JSON reply.
    reasoning = [m["reasoning_content"] for m in out if m["role"] == "assistant"]
    counts["replies"] = len(reasoning)
    counts["reasoning_mentions_json"] = sum(bool(re.search(r"\bJSON\b", r)) for r in reasoning)
    counts["reasoning_mentions_task_complete"] = sum("task_complete" in r for r in reasoning)
    meta = {k: row[k] for k in ("id", "task", "teacher", "license")} | {"tokens": len(ids), "trained_tokens": sum(mask)}
    return {"messages": out, "metadata": meta, "instruction": first_prompt(row["messages"])[0]}, "", counts


def _parquet(path) -> list[dict]:
    return [row for batch in pq.ParquetFile(path).iter_batches(batch_size=64) for row in batch.to_pylist()]


def _rows(repo: str, revision: str, path: str) -> tuple[list[dict], int]:
    """(normalized rows, rows scanned) of one source file, downloaded into the Volume once (SWE-v3: its Terminus-2 rows)."""
    raw = DATA / "raw" / repo / path
    if repo == SWE_V3[0]:
        raw = raw.with_name(f"{raw.name}.terminus.json")
        if not raw.exists():
            with tempfile.TemporaryDirectory() as tmp:
                rows = _parquet(hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=tmp))
            terminus = [r for r in rows if (r["messages"][0]["content"] or "").startswith(TERMINUS)]
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(json.dumps({"scanned": len(rows), "rows": terminus}))
        cached = json.loads(raw.read_text())
        rows = [
            {"id": r["uuid"], "task": r["uuid"], "teacher": None, "license": r["license"], "error": None,
             "messages": r["messages"]}
            for r in cached["rows"]
        ]
        return rows, cached["scanned"]
    if not raw.exists():
        hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=DATA / "raw" / repo)
    rows = [
        {"id": r["trial_name"], "task": r["task"], "teacher": r["model"], "license": LICENSES[repo],
         "error": r.get("result"), "messages": r["conversations"]}
        for r in _parquet(raw)
    ]
    return rows, len(rows)


CPUS = 8


@app.function(
    image=image,
    volumes={str(DATA): volume, str(BENCHMARKS): modal.Volume.from_name("harbor-envs").read_only()},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=CPUS,
    memory=32768,
    timeout=4 * 3600,
    include_source=False,
)
def process(repo: str, revision: str, path: str, group: str) -> dict:
    """Convert one source file into candidates (/candidates/<repo>/<path>.jsonl); returns its counts."""
    rows, scanned = _rows(repo, revision, path)
    tokenizer = snapshot_download(base.SOURCE_MODEL, revision=base.SOURCE_REVISION, allow_patterns=["*.json", "*.txt"])
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(CPUS, mp_context=spawn, initializer=_start_worker, initargs=(tokenizer,)) as pool:
        results = list(pool.map(_convert_row, rows, chunksize=2))
    dropped, counts, kept = Counter(), Counter(), []
    for row, reason, extra in results:
        counts.update(extra)
        if row is None:
            dropped[reason] += 1
        else:
            kept.append(row)
    tasks, benchmarks = {r["metadata"]["task"]: r["instruction"] for r in kept}, general_sets.benchmarks(BENCHMARKS)
    found = general_sets.overlaps(tasks, benchmarks) | {
        task: f"near {bench} (5-gram containment {containment:.2f})"
        for task, (_, containment, bench) in general_sets.closest(tasks, benchmarks).items()
        if containment >= NEAR
    }
    dropped["contaminated"] = sum(r["metadata"]["task"] in found for r in kept)
    kept = [r for r in kept if r["metadata"]["task"] not in found]
    out = DATA / "candidates" / repo / f"{path}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps({"messages": r["messages"], "metadata": r["metadata"] | {"source": group}}) + "\n" for r in kept)
    )
    volume.commit()
    return {
        "repo": repo,
        "revision": revision,
        "path": path,
        "group": group,
        "candidates": str(out.relative_to(DATA)),
        "scanned": scanned,
        "rows": len(rows),
        "kept": len(kept),
        "dropped": dict(dropped),
        "counts": dict(counts),
        "contaminated": dict(sorted(found.items())[:5]),
        "benchmark_tasks": len(benchmarks),
        "tokens": [r["metadata"]["tokens"] for r in kept],
        "trained_tokens": [r["metadata"]["trained_tokens"] for r in kept],
    }


@app.function(image=image, volumes={str(DATA): volume}, cpu=4, memory=65536, timeout=3600, include_source=False)
def select(stats: list[dict], commit: str, name: str) -> dict:
    """A seeded sample of the candidates: each group's rows in a random order, up to its share of TOKENS."""
    volume.reload()
    out = DATA / name
    if out.exists():
        raise FileExistsError(f"{out} exists; a data set is written once")
    rng = random.Random(SEED)
    pool = sorted(
        (rng.random(), s["group"], s["candidates"], i, n, trained)
        for s in stats
        for i, (n, trained) in enumerate(zip(s["tokens"], s["trained_tokens"], strict=True))
    )
    chosen, picked, tokens = {}, Counter(), Counter()
    for _, group, file, line, n, trained in pool:
        if tokens[group] >= SHARES[group] * TOKENS:
            continue
        chosen.setdefault(file, set()).add(line)
        picked[group] += 1
        tokens.update({"all": n, group: n, f"{group}/trained": trained, "trained": trained})
    lines = [
        line
        for file, lines in sorted(chosen.items())
        for i, line in enumerate((DATA / file).read_text().splitlines())
        if i in lines
    ]
    rng.shuffle(lines)
    out.mkdir(parents=True)
    (out / "train.jsonl").write_text("".join(line + "\n" for line in lines))
    groups = {}
    for s in stats:
        g = groups.setdefault(
            s["group"],
            {"files": 0, "scanned": 0, "rows": 0, "candidates": 0, "candidate_tokens": 0, "dropped": Counter(),
             "counts": Counter(), "contaminated_examples": {}},
        )
        g["files"] += 1
        for key in ("scanned", "rows"):
            g[key] += s[key]
        g["candidates"] += s["kept"]
        g["candidate_tokens"] += sum(s["tokens"])
        g["dropped"].update(s["dropped"])
        g["counts"].update(s["counts"])
        g["contaminated_examples"].update(list(s["contaminated"].items())[: 5 - len(g["contaminated_examples"])])
    for group, g in groups.items():
        g |= {"selected": picked[group], "selected_tokens": tokens[group], "selected_trained_tokens": tokens[f"{group}/trained"]}
    manifest = {
        "name": name,
        "train": f"/{name}/train.jsonl",
        "rows": len(lines),
        "tokens": tokens["all"],
        "trained_tokens": tokens["trained"],
        "target": {"tokens": TOKENS, "shares": SHARES, "seed": SEED},
        "decontamination": {
            "benchmark_tasks": min(s["benchmark_tasks"] for s in stats),
            "rule": f"general_sets.overlaps(), or 5-gram containment >= {NEAR} with the closest benchmark prompt",
        },
        "limits": {"reply_tokens": MAX_REPLY, "trajectory_tokens": MAX_TOKENS},
        "sources": {repo: revision for repo, revision in (TERMINAL, SWE_V3, COLDSTART)},
        "harness": f"harbor@{harbor_config.HARBOR_REF} terminus-2, XML parser, interleaved thinking",
        "tokenizer": f"{base.SOURCE_MODEL}@{base.SOURCE_REVISION} with {sft.CHAT_TEMPLATE.name}",
        "stitch_commit": commit,
        "groups": groups,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(name: str = NAME, only: str = "") -> None:
    """--only keeps the source files whose path contains it (a trial run); its set is written under that name."""
    if not ENVS_DIR:
        raise SystemExit("Set EVAL_ENVS_DIR to the eval repository's envs/ folder (it has general_sets.py)")
    git = ["git", "-C", str(Path(__file__).parent)]
    commit = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
        commit += "-dirty"
    stats = list(process.starmap([f for f in FILES if only in f[2]]))
    manifest = select.remote(stats, commit, name)
    print(json.dumps(manifest, indent=2))
