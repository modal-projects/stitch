"""Write pinned MiMo code tasks for the Miles Harbor agent function.

Copy each Harbor task, add a git prune to its healthcheck command, limit the
agent's network, and write one `train.jsonl` row. Skip the tasks in
`mimo_code_excluded.tsv`. Point HARBOR_TASKS_DIR at `<data_root>/tasks`, and put
the model server's host in HARBOR_AGENT_ALLOWED_HOSTS.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

SOURCE_DATASET = "FineEnvs/MiMo-V2.6-RL-harbor-code"
SOURCE_REVISION = "e60dca3794baff1a099d505c02c175f42727e8a4"
EXCLUDED = Path(__file__).with_name("mimo_code_excluded.tsv")


def _excluded() -> set[str]:
    return {
        line.split("\t", 1)[0]
        for line in EXCLUDED.read_text().splitlines()
        if line and not line.startswith("#")
    }


def _prune(workdir: str) -> str:
    """Delete unreachable git objects once, after MiMo setup.

    Some images keep future commits as unreachable objects, and the agent can read
    them (the fix leaks). MiMo setup deletes them only when a branch or tag reaches
    them. Broken refs make gc fail, so delete them first. Skip if MiMo setup hid
    `.git`. `gc.cruftPacks=false` deletes unreachable objects instead of packing them.
    """
    if not re.fullmatch(r"[\w/.-]+", workdir):
        raise ValueError(f"Unexpected workdir: {workdir!r}")
    return (
        "test -f /var/lib/mimo/pruned && exit 0; "
        f"cd {workdir} || exit 1; "
        "if [ -d .git ]; then "
        "for ref in $(cd .git && find refs -type f); do "
        'git rev-parse -q --verify "$ref^{object}" >/dev/null || '
        'git update-ref -d --no-deref "$ref"; '
        "done; "
        "git reflog expire --expire=now --all && "
        "git -c gc.pruneExpire=now -c gc.cruftPacks=false gc --quiet --prune=now "
        "|| exit 1; "
        "fi; "
        "touch /var/lib/mimo/pruned"
    )


def _copy_task(source_dir: Path, task_dir: Path) -> dict:
    """Copy one task, append the prune to its healthcheck, and limit agent network.

    While the agent runs, the sandbox can reach only the hosts in Miles'
    HARBOR_AGENT_ALLOWED_HOSTS (the model server), so the agent cannot look up the
    fix online. Setup and grading keep the task's public network.
    """
    shutil.copytree(source_dir, task_dir)
    path = task_dir / "task.toml"
    text, config = path.read_text(), tomllib.loads(path.read_text())
    healthcheck = config["environment"]["healthcheck"]
    prune = _prune(config["environment"]["workdir"])
    healthcheck["command"] = f"{healthcheck['command']} && bash -c '{prune}'"
    head, header, tail = text.partition("[environment.healthcheck]\n")
    line = next(x for x in tail.splitlines(True) if x.startswith("command = "))
    new_line = f"command = {json.dumps(healthcheck['command'])}\n"
    text = head + header + tail.replace(line, new_line, 1)
    config["agent"]["network_mode"] = "allowlist"
    text = text.replace("[agent]\n", '[agent]\nnetwork_mode = "allowlist"\n', 1)
    if tomllib.loads(text) != config:
        raise RuntimeError(f"task.toml edit changed more than intended: {path}")
    path.write_text(text)
    return config


def prepare_mimo_code(data_root: Path) -> Path:
    """Write the pinned MiMo code training tasks and return their prompt JSONL path."""
    with tempfile.TemporaryDirectory(prefix="mimo-code-") as source:
        # Download with git: per-file Hub downloads hit rate limits.
        # Skip LFS: the task files are normal git files.
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        url = f"https://huggingface.co/datasets/{SOURCE_DATASET}"
        for command in (
            ["init", "-q"],
            ["fetch", "-q", "--depth=1", url, SOURCE_REVISION],
            ["checkout", "-q", "FETCH_HEAD"],
        ):
            subprocess.run(["git", "-C", source, *command], check=True, env=env)
        return _write_tasks(Path(source) / "tasks", data_root)


def _write_tasks(source: Path, data_root: Path) -> Path:
    source_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    excluded = _excluded()
    if missing := excluded - {path.name for path in source_dirs}:
        raise RuntimeError(f"Excluded tasks not in the dataset: {sorted(missing)[:5]}")

    tasks_root = data_root / "tasks"
    shutil.rmtree(tasks_root, ignore_errors=True)
    tasks_root.mkdir(parents=True)
    prompt_rows = []
    for source_dir in source_dirs:
        if source_dir.name in excluded:
            continue
        task_dir = tasks_root / source_dir.name
        config = _copy_task(source_dir, task_dir)
        prompt_rows.append(
            {
                "prompt": (task_dir / "instruction.md").read_text(),
                "metadata": {
                    "instance_id": source_dir.name,
                    "agent_name": "mini-swe-agent",
                    "source_dataset": SOURCE_DATASET,
                    "source_revision": SOURCE_REVISION,
                    "split": "train",
                    "repo_language": config["metadata"]["category"],
                },
            }
        )

    prompt_path = data_root / "train.jsonl"
    prompt_path.write_text("".join(json.dumps(row) + "\n" for row in prompt_rows))
    (data_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": SOURCE_DATASET,
                "dataset_revision": SOURCE_REVISION,
                "split": "train",
                "tasks": len(prompt_rows),
                "excluded": len(excluded),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(prompt_rows)} MiMo code tasks ({len(excluded)} excluded)")
    return prompt_path
