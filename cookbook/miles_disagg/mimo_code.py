"""Write pinned MiMo code tasks for the Modal SWE mini-swe-agent adapter.

For each task, add `environment/setup.sh` (the adapter runs it; Harbor ran the
healthcheck command instead) and one `train.jsonl` row. Skip the tasks in
`mimo_code_excluded.tsv`. The recipe must set MODAL_SWE_CPUS=2 and
MODAL_SWE_MEMORY_MIB=8192 (the value in every `task.toml`).
"""

from __future__ import annotations

import json
import os
import shlex
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


def _setup_script(config: dict) -> str:
    """Run the MiMo setup, then delete unreachable git objects.

    Some images keep future commits as unreachable objects, and the agent can read
    them (the fix leaks). MiMo setup deletes them only when a branch or tag reaches
    them. Broken refs make gc fail, so delete them first. Skip if MiMo setup hid
    `.git`. `gc.cruftPacks=false` deletes unreachable objects instead of packing them.
    """
    workdir = shlex.quote(config["environment"]["workdir"])
    setup_command = config["environment"]["healthcheck"]["command"]
    return (
        f"#!/bin/bash\n{setup_command} || exit\n"
        f"cd {workdir} || exit\n"
        "[ -d .git ] || exit 0\n"
        "for ref in $(cd .git && find refs -type f); do\n"
        '  git rev-parse -q --verify "$ref^{object}" >/dev/null ||\n'
        '    git update-ref -d --no-deref "$ref"\n'
        "done\n"
        "git reflog expire --expire=now --all && "
        "git -c gc.pruneExpire=now -c gc.cruftPacks=false gc --quiet --prune=now\n"
    )


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
        shutil.copytree(source_dir, task_dir)
        config = tomllib.loads((task_dir / "task.toml").read_text())
        (task_dir / "environment" / "setup.sh").write_text(_setup_script(config))
        prompt_rows.append(
            {
                "prompt": (task_dir / "instruction.md").read_text(),
                "metadata": {
                    "instance_id": source_dir.name,
                    "task_dir": str(task_dir),
                    "sandbox_cwd": config["environment"]["workdir"],
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
