"""Materialize pinned SWE-bench Pro rows as executable Harbor tasks."""

from __future__ import annotations

import ast
import json
import shlex
import subprocess
from pathlib import Path

EVALUATOR_REVISION = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"


def _parse_string_list(value: str, field: str, instance_id: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(
            f"{instance_id}: {field} is not a Python list literal"
        ) from error
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise TypeError(f"{instance_id}: {field} must be a list of strings")
    return parsed


def _patched_paths(test_patch: str) -> list[str]:
    prefix = "diff --git a/"
    paths = []
    for line in test_patch.splitlines():
        if line.startswith(prefix):
            path, _, _ = line[len(prefix) :].partition(" b/")
            if path:
                paths.append(path)
    return sorted(set(paths))


def _setup_script(before_repo_set_cmd: str) -> str:
    return rf"""#!/bin/bash
set -euo pipefail
cd /app
{before_repo_set_cmd}

baseline_tree=$(git write-tree)
baseline_commit=$(
    printf '%s\n' 'Miles SWE-bench Pro policy baseline' |
        GIT_AUTHOR_NAME=Miles \
        GIT_AUTHOR_EMAIL=miles@example.invalid \
        GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' \
        GIT_COMMITTER_NAME=Miles \
        GIT_COMMITTER_EMAIL=miles@example.invalid \
        GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
        git commit-tree "$baseline_tree" -p HEAD
)
git update-ref refs/miles/task-baseline "$baseline_commit"
"""


def _verifier_script(selected_tests: list[str]) -> str:
    selected = shlex.quote(",".join(selected_tests))
    return rf"""#!/bin/bash
set -u

write_zero() {{
    mkdir -p /logs/verifier
    printf '0\n' > /logs/verifier/reward.txt
}}

cd /app || exit 1
baseline=$(git rev-parse refs/miles/task-baseline) || exit 1
git add -N . >/dev/null 2>&1 || true
git diff --name-only "$baseline" -- > /tmp/miles_changed_paths.txt || exit 1
if grep -Fxf /tests/protected_test_paths.txt /tmp/miles_changed_paths.txt \
        >/tmp/miles_modified_tests.txt; then
    echo "Policy modified benchmark test files:"
    cat /tmp/miles_modified_tests.txt
    write_zero
    exit 0
fi

git diff --binary "$baseline" -- . > /tmp/miles_policy.patch || exit 1
git reset --hard "$baseline" >/dev/null || exit 1
git clean -fd >/dev/null || exit 1
if ! git apply --whitespace=nowarn /tmp/miles_policy.patch; then
    echo "Policy patch could not be applied to the canonical task baseline."
    write_zero
    exit 0
fi

bash /tests/run_script.sh {selected} \
    > /tmp/miles_test_stdout.log \
    2> /tmp/miles_test_stderr.log || true
python3 /tests/parser.py \
    /tmp/miles_test_stdout.log \
    /tmp/miles_test_stderr.log \
    /tmp/miles_test_results.json || exit 1

tail -c 2000 /tmp/miles_test_stdout.log || true
tail -c 2000 /tmp/miles_test_stderr.log || true
python3 - <<'PY'
import json
from pathlib import Path

required = set(json.loads(Path("/tests/required_tests.json").read_text()))
report = json.loads(Path("/tmp/miles_test_results.json").read_text())
passed = {{
    test["name"]
    for test in report.get("tests", [])
    if test.get("status") == "PASSED"
}}
reward = int(bool(required) and required.issubset(passed))
Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
Path("/logs/verifier/reward.txt").write_text(f"{{reward}}\n")
print(
    f"SWE-bench Pro verifier: passed_required={{len(required & passed)}}/"
    f"{{len(required)}} reward={{reward}}"
)
PY
"""


def _checkout_evaluator(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(source_root), "init"], check=True)
    has_origin = (
        subprocess.run(
            ["git", "-C", str(source_root), "remote", "get-url", "origin"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "remote",
            "set-url" if has_origin else "add",
            "origin",
            "https://github.com/scaleapi/SWE-bench_Pro-os.git",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(source_root),
            "fetch",
            "--depth=1",
            "origin",
            EVALUATOR_REVISION,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source_root), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )


def prepare_swebench_pro(data_root: Path) -> Path:
    """Write the pinned 731-task benchmark and return its prompt JSONL path."""
    from datasets import load_dataset

    source_revision = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
    tasks_root = data_root / "tasks"
    source_root = data_root / "SWE-bench_Pro-os"
    data_root.mkdir(parents=True, exist_ok=True)
    _checkout_evaluator(source_root)

    source_rows = load_dataset(
        "ScaleAI/SWE-bench_Pro",
        revision=source_revision,
        split="test",
    )
    if len(source_rows) != 731:
        raise RuntimeError(
            f"Expected 731 SWE-bench Pro test tasks; got {len(source_rows)}"
        )

    tasks_root.mkdir(parents=True, exist_ok=True)
    prompt_rows = []
    instance_ids = set()
    for source in source_rows:
        instance_id = source["instance_id"]
        if instance_id in instance_ids:
            raise RuntimeError(f"Duplicate SWE-bench Pro task: {instance_id}")
        instance_ids.add(instance_id)

        official_assets = source_root / "run_scripts" / instance_id
        run_script = official_assets / "run_script.sh"
        parser_script = official_assets / "parser.py"
        if not run_script.is_file() or not parser_script.is_file():
            raise FileNotFoundError(
                f"{instance_id}: missing official run_script.sh or parser.py"
            )

        selected_tests = _parse_string_list(
            source["selected_test_files_to_run"],
            "selected_test_files_to_run",
            instance_id,
        )
        required_tests = sorted(
            set(
                _parse_string_list(source["fail_to_pass"], "fail_to_pass", instance_id)
                + _parse_string_list(
                    source["pass_to_pass"], "pass_to_pass", instance_id
                )
            )
        )
        if not selected_tests or not required_tests:
            raise RuntimeError(
                f"{instance_id}: selected and required tests must be non-empty"
            )

        task_dir = tasks_root / instance_id
        environment_dir = task_dir / "environment"
        tests_dir = task_dir / "tests"
        environment_dir.mkdir(parents=True, exist_ok=True)
        tests_dir.mkdir(parents=True, exist_ok=True)
        (environment_dir / "Dockerfile").write_text(
            f"FROM jefzda/sweap-images:{source['dockerhub_tag']}\n"
        )
        (environment_dir / "setup.sh").write_text(
            _setup_script(source["before_repo_set_cmd"])
        )
        (tests_dir / "test.sh").write_text(_verifier_script(selected_tests))
        (tests_dir / "run_script.sh").write_text(run_script.read_text())
        (tests_dir / "parser.py").write_text(parser_script.read_text())
        (tests_dir / "required_tests.json").write_text(
            json.dumps(required_tests) + "\n"
        )
        (tests_dir / "protected_test_paths.txt").write_text(
            "".join(f"{path}\n" for path in _patched_paths(source["test_patch"]))
        )
        (task_dir / "task.toml").write_text("[verifier]\ntimeout_sec = 3600\n")

        prompt_rows.append(
            {
                "prompt": (
                    f"{source['problem_statement']}\n\n"
                    f"Requirements:\n{source['requirements']}\n\n"
                    f"New interfaces introduced:\n{source['interface']}"
                ),
                "metadata": {
                    "instance_id": instance_id,
                    "task_dir": str(task_dir),
                    "sandbox_cwd": "/app",
                    "agent_name": "mini-swe-agent",
                    "source_dataset": "ScaleAI/SWE-bench_Pro",
                    "source_revision": source_revision,
                    "split": "test",
                    "repo": source["repo"],
                    "repo_language": source["repo_language"],
                },
            }
        )

    prompt_path = data_root / "test.jsonl"
    prompt_path.write_text("".join(json.dumps(row) + "\n" for row in prompt_rows))
    (data_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": "ScaleAI/SWE-bench_Pro",
                "dataset_revision": source_revision,
                "evaluator": "scaleapi/SWE-bench_Pro-os",
                "evaluator_revision": EVALUATOR_REVISION,
                "split": "test",
                "tasks": len(prompt_rows),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(prompt_rows)} SWE-bench Pro tasks at {prompt_path}")
    return prompt_path
