"""Materialize pinned SWE-bench Pro rows as executable Harbor tasks."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from cookbook.miles_disagg.task_history import PRUNE_LATER_HISTORY

EVALUATOR_REVISION = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"

# SWE-bench Pro V2 as Scale released it (2026-09-22): the HuggingFace rows and the
# Harbor task directories in the benchmark's repository, pinned together.
V2_DATASET_REVISION = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"
V2_REPOSITORY_REVISION = "66f92766bba642462d4bbe5479e83f91f9211862"
V2_TASKS = 642
# Every task's setup records the tree the agent starts from under this ref.
TASK_BASELINE_REF = "refs/miles/task-baseline"
# A grader that runs the verifier in a fresh Sandbox puts the policy's patch here.
GRADE_PATCH_PATH = "/tmp/miles-grade/policy.patch"


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


_FIX_TEST_CHECKOUT = re.compile(r"git checkout [0-9a-f]{7,40} -- \S")


def _agent_start_commands(before_repo_set_cmd: str, instance_id: str) -> str:
    """The benchmark's repository setup without its last line, which checks out the
    fix commit's test files. The benchmark's agent starts from the base commit, and its
    evaluator runs that line only after applying the agent's patch."""
    lines = before_repo_set_cmd.strip().splitlines()
    if not lines or not _FIX_TEST_CHECKOUT.match(lines[-1].strip()):
        raise ValueError(
            f"{instance_id}: before_repo_set_cmd must end with the fix test checkout"
        )
    return "\n".join(lines[:-1])


def _setup_script(agent_start_commands: str) -> str:
    return rf"""#!/bin/bash
set -euo pipefail

# The agent and the verifier run in login shells, where /etc/profile resets PATH and
# drops the toolchain directories the image adds (Go's /usr/local/go/bin). The
# benchmark's evaluator runs with the image's PATH: give every login shell that PATH,
# read from the Sandbox's first process, which holds the image's environment.
image_path=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^PATH=//p') || true
if [ -n "$image_path" ]; then
    printf 'export PATH=%q\n' "$image_path" > /etc/profile.d/zz-image-path.sh
fi

cd /app
{agent_start_commands}

# The image's clone carries the repository's later history, the fix commit included,
# where `git log --all` finds it. Keep only what the task starts from: drop every ref
# HEAD does not contain, every remote, and the objects only they reached.
{PRUNE_LATER_HISTORY}
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
git diff --binary "$baseline" -- . > /tmp/miles_policy.patch || exit 1
git reset --hard "$baseline" >/dev/null || exit 1
git clean -fd >/dev/null || exit 1
# An unchanged tree is still graded, as the benchmark's evaluator grades an empty patch.
if [ -s /tmp/miles_policy.patch ] &&
        ! git apply --whitespace=nowarn /tmp/miles_policy.patch; then
    echo "Policy patch could not be applied to the canonical task baseline."
    write_zero
    exit 0
fi

# As the benchmark's evaluator does after the agent's patch, put the fix commit's test
# files in place, whatever the policy did to them: the baseline's versions plus the
# benchmark's test patch. A test patch that does not apply is the task's fault, not
# the policy's, so it ends without a reward.
while read -r path; do
    [ -n "$path" ] || continue
    if git cat-file -e "$baseline:$path" 2>/dev/null; then
        git checkout "$baseline" -- "$path" || exit 1
    else
        rm -rf -- "$path"
    fi
done < /tests/test_paths.txt
if ! git apply --whitespace=nowarn /tests/test.patch; then
    echo "The benchmark test patch does not apply to the task baseline."
    exit 1
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


def _checkout_evaluator(source_root: Path, revision: str = EVALUATOR_REVISION) -> None:
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
            revision,
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
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="swebench-pro-evaluator-") as temp_dir:
        source_root = Path(temp_dir) / "SWE-bench_Pro-os"
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
                    _parse_string_list(
                        source["fail_to_pass"], "fail_to_pass", instance_id
                    )
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
                _setup_script(
                    _agent_start_commands(source["before_repo_set_cmd"], instance_id)
                )
            )
            (tests_dir / "test.sh").write_text(_verifier_script(selected_tests))
            (tests_dir / "run_script.sh").write_text(run_script.read_text())
            (tests_dir / "parser.py").write_text(parser_script.read_text())
            (tests_dir / "required_tests.json").write_text(
                json.dumps(required_tests) + "\n"
            )
            # The fix commit's test files reach the Sandbox with the verifier only.
            test_paths = _patched_paths(source["test_patch"])
            checkout = source["before_repo_set_cmd"].strip().splitlines()[-1]
            if sorted(set(checkout.split(" -- ", 1)[1].split())) != test_paths:
                raise RuntimeError(
                    f"{instance_id}: the fix test checkout and test_patch name "
                    "different files"
                )
            (tests_dir / "test.patch").write_text(source["test_patch"])
            (tests_dir / "test_paths.txt").write_text(
                "".join(f"{path}\n" for path in test_paths)
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


def _v2_agent_start_commands(base_commit: str) -> str:
    """V2 images are already at the task's base commit with a sanitised history, and
    V2 runs no repository setup before the agent; only check that the image agrees."""
    return rf"""head=$(git rev-parse HEAD)
if [ "$head" != {shlex.quote(base_commit)} ]; then
    echo "the image is at $head, not the task's base commit {base_commit}" >&2
    exit 1
fi"""


def _v2_verifier_script() -> str:
    """The policy's patch on the task baseline, then V2's own verifier unchanged.

    A fresh-Sandbox grader puts the patch at ``GRADE_PATCH_PATH`` in a Sandbox whose
    tree is the baseline, and it is applied as V2's re-grade (``patch_replay``) applies
    a captured diff to a pristine image: ``git apply``, else a three-way apply, else a
    fuzzy ``patch``, and the verifier grades whatever applied. Files a service in the
    image writes under the repository (NodeBB's Redis log) are in both trees, so a
    strict apply would refuse the patch. In the agent's own Sandbox the patch is the
    policy's diff from the baseline, applied strictly to a clean baseline tree.
    """
    return rf"""#!/bin/bash
set -u

write_zero() {{
    mkdir -p /logs/verifier
    printf '0\n' > /logs/verifier/reward.txt
}}

cd /app || exit 1
patch={GRADE_PATCH_PATH}
if [ -f "$patch" ]; then
    if [ -s "$patch" ]; then
        git apply --verbose "$patch" ||
            git apply --3way "$patch" ||
            patch --fuzz=3 -p1 -i "$patch" < /dev/null ||
            echo "The policy patch applied only in part."
    fi
else
    baseline=$(git rev-parse {TASK_BASELINE_REF}) || exit 1
    patch=/tmp/miles_policy.patch
    git add -N . >/dev/null 2>&1 || true
    git diff --binary "$baseline" -- . > "$patch" || exit 1
    git reset --hard "$baseline" >/dev/null || exit 1
    git clean -fd >/dev/null || exit 1
    # An unchanged tree is still graded, as the benchmark grades an empty patch.
    if [ -s "$patch" ] && ! git apply --whitespace=nowarn "$patch"; then
        echo "Policy patch could not be applied to the canonical task baseline."
        write_zero
        exit 0
    fi
fi
exec bash /tests/harbor_test.sh
"""


def _verify_checksums(root: Path) -> None:
    """Every file under ``root`` that its SHA256SUMS lists must match."""
    listed = 0
    for line in (root / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = root / name.lstrip("*")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"checksum mismatch: {path}")
        listed += 1
    if listed == 0:
        raise RuntimeError(f"{root}/SHA256SUMS lists no files")


def _write_v2_task(source: Path, task_dir: Path, base_commit: str) -> str:
    """Copy one V2 Harbor task and put our setup and verifier preamble around it.
    Returns the task's instruction, which is what the agent sees."""
    spec = tomllib.loads((source / "task.toml").read_text())
    image = spec["environment"]["docker_image"]
    dockerfile = (source / "environment" / "Dockerfile").read_text().split()
    if dockerfile[:2] != ["FROM", image]:
        raise RuntimeError(
            f"{source.name}: Dockerfile and task.toml name different images"
        )
    if spec["agent"]["network_mode"] != "no-network":
        raise RuntimeError(f"{source.name}: V2's agent phase must be offline")
    shutil.copytree(source, task_dir, dirs_exist_ok=True)
    (task_dir / "tests" / "test.sh").rename(task_dir / "tests" / "harbor_test.sh")
    (task_dir / "tests" / "test.sh").write_text(_v2_verifier_script())
    (task_dir / "environment" / "setup.sh").write_text(
        _setup_script(_v2_agent_start_commands(base_commit))
    )
    return (source / "instruction.md").read_text()


def prepare_swebench_pro_v2(data_root: Path) -> Path:
    """Write SWE-bench Pro V2's 642 tasks and return the prompt JSONL path.

    Each task is V2's Harbor directory: its image, its instruction (the agent's
    prompt), and its verifier, which runs after our preamble grades the policy's
    patch. Our setup gives login shells the image's PATH and records the baseline.
    """
    from datasets import load_dataset

    tasks_root = data_root / "tasks"
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="swebench-pro-v2-") as temp_dir:
        source_root = Path(temp_dir) / "SWE-bench_Pro-os"
        _checkout_evaluator(source_root, revision=V2_REPOSITORY_REVISION)
        v2 = source_root / "v2"
        _verify_checksums(v2)
        rows = {
            row["instance_id"]: row
            for row in load_dataset(
                "ScaleAI/SWE-bench_Pro", revision=V2_DATASET_REVISION, split="test"
            )
        }
        sources = sorted(path for path in (v2 / "tasks").iterdir() if path.is_dir())
        if len(rows) != V2_TASKS or {path.name for path in sources} != set(rows):
            raise RuntimeError(
                f"V2 rows ({len(rows)}) and task directories ({len(sources)}) disagree"
            )

        tasks_root.mkdir(parents=True, exist_ok=True)
        prompt_rows = []
        for source in sources:
            row = rows[source.name]
            task_dir = tasks_root / source.name
            instruction = _write_v2_task(source, task_dir, row["base_commit"])
            prompt_rows.append(
                {
                    "prompt": instruction,
                    "metadata": {
                        "instance_id": source.name,
                        "task_dir": str(task_dir),
                        "sandbox_cwd": "/app",
                        "agent_name": "mini-swe-agent",
                        "source_dataset": "ScaleAI/SWE-bench_Pro",
                        "source_revision": V2_DATASET_REVISION,
                        "benchmark_revision": V2_REPOSITORY_REVISION,
                        "split": "test",
                        "repo": row["repo"],
                        "repo_language": row["repo_language"],
                    },
                }
            )

    prompt_path = data_root / "test.jsonl"
    prompt_path.write_text("".join(json.dumps(row) + "\n" for row in prompt_rows))
    (data_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": "ScaleAI/SWE-bench_Pro",
                "version": "2.0.0",
                "dataset_revision": V2_DATASET_REVISION,
                "benchmark": "scaleapi/SWE-bench_Pro-os v2/tasks",
                "benchmark_revision": V2_REPOSITORY_REVISION,
                "split": "test",
                "tasks": len(prompt_rows),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Prepared {len(prompt_rows)} SWE-bench Pro V2 tasks at {prompt_path}")
    return prompt_path
