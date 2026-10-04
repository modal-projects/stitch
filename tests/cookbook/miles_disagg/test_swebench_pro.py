from __future__ import annotations

import hashlib
import subprocess

import pytest

from cookbook.miles_disagg.swebench_pro import (
    GRADE_PATCH_PATH,
    TASK_BASELINE_REF,
    _agent_start_commands,
    _parse_string_list,
    _patched_paths,
    _setup_script,
    _v2_agent_start_commands,
    _v2_verifier_script,
    _verifier_script,
    _verify_checksums,
    _write_v2_task,
)


def test_parse_string_list_rejects_non_string_items() -> None:
    assert _parse_string_list("['a', 'b']", "tests", "task") == ["a", "b"]
    with pytest.raises(TypeError, match="list of strings"):
        _parse_string_list("['a', 1]", "tests", "task")


def test_patched_paths_are_unique_and_sorted() -> None:
    patch = "\n".join(
        [
            "diff --git a/z.py b/z.py",
            "diff --git a/a.py b/a.py",
            "diff --git a/z.py b/z.py",
        ]
    )
    assert _patched_paths(patch) == ["a.py", "z.py"]


@pytest.mark.parametrize(
    "script",
    [_setup_script("true"), _verifier_script(["test_a.py", "test b.py"])],
)
def test_generated_shell_is_valid(script: str) -> None:
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_generated_verifier_python_is_valid() -> None:
    script = _verifier_script(["test_a.py"])
    source = script.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(source, "verifier.py", "exec")


def test_setup_gives_login_shells_the_image_path(tmp_path) -> None:
    """The restoring lines run against a stand-in for PID 1's environment and
    profile directory, then a login-style shell reads the result."""
    environ = tmp_path / "environ"
    environ.write_bytes(b"HOME=/root\0PATH=/go/bin:/usr/local/go/bin:/usr/bin:/bin\0")
    profile = tmp_path / "profile.d"
    profile.mkdir()
    script = _setup_script("true")
    restore = script[: script.index("cd /app")]
    restore = restore.replace("/proc/1/environ", str(environ)).replace(
        "/etc/profile.d", str(profile)
    )
    subprocess.run(["bash", "-c", restore], check=True)

    path = subprocess.run(
        ["bash", "-c", f"PATH=/usr/bin:/bin; . {profile}/zz-image-path.sh; echo $PATH"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert path == "/go/bin:/usr/local/go/bin:/usr/bin:/bin"


def test_agent_start_drops_the_fix_test_checkout() -> None:
    commands = "git reset --hard abc1234\ngit clean -fd\ngit checkout abc1234\ngit checkout def5678 -- test/a.py test/b.py\n"

    assert _agent_start_commands(commands, "task") == (
        "git reset --hard abc1234\ngit clean -fd\ngit checkout abc1234"
    )
    with pytest.raises(ValueError, match="fix test checkout"):
        _agent_start_commands("git reset --hard abc1234", "task")


def _git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _commit(repo, files: dict[str, str], message: str) -> str:
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _task_repo(tmp_path):
    """A repository whose clone, like the benchmark images', holds the fix commit and
    a remote: base has a bug and an old test, the fix commit fixes it and adds a test."""
    repo = tmp_path / "app"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, {"src.py": "VALUE = 0\n", "test_feature.py": "OLD\n"}, "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v-old", base)
    fix = _commit(
        repo,
        {"src.py": "VALUE = 1\n", "test_feature.py": "assert VALUE == 1\n"},
        "fix the bug (#42)",
    )
    _git(repo, "tag", "v-new", fix)
    _git(repo, "remote", "add", "origin", "https://example.invalid/upstream.git")
    _git(repo, "update-ref", "refs/remotes/origin/main", fix)
    _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
    return repo, base, fix


def _run_setup(repo, base: str, fix: str) -> None:
    commands = f"git reset --hard {base}\ngit clean -fd\ngit checkout -q {base}\ngit checkout {fix} -- test_feature.py"
    script = _setup_script(_agent_start_commands(commands, "task"))
    script = script.replace("cd /app", f"cd {repo}")
    subprocess.run(["bash", "-c", script], check=True, capture_output=True)


def test_setup_leaves_no_trace_of_the_fix(tmp_path) -> None:
    repo, base, fix = _task_repo(tmp_path)

    _run_setup(repo, base, fix)

    assert (repo / "test_feature.py").read_text() == "OLD\n"
    assert _git(repo, "remote") == ""
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", fix], capture_output=True
        ).returncode
        != 0
    )
    refs = _git(repo, "for-each-ref", "--format=%(refname)").split()
    assert "refs/tags/v-old" in refs and "refs/tags/v-new" not in refs
    assert not any(ref.startswith("refs/remotes/") for ref in refs)
    assert "refs/miles/task-baseline" in refs


def _run_verifier(tmp_path, repo, test_patch: str) -> str:
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test.patch").write_text(test_patch)
    (tests / "test_paths.txt").write_text("test_feature.py\n")
    (tests / "required_tests.json").write_text('["test_feature"]\n')
    (tests / "run_script.sh").write_text(
        'cd "$(dirname "$0")/../app" && python3 -c "exec(open(\'src.py\').read()); exec(open(\'test_feature.py\').read())" && echo PASSED test_feature || echo FAILED test_feature\n'
    )
    (tests / "parser.py").write_text(
        "import json, sys\n"
        "lines = open(sys.argv[1]).read().split()\n"
        "status = 'PASSED' if 'PASSED' in lines else 'FAILED'\n"
        "json.dump({'tests': [{'name': 'test_feature', 'status': status}]}, open(sys.argv[3], 'w'))\n"
    )
    logs = tmp_path / "logs"
    script = (
        _verifier_script(["test_feature.py"])
        .replace("cd /app", f"cd {repo}")
        .replace("/tests/", f"{tests}/")
        .replace("/logs/verifier", str(logs))
    )
    subprocess.run(["bash", "-c", script], capture_output=True, check=True)
    return (logs / "reward.txt").read_text().strip()


@pytest.mark.parametrize(
    ("policy_files", "reward"),
    [
        ({"src.py": "VALUE = 1\n"}, "1"),
        ({}, "0"),
        # Gaming the hidden test does not help: the benchmark's version replaces it.
        ({"test_feature.py": "pass\n"}, "0"),
        ({"src.py": "VALUE = 1\n", "test_feature.py": "assert False\n"}, "1"),
    ],
)
def test_verifier_grades_the_policy_against_the_fix_tests(
    tmp_path, policy_files, reward
) -> None:
    repo, base, fix = _task_repo(tmp_path)
    test_patch = subprocess.run(
        ["git", "-C", str(repo), "diff", base, fix, "--", "test_feature.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    _run_setup(repo, base, fix)
    for name, content in policy_files.items():
        (repo / name).write_text(content)

    assert _run_verifier(tmp_path, repo, test_patch) == reward


# SWE-bench Pro V2: Scale's Harbor tasks behind our setup and verifier preamble.


def _v2_harbor_test(tests) -> None:
    """A stand-in for V2's tests/test.sh: reward 1 when the fix is in place."""
    (tests / "harbor_test.sh").write_text(
        'mkdir -p "$LOGS"; cd "$REPO" && grep -q "VALUE = 1" src.py'
        ' && echo 1 > "$LOGS/reward.txt" || echo 0 > "$LOGS/reward.txt"\n'
    )


def _run_v2_verifier(tmp_path, repo, *, graded_patch: bytes | None = None) -> str:
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    _v2_harbor_test(tests)
    logs = tmp_path / "logs"
    grade_patch = tmp_path / "grade" / "policy.patch"
    if graded_patch is not None:
        grade_patch.parent.mkdir(exist_ok=True)
        grade_patch.write_bytes(graded_patch)
    script = (
        _v2_verifier_script()
        .replace("cd /app", f"cd {repo}")
        .replace(GRADE_PATCH_PATH, str(grade_patch))
        .replace("/tests/", f"{tests}/")
        .replace("/logs/verifier", str(logs))
    )
    subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        check=True,
        env={"PATH": "/usr/bin:/bin", "LOGS": str(logs), "REPO": str(repo)},
    )
    return (logs / "reward.txt").read_text().strip()


def _v2_task_repo(tmp_path):
    """A V2-style image repository: already at the base commit, history sanitised,
    with the task baseline recorded by our setup."""
    repo = tmp_path / "app"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    base = _commit(repo, {"src.py": "VALUE = 0\n", "notes.txt": "keep\n"}, "base")
    script = _setup_script(_v2_agent_start_commands(base)).replace(
        "cd /app", f"cd {repo}"
    )
    subprocess.run(["bash", "-c", script], check=True, capture_output=True)
    return repo, base


def test_v2_setup_refuses_an_image_away_from_the_base_commit(tmp_path) -> None:
    repo, base = _v2_task_repo(tmp_path)
    _commit(repo, {"src.py": "VALUE = 2\n"}, "drift")
    script = _setup_script(_v2_agent_start_commands(base)).replace(
        "cd /app", f"cd {repo}"
    )

    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    assert result.returncode != 0
    assert "not the task's base commit" in result.stderr


@pytest.mark.parametrize(
    ("policy_files", "reward"),
    [({"src.py": "VALUE = 1\n"}, "1"), ({}, "0"), ({"other.py": "VALUE = 1\n"}, "0")],
)
def test_v2_verifier_grades_the_policy_in_its_own_sandbox(
    tmp_path, policy_files, reward
) -> None:
    repo, _ = _v2_task_repo(tmp_path)
    for name, content in policy_files.items():
        (repo / name).write_text(content)

    assert _run_v2_verifier(tmp_path, repo) == reward


@pytest.mark.parametrize(
    ("policy_files", "commit", "reward"),
    [
        ({"src.py": "VALUE = 1\n"}, False, "1"),
        # A policy that commits its change is graded on the same diff.
        ({"src.py": "VALUE = 1\n"}, True, "1"),
        ({}, False, "0"),
    ],
)
def test_v2_fresh_sandbox_grades_only_the_captured_patch(
    tmp_path, policy_files, commit, reward
) -> None:
    """The agent's Sandbox yields its diff from the baseline; a fresh baseline tree
    receives only that patch, whatever else the agent's Sandbox holds."""
    agent_repo, _ = _v2_task_repo(tmp_path / "agent")
    for name, content in policy_files.items():
        (agent_repo / name).write_text(content)
    if commit:
        _git(agent_repo, "add", "-A")
        _git(
            agent_repo,
            "-c",
            "user.name=a",
            "-c",
            "user.email=a@a",
            "commit",
            "-qm",
            "x",
        )
    (agent_repo / "untracked_scratch.txt").write_text("agent-only state\n")
    _git(agent_repo, "add", "-N", ".")
    patch = subprocess.run(
        [
            "git",
            "-C",
            str(agent_repo),
            "diff",
            "--binary",
            TASK_BASELINE_REF,
            "--",
            ".",
        ],
        capture_output=True,
        check=True,
    ).stdout

    fresh_repo, _ = _v2_task_repo(tmp_path / "fresh")
    assert (
        _run_v2_verifier(tmp_path / "fresh", fresh_repo, graded_patch=patch) == reward
    )


def test_v2_fresh_sandbox_grades_an_unapplied_patch_on_the_baseline(tmp_path) -> None:
    repo, _ = _v2_task_repo(tmp_path)
    bad = b"diff --git a/src.py b/src.py\n--- a/src.py\n+++ b/src.py\n@@ -1 +1 @@\n-NOPE\n+VALUE = 1\n"

    assert _run_v2_verifier(tmp_path, repo, graded_patch=bad) == "0"
    assert (repo / "src.py").read_text() == "VALUE = 0\n"


def test_v2_fresh_sandbox_keeps_the_fix_when_a_service_file_conflicts(tmp_path) -> None:
    """NodeBB's Redis writes its log under the repository in every Sandbox, so the
    captured patch creates a file the fresh tree already has; V2's replay still applies
    the rest of the patch."""
    agent_repo, _ = _v2_task_repo(tmp_path / "agent")
    (agent_repo / "src.py").write_text("VALUE = 1\n")
    (agent_repo / "appendonlydir").mkdir()
    (agent_repo / "appendonlydir" / "a.aof").write_text("agent redis state\n")
    _git(agent_repo, "add", "-N", ".")
    patch = subprocess.run(
        [
            "git",
            "-C",
            str(agent_repo),
            "diff",
            "--binary",
            TASK_BASELINE_REF,
            "--",
            ".",
        ],
        capture_output=True,
        check=True,
    ).stdout
    fresh_repo, _ = _v2_task_repo(tmp_path / "fresh")
    (fresh_repo / "appendonlydir").mkdir()
    (fresh_repo / "appendonlydir" / "a.aof").write_text("fresh redis state\n")

    assert _run_v2_verifier(tmp_path / "fresh", fresh_repo, graded_patch=patch) == "1"
    assert (fresh_repo / "appendonlydir" / "a.aof").read_text() == "fresh redis state\n"


def _v2_source_task(root, name="instance_a__b-1"):
    source = root / name
    (source / "environment").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "solution").mkdir()
    image = f"ghcr.io/scaleapi/swe-bench_pro-v2:{name}"
    (source / "environment" / "Dockerfile").write_text(f"FROM {image}\n")
    (source / "task.toml").write_text(
        f'[verifier]\ntimeout_sec = 3000.0\n[agent]\nnetwork_mode = "no-network"\n'
        f'[environment]\ndocker_image = "{image}"\ncpus = 1\nmemory_mb = 4096\n'
    )
    (source / "tests" / "test.sh").write_text("echo v2 verifier\n")
    (source / "tests" / "config.json").write_text("{}\n")
    (source / "solution" / "gold_patch.diff").write_text("")
    (source / "instruction.md").write_text("Fix the bug in /app.\n")
    return source


def test_v2_task_keeps_scales_verifier_behind_our_preamble(tmp_path) -> None:
    source = _v2_source_task(tmp_path / "v2")
    task_dir = tmp_path / "tasks" / source.name

    instruction = _write_v2_task(source, task_dir, "abc1234")

    assert instruction == "Fix the bug in /app.\n"
    assert (task_dir / "tests" / "harbor_test.sh").read_text() == "echo v2 verifier\n"
    assert (task_dir / "tests" / "test.sh").read_text() == _v2_verifier_script()
    setup = (task_dir / "environment" / "setup.sh").read_text()
    assert "abc1234" in setup and TASK_BASELINE_REF in setup
    assert (
        (task_dir / "environment" / "Dockerfile")
        .read_text()
        .startswith("FROM ghcr.io/scaleapi/swe-bench_pro-v2:")
    )


def test_v2_task_must_keep_the_agent_offline(tmp_path) -> None:
    source = _v2_source_task(tmp_path / "v2")
    toml = (source / "task.toml").read_text().replace("no-network", "public")
    (source / "task.toml").write_text(toml)

    with pytest.raises(RuntimeError, match="offline"):
        _write_v2_task(source, tmp_path / "task", "abc1234")


def test_v2_checksums_must_match(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("a\n")
    good = hashlib.sha256(b"a\n").hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{good}  a.txt\n")
    _verify_checksums(tmp_path)

    (tmp_path / "a.txt").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _verify_checksums(tmp_path)


def test_harness_and_tasks_agree_on_the_grading_contract() -> None:
    from cookbook.miles_disagg.modal_swe import agent

    assert agent._TASK_BASELINE_REF == TASK_BASELINE_REF
    assert f"{agent._GRADE_PATCH_DIR}/{agent._GRADE_PATCH_NAME}" == GRADE_PATCH_PATH
