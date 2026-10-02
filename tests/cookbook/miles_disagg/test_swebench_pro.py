from __future__ import annotations

import subprocess

import pytest

from cookbook.miles_disagg.swebench_pro import (
    _agent_start_commands,
    _parse_string_list,
    _patched_paths,
    _setup_script,
    _verifier_script,
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
