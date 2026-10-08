from __future__ import annotations

import json

import pytest

from cookbook.miles_disagg.mimo_v2_6 import (
    _code_setup_script,
    _code_verifier_script,
    _materialize_code_task,
    _patch_paths,
    normalize_row,
    normalize_rows,
)


def _row(*, image: str = "source:image", instance_id: str = "task-1") -> dict:
    return {
        "data_source": "opensource-code",
        "ability": "swe",
        "agent_name": "mimo_swe_agent",
        "prompt": [{"role": "user", "content": "Fix the issue"}],
        "extra_info": {
            "instance_json": json.dumps(
                {
                    "instance_id": instance_id,
                    "dataset_type": "opensource-code",
                    "docker_image": image,
                    "cwd": "/testbed",
                    "test_patch": "diff --git a/a b/a",
                    "test_command": "true",
                    "verifier_timeout_sec": 1,
                }
            )
        },
    }


def test_normalize_code_row_resolves_image_without_exposing_hidden_state() -> None:
    output = normalize_row(
        "code",
        _row(),
        index=0,
        image_mapping={"source:image": "registry/image:tag"},
    )

    assert output["prompt"] == "Fix the issue"
    assert output["metadata"]["domain"] == "code"
    assert output["metadata"]["image"] == "registry/image:tag"
    assert output["metadata"]["instance"]["test_command"] == "true"
    assert "test_patch" not in output["prompt"]


def test_normalize_music_preserves_task_constraints() -> None:
    output = normalize_row(
        "music",
        {
            "data_source": "music",
            "ability": "music_generation",
            "prompt": [{"role": "user", "content": "Write ABC music"}],
            "extra_info": {"src_id": "music-1", "bpm": 120, "nvoice_want": 3},
        },
        index=0,
        image_mapping={},
    )

    assert output["metadata"]["image"] is None
    assert output["metadata"]["task"]["nvoice_want"] == 3


def test_normalize_rejects_unknown_images_and_duplicate_instances() -> None:
    with pytest.raises(ValueError, match="unmapped Docker image"):
        normalize_row("code", _row(), index=0, image_mapping={})

    with pytest.raises(ValueError, match="duplicate instance identities"):
        normalize_rows(
            "code",
            [_row(), _row()],
            image_mapping={"source:image": "registry/image:tag"},
        )


def test_normalize_rejects_prompt_shape_changes() -> None:
    row = _row()
    row["prompt"].append({"role": "assistant", "content": "leak"})
    with pytest.raises(ValueError, match="exactly one prompt message"):
        normalize_row(
            "code",
            row,
            index=0,
            image_mapping={"source:image": "registry/image:tag"},
        )


def test_patch_paths_include_rename_source_and_target() -> None:
    patch = "diff --git a/old name.py b/new name.py\n"
    assert _patch_paths(patch) == ["new name.py", "old name.py"]


def test_materialize_code_task_keeps_tests_out_of_environment(tmp_path) -> None:
    row = normalize_row(
        "code",
        _row(),
        index=0,
        image_mapping={"source:image": "registry/image:tag"},
    )
    task_dir = _materialize_code_task(tmp_path, row)

    assert (task_dir / "environment" / "Dockerfile").read_text() == (
        "FROM registry/image:tag\n"
    )
    assert not (task_dir / "environment" / "test.patch").exists()
    assert (task_dir / "tests" / "test.patch").is_file()
    assert row["metadata"]["task_dir"] == str(task_dir)
    assert "instance" not in row["metadata"]


@pytest.mark.parametrize(
    "script",
    [_code_setup_script("/testbed"), _code_verifier_script("/testbed", "true")],
)
def test_generated_code_scripts_are_valid(script: str) -> None:
    import subprocess

    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_code_setup_uses_checked_out_head_when_other_refs_exist(tmp_path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "MiMo"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "mimo@example.invalid"],
        cwd=repo,
        check=True,
    )
    (repo / "baseline.txt").write_text("baseline\n")
    subprocess.run(["git", "add", "baseline.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "baseline"], cwd=repo, check=True)
    baseline = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()

    subprocess.run(["git", "checkout", "-q", "-b", "other-ref"], cwd=repo, check=True)
    (repo / "other.txt").write_text("other\n")
    subprocess.run(["git", "add", "other.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "other"], cwd=repo, check=True)
    subprocess.run(
        ["git", "checkout", "-q", "--detach", baseline], cwd=repo, check=True
    )

    subprocess.run(["bash"], input=_code_setup_script(str(repo)), text=True, check=True)

    recorded = subprocess.check_output(
        ["git", "rev-parse", "refs/miles/task-baseline"], cwd=repo, text=True
    ).strip()
    assert recorded == baseline


def _git(repo, *args: str) -> str:
    import subprocess

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


_FIX = {"pkg/src.py": "VALUE = 1\n", "tests/test_src.py": "assert VALUE == 1\n"}


def _testbed_image(tmp_path):
    """A repository shaped like the /testbed images: the fix commit was made and then
    reset away, so only the reflog still reaches it, and the files it touched are
    newer than the rest."""
    import os

    repo = tmp_path / "testbed"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    base = _commit(
        repo,
        {"pkg/src.py": "VALUE = 0\n", "pkg/other.py": "", "README": "r\n"},
        "base",
    )
    fix = _commit(repo, _FIX, "fix the bug")
    _git(repo, "reset", "-q", "--hard", base)
    for path in repo.rglob("*"):
        if ".git" not in path.parts:
            os.utime(path, (1_600_000_000, 1_600_000_000))
    os.utime(repo / "pkg" / "src.py", (1_700_000_000, 1_700_000_000))
    return repo, base, fix


def _workspace_image(tmp_path):
    """A repository shaped like the /workspace/repo images: the branch sits at the
    task's start while upstream refs and a release tag point past it to the fix."""
    repo = tmp_path / "workspace"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    base = _commit(repo, {"pkg/src.py": "VALUE = 0\n", "README": "r\n"}, "base")
    _git(repo, "tag", "v1.0", base)
    fix = _commit(repo, _FIX, "fix the bug")
    _git(repo, "tag", "v1.1", fix)
    _git(repo, "remote", "add", "origin", "https://example.invalid/upstream.git")
    _git(repo, "update-ref", "refs/remotes/origin/main", fix)
    _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
    _git(repo, "update-ref", "refs/heads/upstream-main", fix)
    _git(repo, "reset", "-q", "--hard", base)
    return repo, base, fix


def _run_code_setup(tmp_path, repo) -> None:
    import os
    import subprocess

    # The script adds a global safe.directory; keep it out of the real home.
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {**os.environ, "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(
        ["bash"], input=_code_setup_script(str(repo)), text=True, check=True, env=env
    )


def _assert_only_the_start_is_reachable(repo, base: str, fix: str) -> None:
    import subprocess

    gone = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", fix])
    assert gone.returncode != 0, "the fix commit is still in the object store"
    fixed_blob = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "--stdin"],
        input=_FIX["tests/test_src.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", fixed_blob]
        ).returncode
        != 0
    ), "the hidden test's content is still in the object store"
    assert _git(repo, "rev-list", "--all", "--reflog") == _git(repo, "rev-list", "HEAD")
    assert _git(repo, "fsck", "--unreachable", "--no-reflogs", "--no-progress") == ""
    assert _git(repo, "reflog", "--all") == ""
    assert _git(repo, "stash", "list") == ""
    assert _git(repo, "rev-parse", "refs/miles/task-baseline") == base


@pytest.mark.parametrize("image", [_testbed_image, _workspace_image])
def test_code_setup_leaves_no_way_to_the_fix(tmp_path, image) -> None:
    repo, base, fix = image(tmp_path)

    _run_code_setup(tmp_path, repo)

    _assert_only_the_start_is_reachable(repo, base, fix)
    assert (repo / "pkg" / "src.py").read_text() == "VALUE = 0\n"
    assert not (repo / "tests").exists()


def test_code_setup_keeps_tags_on_the_start_and_drops_the_rest(tmp_path) -> None:
    repo, base, _ = _workspace_image(tmp_path)

    _run_code_setup(tmp_path, repo)

    refs = _git(repo, "for-each-ref", "--format=%(refname)").split()
    assert refs == ["refs/heads/main", "refs/miles/task-baseline", "refs/tags/v1.0"]
    assert _git(repo, "remote") == ""


def test_code_setup_gives_every_tracked_file_and_directory_one_time(tmp_path) -> None:
    import os

    repo, _, _ = _testbed_image(tmp_path)
    # Build output compiled from the fixed code, newer than every source.
    (repo / "build").mkdir()
    (repo / "build" / "src.o").write_text("compiled with the fix\n")
    os.utime(repo / "build" / "src.o", (1_750_000_000, 1_750_000_000))

    _run_code_setup(tmp_path, repo)

    tracked = [repo / "pkg" / "src.py", repo / "pkg" / "other.py", repo / "README"]
    times = {path.stat().st_mtime for path in [*tracked, repo / "pkg", repo]}
    assert len(times) == 1
    # No source may look older than build output, or a build tool would reuse the
    # fixed code's output and the unfixed tree would pass.
    assert times.pop() > 1_750_000_000
    assert (repo / "build" / "src.o").stat().st_mtime == 1_750_000_000
    assert _git(repo, "status", "--porcelain") == "?? build/"


def test_code_setup_does_not_recreate_a_deleted_tracked_file(tmp_path) -> None:
    repo, _, _ = _testbed_image(tmp_path)
    (repo / "pkg" / "other.py").unlink()

    _run_code_setup(tmp_path, repo)

    assert not (repo / "pkg" / "other.py").exists()


def test_code_setup_baselines_a_directory_without_git(tmp_path) -> None:
    repo = tmp_path / "plain"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "src.py").write_text("VALUE = 0\n")

    _run_code_setup(tmp_path, repo)

    assert _git(repo, "rev-list", "--count", "--all") == "1"
    assert _git(repo, "rev-parse", "refs/miles/task-baseline") == _git(
        repo, "rev-parse", "HEAD"
    )


@pytest.mark.parametrize(("policy_fixes", "reward"), [(True, "1"), (False, "0")])
def test_the_verifier_grades_a_pruned_repository(
    tmp_path, policy_fixes: bool, reward: str
) -> None:
    import subprocess

    repo, _, fix = _testbed_image(tmp_path)
    test_patch = _git(repo, "diff", "HEAD", fix, "--", "tests/test_src.py") + "\n"
    _run_code_setup(tmp_path, repo)
    if policy_fixes:
        (repo / "pkg" / "src.py").write_text("VALUE = 1\n")

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.patch").write_text(test_patch)
    (tests / "patch_paths.txt").write_text(
        "".join(f"{path}\n" for path in _patch_paths(test_patch))
    )
    logs = tmp_path / "logs"
    command = "python3 -c \"exec(open('pkg/src.py').read()); exec(open('tests/test_src.py').read())\""
    script = (
        _code_verifier_script(str(repo), command)
        .replace("/tests/", f"{tests}/")
        .replace("/logs/verifier", str(logs))
    )
    subprocess.run(["bash", "-c", script], capture_output=True)

    assert (logs / "reward.txt").read_text().strip() == reward
    assert not (repo / "tests" / "test_src.py").exists()


def test_code_setup_drops_broken_refs_that_would_stop_gc(tmp_path) -> None:
    repo, base, fix = _workspace_image(tmp_path)
    missing = "0123456789abcdef0123456789abcdef01234567"
    # origin/HEAD naming a ref the image lacks, a loose ref and a packed ref to an
    # object it lacks: git skips all three, except gc, which fails on them.
    _git(repo, "update-ref", "-d", "refs/remotes/origin/main")
    (repo / ".git" / "refs" / "heads" / "ghost").write_text(missing + "\n")
    with (repo / ".git" / "packed-refs").open("a") as packed:
        packed.write(f"{missing} refs/tags/ghost-tag\n")

    _run_code_setup(tmp_path, repo)

    _assert_only_the_start_is_reachable(repo, base, fix)
    assert not (repo / ".git" / "refs" / "remotes" / "origin" / "HEAD").exists()
    assert not (repo / ".git" / "refs" / "heads" / "ghost").exists()
    assert "ghost-tag" not in (repo / ".git" / "packed-refs").read_text()


@pytest.mark.parametrize("mark", ["promisor", "keep"])
def test_code_setup_prunes_history_in_a_marked_pack(tmp_path, mark: str) -> None:
    repo, base, fix = _testbed_image(tmp_path)
    # Pack everything, the reflog's fix included, into one pack that gc would keep.
    _git(repo, "repack", "-a", "-d", "-q")
    for pack in (repo / ".git" / "objects" / "pack").glob("*.pack"):
        pack.with_suffix(f".{mark}").write_text("")

    _run_code_setup(tmp_path, repo)

    _assert_only_the_start_is_reachable(repo, base, fix)


def test_excluded_code_tasks_are_checked_against_this_revision(tmp_path) -> None:
    from cookbook.miles_disagg.mimo_v2_6 import DATASET_REVISION, excluded_code_tasks

    path = tmp_path / "excluded.json"
    data = {
        "dataset_revision": DATASET_REVISION,
        "reasons": {"passes_unfixed": "the empty patch passes"},
        "tasks": {"task-1": "passes_unfixed"},
    }
    path.write_text(json.dumps(data))
    assert excluded_code_tasks(path) == {"task-1": "passes_unfixed"}

    path.write_text(json.dumps({**data, "dataset_revision": "another"}))
    with pytest.raises(ValueError, match="another dataset revision"):
        excluded_code_tasks(path)

    path.write_text(json.dumps({**data, "tasks": {"task-1": "undefined"}}))
    with pytest.raises(ValueError, match="undefined reasons"):
        excluded_code_tasks(path)


def test_the_checked_in_exclusions_load() -> None:
    from cookbook.miles_disagg.mimo_v2_6 import excluded_code_tasks

    assert isinstance(excluded_code_tasks(), dict)


def test_drop_excluded_keeps_the_rest_and_rejects_unknown_tasks() -> None:
    from cookbook.miles_disagg.mimo_v2_6 import drop_excluded

    rows = [
        normalize_row(
            "code", _row(instance_id=name), index=i, image_mapping={"source:image": "x"}
        )
        for i, name in enumerate(["task-1", "task-2"])
    ]

    kept = drop_excluded(rows, {"task-1": "passes_unfixed"})

    assert [row["metadata"]["instance_id"] for row in kept] == ["task-2"]
    with pytest.raises(ValueError, match="not in the data"):
        drop_excluded(rows, {"task-9": "passes_unfixed"})
