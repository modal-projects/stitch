"""Fresh-Sandbox grading: only the policy's patch crosses into a new Sandbox of the
task image, which runs the task's setup and verifier with its network open."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cookbook.miles_disagg.modal_swe import agent

_SETTINGS = {
    "app_name": "test-sandboxes",
    "episode_timeout": 600,
    "exec_timeout": 120,
    "verify_timeout": 1200,
}


class _AgentEnvironment:
    """The agent's Sandbox after the episode: a diff from the baseline, or no baseline."""

    cwd = "/app"

    def __init__(self, patch: bytes | None, *, corrupt: bool = False):
        self.patch = patch
        self.corrupt = corrupt
        self.commands = []

    def exec(self, command, *, cwd, timeout):
        self.commands.append(command)
        if self.patch is None:
            return 1, ""
        return (
            0,
            f"{hashlib.sha256(self.patch).hexdigest()}  {agent._CAPTURED_PATCH_PATH}\n",
        )

    def download_file(self, path):
        assert path == agent._CAPTURED_PATCH_PATH
        return self.patch + (b"x" if self.corrupt else b"")


class _GradeEnvironment:
    instances = []

    def __init__(self, task_dir, **kwargs):
        self.task_dir = task_dir
        self.kwargs = kwargs
        self.cwd = kwargs["cwd"]
        self.uploads = {}
        self.boot_time = 4.0
        self.stopped = False
        _GradeEnvironment.instances.append(self)

    def upload_tree(self, source, destination):
        self.uploads[destination] = {
            path.name: path.read_bytes() for path in Path(source).iterdir()
        }

    def stop(self):
        self.stopped = True


@pytest.fixture
def grading(monkeypatch, tmp_path):
    _GradeEnvironment.instances = []
    calls = {"prepared": [], "verified": []}
    monkeypatch.setattr(agent, "ModalSWEEnvironment", _GradeEnvironment)
    monkeypatch.setattr(
        agent, "_prepare_environment", lambda env, _task: calls["prepared"].append(env)
    )

    def run_verifier(env, task_dir, *, configured_timeout):
        calls["verified"].append(env)
        return {"reward": 1.0, "return_code": 0, "timeout_sec": 1200, "output_tail": ""}

    monkeypatch.setattr(agent, "run_verifier", run_verifier)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[verifier]\ntimeout_sec = 3000.0\n[environment]\ncpus = 1\nmemory_mb = 4096\n"
    )
    return task_dir, calls


def test_fresh_sandbox_receives_only_the_policy_patch(grading):
    task_dir, calls = grading
    patch = b"diff --git a/src.py b/src.py\n"

    result = agent.grade_in_fresh_sandbox(
        _AgentEnvironment(patch), task_dir, settings=_SETTINGS
    )

    (grade_env,) = _GradeEnvironment.instances
    assert grade_env.kwargs["block_network"] is False
    assert (grade_env.kwargs["cpu"], grade_env.kwargs["memory_mib"]) == (1.0, 4096)
    assert calls["prepared"] == [grade_env] and calls["verified"] == [grade_env]
    assert grade_env.uploads == {
        agent._GRADE_PATCH_DIR: {agent._GRADE_PATCH_NAME: patch}
    }
    assert grade_env.stopped
    assert result["reward"] == 1.0
    assert result["grade_metrics"]["policy_patch_bytes"] == len(patch)


def test_empty_patch_is_still_graded(grading):
    task_dir, calls = grading

    agent.grade_in_fresh_sandbox(_AgentEnvironment(b""), task_dir, settings=_SETTINGS)

    (grade_env,) = _GradeEnvironment.instances
    assert grade_env.uploads[agent._GRADE_PATCH_DIR] == {agent._GRADE_PATCH_NAME: b""}
    assert calls["verified"] == [grade_env]


def test_missing_baseline_yields_no_reward_without_a_grading_sandbox(grading):
    task_dir, calls = grading

    result = agent.grade_in_fresh_sandbox(
        _AgentEnvironment(None), task_dir, settings=_SETTINGS
    )

    assert result["reward"] is None
    assert _GradeEnvironment.instances == [] and calls["verified"] == []


def test_a_patch_changed_in_transfer_is_an_infrastructure_failure(grading):
    task_dir, _ = grading

    with pytest.raises(agent.SandboxTransportError, match="changed in transfer"):
        agent.grade_in_fresh_sandbox(
            _AgentEnvironment(b"patch", corrupt=True), task_dir, settings=_SETTINGS
        )


def test_grading_sandbox_stops_even_when_the_verifier_fails(grading, monkeypatch):
    task_dir, _ = grading

    def timed_out(*_args, **_kwargs):
        raise agent.SandboxCommandTimeoutError("command exceeded 1200s")

    monkeypatch.setattr(agent, "run_verifier", timed_out)

    with pytest.raises(agent.SandboxCommandTimeoutError):
        agent.grade_in_fresh_sandbox(
            _AgentEnvironment(b"patch"), task_dir, settings=_SETTINGS
        )
    (grade_env,) = _GradeEnvironment.instances
    assert grade_env.stopped


def test_fresh_sandbox_grading_is_off_unless_an_eval_turns_it_on(monkeypatch):
    monkeypatch.delenv("MODAL_SWE_GRADE_IN_FRESH_SANDBOX", raising=False)
    assert agent._fresh_sandbox_grading() is False
    monkeypatch.setenv("MODAL_SWE_GRADE_IN_FRESH_SANDBOX", "1")
    assert agent._fresh_sandbox_grading() is True


def test_training_recipes_grade_the_patch_fresh_and_require_a_submission():
    from importlib import import_module

    for recipe in (
        "qwen3_6_35b_a3b_hetero_score_centering_mis",
        "qwen3_6_35b_a3b_hetero_score_centering_mis_top_p",
        "qwen3_6_35b_a3b_b200_bf16_score_centering_mis_top_p",
    ):
        env = import_module(f"cookbook.miles_disagg.configs.{recipe}").miles.environment
        assert env["MODAL_SWE_GRADE_IN_FRESH_SANDBOX"] == "1"
        assert env["MODAL_SWE_FRESH_GRADE_APPLY_PATCH"] == "1"
        assert env["MODAL_SWE_REQUIRE_SUBMISSION"] == "1"
        assert env["MODAL_SWE_TIME_EXCEEDED_IS_INFRA"] == "1"
        assert "MODAL_SWE_GRADE_BOTH_WAYS" not in env


def test_a_grade_keeps_the_exact_patch_it_graded(grading):
    task_dir, _ = grading
    patch = b"diff --git a/x b/x\n\xff binary-ish bytes\n"

    result = agent.grade_in_fresh_sandbox(
        _AgentEnvironment(patch), task_dir, settings=_SETTINGS
    )
    outputs = agent._graded_outputs(result, result["policy_patch"])

    import base64

    assert base64.b64decode(outputs["policy_patch_b64"]) == patch
    assert outputs["verifier_output_tail"] == ""


def test_a_verifier_timeout_still_carries_the_graded_patch(grading, monkeypatch):
    task_dir, _ = grading

    def timed_out(*_args, **_kwargs):
        raise agent.SandboxCommandTimeoutError("command exceeded 1200s")

    monkeypatch.setattr(agent, "run_verifier", timed_out)

    with pytest.raises(agent.SandboxCommandTimeoutError) as raised:
        agent.grade_in_fresh_sandbox(
            _AgentEnvironment(b"patch"), task_dir, settings=_SETTINGS
        )
    assert raised.value.policy_patch == b"patch"


def test_grading_in_the_agents_sandbox_keeps_no_extra_outputs():
    assert agent._graded_outputs({"output_tail": "x"}, None) == {}


class _ApplyingGradeEnvironment(_GradeEnvironment):
    """A fresh Sandbox that records commands and fails ``git apply`` when told to."""

    apply_code = 0

    def exec(self, command, *, cwd, timeout):
        self.commands = getattr(self, "commands", []) + [command]
        return _ApplyingGradeEnvironment.apply_code, "error: patch failed"


def test_fresh_grade_applies_the_patch_only_when_the_task_verifier_will_not(
    grading, monkeypatch
):
    task_dir, calls = grading
    monkeypatch.setattr(agent, "ModalSWEEnvironment", _ApplyingGradeEnvironment)
    patch = b"diff --git a/src.py b/src.py\n"

    monkeypatch.delenv("MODAL_SWE_FRESH_GRADE_APPLY_PATCH", raising=False)
    agent.grade_in_fresh_sandbox(_AgentEnvironment(patch), task_dir, settings=_SETTINGS)
    assert not getattr(_GradeEnvironment.instances[-1], "commands", [])

    monkeypatch.setenv("MODAL_SWE_FRESH_GRADE_APPLY_PATCH", "1")
    result = agent.grade_in_fresh_sandbox(
        _AgentEnvironment(patch), task_dir, settings=_SETTINGS
    )
    grade_env = _GradeEnvironment.instances[-1]
    assert grade_env.commands == [
        f"git apply --binary --whitespace=nowarn "
        f"{agent._GRADE_PATCH_DIR}/{agent._GRADE_PATCH_NAME}"
    ]
    assert calls["verified"][-1] is grade_env and result["reward"] == 1.0


def test_a_patch_that_does_not_apply_yields_no_reward_without_running_the_verifier(
    grading, monkeypatch
):
    task_dir, calls = grading
    monkeypatch.setattr(agent, "ModalSWEEnvironment", _ApplyingGradeEnvironment)
    monkeypatch.setattr(_ApplyingGradeEnvironment, "apply_code", 1)
    monkeypatch.setenv("MODAL_SWE_FRESH_GRADE_APPLY_PATCH", "1")

    result = agent.grade_in_fresh_sandbox(
        _AgentEnvironment(b"diff"), task_dir, settings=_SETTINGS
    )

    assert result["reward"] is None and calls["verified"] == []
    assert "did not apply" in result["output_tail"]
    assert _GradeEnvironment.instances[-1].stopped


def test_an_empty_patch_is_graded_without_applying_it(grading, monkeypatch):
    task_dir, calls = grading
    monkeypatch.setattr(agent, "ModalSWEEnvironment", _ApplyingGradeEnvironment)
    monkeypatch.setenv("MODAL_SWE_FRESH_GRADE_APPLY_PATCH", "1")

    agent.grade_in_fresh_sandbox(_AgentEnvironment(b""), task_dir, settings=_SETTINGS)

    assert not getattr(_GradeEnvironment.instances[-1], "commands", [])
    assert len(calls["verified"]) == 1


@pytest.mark.parametrize(
    ("status", "required", "reward", "expected"),
    [
        (
            "Submitted",
            True,
            1.0,
            (1.0, {"unsubmitted": 0, "unsubmitted_diff_passed": 0}),
        ),
        (
            "RepeatedFormatError",
            False,
            1.0,
            (1.0, {"unsubmitted": 1, "unsubmitted_diff_passed": 1}),
        ),
        (
            "RepeatedFormatError",
            True,
            1.0,
            (0.0, {"unsubmitted": 1, "unsubmitted_diff_passed": 1}),
        ),
        (
            "LimitsExceeded",
            True,
            0.0,
            (0.0, {"unsubmitted": 1, "unsubmitted_diff_passed": 0}),
        ),
    ],
)
def test_an_unsubmitted_episode_scores_zero_only_when_submission_is_required(
    monkeypatch, status, required, reward, expected
):
    if required:
        monkeypatch.setenv("MODAL_SWE_REQUIRE_SUBMISSION", "1")
    else:
        monkeypatch.delenv("MODAL_SWE_REQUIRE_SUBMISSION", raising=False)

    assert agent.submission_outcome(status, reward) == expected


def test_the_wall_clock_budget_is_graded_unless_a_recipe_drops_it(monkeypatch):
    monkeypatch.delenv("MODAL_SWE_TIME_EXCEEDED_IS_INFRA", raising=False)
    assert agent._time_exceeded_is_infrastructure() is False
    monkeypatch.setenv("MODAL_SWE_TIME_EXCEEDED_IS_INFRA", "1")
    assert agent._time_exceeded_is_infrastructure() is True


class _OwnSandbox(_AgentEnvironment):
    """The agent's Sandbox, recording whether its patch was captured before its own
    verifier ran there."""

    def __init__(self, patch):
        super().__init__(patch)
        self.events = []

    def exec(self, command, *, cwd, timeout):
        self.events.append("capture")
        return super().exec(command, cwd=cwd, timeout=timeout)


def test_grading_both_ways_trains_the_fresh_grade_and_logs_the_disagreement(
    grading, monkeypatch, caplog
):
    task_dir, calls = grading
    monkeypatch.setenv("MODAL_SWE_GRADE_BOTH_WAYS", "1")
    own = _OwnSandbox(b"diff --git a/src.py b/src.py\n")

    def run_verifier(env, task_dir, *, configured_timeout):
        if env is own:
            own.events.append("own verifier")
            return {
                "reward": 1.0,
                "return_code": 0,
                "timeout_sec": 1,
                "output_tail": "3 passed",
            }
        calls["verified"].append(env)
        return {
            "reward": 0.0,
            "return_code": 1,
            "timeout_sec": 1,
            "output_tail": "1 failed",
        }

    monkeypatch.setattr(agent, "run_verifier", run_verifier)

    with caplog.at_level("WARNING"):
        verifier, metrics = agent.grade_episode(own, task_dir, settings=_SETTINGS)

    assert own.events == ["capture", "own verifier"]
    assert verifier["reward"] == 0.0
    assert metrics == {"grade_own_sandbox_reward": 1.0, "grade_disagreement": 1}
    # Each disagreement names its task and shows both verifiers' output.
    [line] = [
        r.getMessage() for r in caplog.records if "grades disagree" in r.getMessage()
    ]
    assert "own Sandbox 1.0 (rc=0), fresh 0.0 (rc=1)" in line
    assert "own tail: 3 passed | fresh tail: 1 failed" in line


def test_grading_one_way_logs_nothing_extra(grading, monkeypatch):
    task_dir, calls = grading
    monkeypatch.delenv("MODAL_SWE_GRADE_BOTH_WAYS", raising=False)
    monkeypatch.delenv("MODAL_SWE_GRADE_IN_FRESH_SANDBOX", raising=False)
    own = _AgentEnvironment(b"diff")

    verifier, metrics = agent.grade_episode(own, task_dir, settings=_SETTINGS)

    assert metrics == {} and calls["verified"] == [own] and verifier["reward"] == 1.0


class _LocalSandbox:
    """A Sandbox stand-in that runs the grader's real shell commands in a directory."""

    def __init__(self, root: Path):
        self.root = root
        self.cwd = str(root)

    def exec(self, command, *, cwd, timeout):
        done = subprocess.run(
            ["bash", "-c", command],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "HOME": str(self.root.parent),
                "GIT_CONFIG_NOSYSTEM": "1",
            },
        )
        return done.returncode, done.stdout + done.stderr

    def download_file(self, path):
        return Path(path).read_bytes()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(root.parent), "GIT_CONFIG_NOSYSTEM": "1"},
    ).stdout


@pytest.fixture
def task_image(tmp_path, monkeypatch):
    """Two Sandboxes of one task image after its setup: a committed baseline, plus
    files the image leaves untracked (a lockfile, a .venv) and an ignored build dir."""
    monkeypatch.setattr(agent, "_CAPTURED_PATCH_PATH", str(tmp_path / "capture.patch"))
    monkeypatch.setattr(agent, "_TASK_START_INDEX", str(tmp_path / "start.index"))
    image = tmp_path / "image"
    image.mkdir()
    (image / "src.py").write_text("x = 1\n")
    (image / ".gitignore").write_text("build/\n")
    _git(image, "init", "-q")
    _git(image, "add", "src.py", ".gitignore")
    _git(image, "commit", "-q", "-m", "baseline")
    _git(image, "update-ref", agent._TASK_BASELINE_REF, "HEAD")
    (image / "package-lock.json").write_text('{"lockfileVersion": 3}\n')
    (image / ".venv" / "bin").mkdir(parents=True)
    (image / ".venv" / "bin" / "activate").write_text("# venv\n")
    (image / "build").mkdir()
    (image / "build" / "out.o").write_bytes(b"\x00\x01")
    agent_tree, fresh_tree = tmp_path / "agent", tmp_path / "fresh"
    shutil.copytree(image, agent_tree)
    shutil.copytree(image, fresh_tree)
    return agent_tree, fresh_tree


def _apply(tree: Path, patch: bytes) -> subprocess.CompletedProcess:
    (tree.parent / "apply.patch").write_bytes(patch)
    return subprocess.run(
        [
            "git",
            "apply",
            "--binary",
            "--whitespace=nowarn",
            str(tree.parent / "apply.patch"),
        ],
        cwd=tree,
        capture_output=True,
        text=True,
    )


def test_patch_from_the_start_tree_applies_to_a_fresh_sandbox_of_the_image(task_image):
    agent_tree, fresh_tree = task_image
    sandbox = _LocalSandbox(agent_tree)
    status_before = _git(agent_tree, "status", "--porcelain")
    agent.snapshot_task_start(sandbox, timeout=60)
    # The snapshot uses its own index: the agent sees the same git status.
    assert _git(agent_tree, "status", "--porcelain") == status_before

    (agent_tree / "src.py").write_text("x = 2\n")
    (agent_tree / "package-lock.json").write_text('{"lockfileVersion": 4}\n')
    (agent_tree / "new.py").write_text("y = 1\n")
    (agent_tree / ".venv" / "bin" / "activate").unlink()
    patch = agent.capture_policy_patch(sandbox, timeout=60)

    assert (
        b"a/package-lock.json" in patch
        and b"new file mode" not in patch.split(b"diff --git a/new.py")[0]
    )
    assert b"build/out.o" not in patch
    applied = _apply(fresh_tree, patch)
    assert applied.returncode == 0, applied.stderr
    for name in ("src.py", "package-lock.json", "new.py"):
        assert (fresh_tree / name).read_text() == (agent_tree / name).read_text()
    assert not (fresh_tree / ".venv" / "bin" / "activate").exists()


def test_an_untouched_start_tree_captures_an_empty_patch(task_image):
    agent_tree, _ = task_image
    sandbox = _LocalSandbox(agent_tree)
    agent.snapshot_task_start(sandbox, timeout=60)

    assert agent.capture_policy_patch(sandbox, timeout=60) == b""


def test_a_baseline_diff_re_adds_the_images_untracked_files(task_image):
    """Why the start tree exists: without it the patch adds the image's own untracked
    files, and a fresh Sandbox of the image, which already has them, rejects it."""
    agent_tree, fresh_tree = task_image
    patch = agent.capture_policy_patch(_LocalSandbox(agent_tree), timeout=60)

    assert b"package-lock.json" in patch
    applied = _apply(fresh_tree, patch)
    assert applied.returncode != 0 and "already exists" in applied.stderr
