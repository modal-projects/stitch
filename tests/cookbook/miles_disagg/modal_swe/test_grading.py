"""Fresh-Sandbox grading: only the policy's patch crosses into a new Sandbox of the
task image, which runs the task's setup and verifier with its network open."""

import hashlib
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


def test_training_recipes_grade_in_the_agents_own_sandbox():
    from importlib import import_module

    for recipe in ("qwen3_6_35b_a3b_hetero_grpo", "qwen3_6_35b_a3b_b200_bf16_icepop"):
        cfg = import_module(f"cookbook.miles_disagg.configs.{recipe}").miles
        assert "MODAL_SWE_GRADE_IN_FRESH_SANDBOX" not in cfg.environment


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
