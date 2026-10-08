"""The opt-in trajectory dump the sampler benchmark replays (MODAL_SWE_TRAJECTORY_DUMP_DIR)."""

import json
from importlib import import_module
from types import SimpleNamespace

import pytest

try:
    import_module("jinja2")
    from cookbook.miles_disagg.modal_swe import agent as agent_module
except ModuleNotFoundError as error:
    if error.name not in {"jinja2", "miles"}:
        raise
    pytest.skip(
        "Modal SWE adapter tests require the Miles trainer environment",
        allow_module_level=True,
    )

DUMP_ENV = agent_module.TRAJECTORY_DUMP_ENV


def _response(prompt, completion, finish="tool_calls", cached=None):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"usage": usage, "choices": [{"finish_reason": finish}]}


def _messages():
    """What mini-swe-agent's DefaultAgent keeps for a short episode: two model calls,
    the second of which failed to parse and was replaced by a format-error message."""
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the issue."},
        {
            "role": "assistant",
            "content": "Let me look.",
            "reasoning_content": "I should list files.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "index": 0,
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ],
            "function_call": None,
            "extra": {
                "actions": [{"command": "ls"}],
                "response": _response(120, 30, cached=100),
                "cost": 0.0,
                "timestamp": 1700000000.5,
            },
        },
        {"role": "tool", "content": "a.py", "tool_call_id": "call_1", "extra": {}},
        {
            "role": "user",
            "content": "No tool call found.",
            "extra": {"response": _response(160, 12, finish="stop")},
        },
        {
            "role": "exit",
            "content": "LimitsExceeded",
            "extra": {"exit_status": "LimitsExceeded", "submission": ""},
        },
    ]


def test_record_keeps_messages_in_order_with_each_model_calls_tokens():
    record = agent_module.trajectory_record(
        _messages(),
        metadata={"instance_id": "repo__task-1", "eval_sample_index": 3},
        result={"exit_status": "LimitsExceeded", "reward": 0.0},
        durations=[4.0, 1.5],
        request_kwargs={"temperature": 1.0, "max_tokens": 32768},
        tools=[{"type": "function", "function": {"name": "bash"}}],
    )

    assert record["format"] == agent_module.TRAJECTORY_FORMAT
    assert (record["instance_id"], record["sample_index"]) == ("repo__task-1", 3)
    assert (record["exit_status"], record["reward"]) == ("LimitsExceeded", 0.0)
    assert record["model_calls"] == 2
    assert record["model_request_durations_seconds"] == [4.0, 1.5]
    assert record["request_kwargs"]["max_tokens"] == 32768
    assert record["tools"][0]["function"]["name"] == "bash"
    roles = [message["role"] for message in record["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "user", "exit"]

    system, user, assistant, tool, format_error, exit_message = record["messages"]
    assert system == {"role": "system", "content": "You are an agent."}
    assert "model_call" not in user
    assert assistant["reasoning_content"] == "I should list files."
    assert assistant["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
        }
    ]
    assert assistant["model_call"] is True
    assert (assistant["prompt_tokens"], assistant["completion_tokens"]) == (120, 30)
    assert assistant["cached_tokens"] == 100
    assert assistant["finish_reason"] == "tool_calls"
    assert assistant["timestamp"] == 1700000000.5
    assert assistant["request_seconds"] == 4.0
    # Internal bookkeeping never reaches the dump.
    assert "extra" not in assistant and "function_call" not in assistant
    assert tool == {"role": "tool", "content": "a.py", "tool_call_id": "call_1"}
    # The rejected response's tokens stay on the message that replaced it.
    assert format_error["model_call"] is True
    assert format_error["completion_tokens"] == 12
    assert format_error["request_seconds"] == 1.5
    assert exit_message["exit_status"] == "LimitsExceeded"
    json.dumps(record)


def test_durations_pair_with_model_calls_only_when_their_counts_match():
    # A resent request adds a duration without a model call.
    record = agent_module.trajectory_record(
        _messages(), metadata={}, result={}, durations=[600.0, 4.0, 1.5]
    )

    assert record["model_request_durations_seconds"] == [600.0, 4.0, 1.5]
    assert all("request_seconds" not in m for m in record["messages"])
    assert record["sample_index"] is None


def test_dump_is_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv(DUMP_ENV, raising=False)
    seen = {}

    def fake_episode(**kwargs):
        seen.update(kwargs)
        return {"exit_status": "Submitted", "reward": 1.0}

    monkeypatch.setattr(agent_module, "_run_episode", fake_episode)

    result = agent_module._run_episode_sync(
        base_url="http://x",
        prompt="p",
        request_kwargs={},
        metadata={"instance_id": "a"},
        queued_at=0.0,
    )

    assert result == {"exit_status": "Submitted", "reward": 1.0}
    assert seen["trajectory"] is None
    assert agent_module._TrajectoryCapture.from_environment({}, {}) is None
    assert list(tmp_path.iterdir()) == []


def _episode_with_agent(messages, result):
    def fake_episode(*, trajectory, **_kwargs):
        durations = [2.0, 1.0]
        trajectory.attach(SimpleNamespace(messages=messages), object(), durations)
        return result

    return fake_episode


def test_episode_end_writes_one_trajectory_file(monkeypatch, tmp_path):
    dump_dir = tmp_path / "traces"
    monkeypatch.setenv(DUMP_ENV, str(dump_dir))
    result = {"exit_status": "Submitted", "reward": 1.0, "agent_metrics": {}}
    monkeypatch.setattr(
        agent_module, "_run_episode", _episode_with_agent(_messages(), result)
    )

    returned = agent_module._run_episode_sync(
        base_url="http://x",
        prompt="p",
        request_kwargs={"temperature": 1.0},
        metadata={"instance_id": "org/repo#1", "eval_sample_index": 0},
        queued_at=0.0,
    )

    assert returned is result
    (path,) = dump_dir.iterdir()
    assert path.name.startswith("org_repo_1.s0.") and path.suffix == ".json"
    record = json.loads(path.read_text())
    assert record["instance_id"] == "org/repo#1"
    assert record["sample_index"] == 0
    assert (record["exit_status"], record["reward"]) == ("Submitted", 1.0)
    assert record["request_kwargs"] == {"temperature": 1.0}
    assert record["model_request_durations_seconds"] == [2.0, 1.0]
    assert len(record["messages"]) == len(_messages())
    # The fake model is not mini-swe-agent's LiteLLM model, so no tools are claimed.
    assert record["tools"] is None


def test_no_trajectory_is_written_when_the_agent_never_started(monkeypatch, tmp_path):
    monkeypatch.setenv(DUMP_ENV, str(tmp_path))
    monkeypatch.setattr(
        agent_module,
        "_run_episode",
        lambda **_kwargs: {"exit_status": "sandbox_infra_error"},
    )

    agent_module._run_episode_sync(
        base_url="http://x",
        prompt="p",
        request_kwargs={},
        metadata={"instance_id": "a"},
        queued_at=0.0,
    )

    assert list(tmp_path.iterdir()) == []


def test_a_failed_dump_never_fails_the_episode(monkeypatch, tmp_path, caplog):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    monkeypatch.setenv(DUMP_ENV, str(blocked))
    result = {"exit_status": "Submitted", "reward": 1.0}
    monkeypatch.setattr(
        agent_module, "_run_episode", _episode_with_agent(_messages(), result)
    )

    returned = agent_module._run_episode_sync(
        base_url="http://x",
        prompt="p",
        request_kwargs={},
        metadata={"instance_id": "a"},
        queued_at=0.0,
    )

    assert returned is result
    assert "Failed to write the trajectory" in caplog.text
