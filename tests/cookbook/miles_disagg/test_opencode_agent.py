import asyncio
import json
import logging
import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest

SIMPLE_STEP = [
    {"type": "step_start"},
    {"type": "reasoning", "part": {"text": "\nI should run the first command.\n"}},
    {"type": "tool_use", "part": {"tool": "bash", "state": {"status": "completed"}}},
    {"type": "step_finish", "part": {"reason": "tool-calls"}},
]
# OpenCode 1.18.32 on a reply cut off at the reply limit that held a tool call (local probe): it ran the tool, then
# its next request extended the cut-off reply and failed.
CUT_OFF_WITH_TOOL = [
    *SIMPLE_STEP,
    {"type": "step_start"},
    {"type": "reasoning", "part": {"text": "\nA tool call parsed from a cut-off reply.\n"}},
    {"type": "tool_use", "part": {"tool": "bash", "state": {"status": "completed"}}},
    {"type": "step_finish", "part": {"reason": "length"}},
    {"type": "error", "error": {"name": "APIError", "data": {"statusCode": 400}}},
]


def _trace(events) -> str:
    return "".join(json.dumps(e) + "\n" for e in events)


class _FakeOpenCode:
    """The part of Harbor's OpenCode agent that the harness uses."""

    _DEFAULT_CONFIG: dict = {}
    model_name = "openai-compatible/model"
    logger = logging.getLogger("opencode-test")

    def _build_register_skills_command(self):
        return None

    def _build_register_config_command(self):
        return "write-config"

    def build_cli_flags(self):
        return ""


@pytest.fixture
def oc(monkeypatch):
    opencode = ModuleType("harbor.agents.installed.opencode")
    opencode.OpenCode = _FakeOpenCode
    for name in ("harbor", "harbor.agents", "harbor.agents.installed"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "harbor.agents.installed.opencode", opencode)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.opencode_agent", raising=False)
    return import_module("cookbook.miles_disagg.opencode_agent")


def test_the_trial_ends_on_a_reply_cut_off_at_the_reply_limit(oc):
    assert oc.at_reply_limit(_trace(CUT_OFF_WITH_TOOL))
    assert oc.at_reply_limit(_trace(CUT_OFF_WITH_TOOL[:8]))  # ended the pass itself (no tool call)
    assert not oc.at_reply_limit(_trace(SIMPLE_STEP))
    assert not oc.at_reply_limit(_trace([*SIMPLE_STEP, {"type": "step_start"}]))  # stopped mid-reply
    assert not oc.at_reply_limit("")


def test_cut_off_finds_a_reply_stopped_by_a_typed_special_token(oc):
    def reply(text):
        return _trace([{"type": "step_start"}, {"type": "text", "part": {"text": text}},
                       {"type": "step_finish", "part": {"reason": "stop"}}])

    assert oc.cut_off(reply("Set eos to `"))
    assert oc.cut_off(reply("```python\nprint(1)"))
    assert not oc.cut_off(reply("The model is saved."))
    assert not oc.cut_off(_trace(SIMPLE_STEP))


def test_after_pass_matches_the_evaluation_harness(oc):
    assert oc.after_pass(True, 3600, False, 0) == (0, False, 0.0)
    assert oc.after_pass(True, 10, False, 0) == (0, False, 30.0)
    assert oc.after_pass(False, 10, True, 1) == (0, False, 30.0)
    assert oc.after_pass(False, 120, False, 3) == (0, False, 0.0)
    assert oc.after_pass(False, 5, False, 0) == (1, False, 30.0)
    assert oc.after_pass(False, 5, False, 1) == (2, True, 60.0)
    assert oc.after_pass(False, 5, False, 5) == (6, True, 300.0)


class _Sandbox:
    """A task sandbox on a fake clock that only an OpenCode pass moves, 1 s per step: `time_left` counts down from
    `limit` seconds, and passes follow a script of (seconds, trace events, clean exit)."""

    def __init__(self, oc, clock, limit, passes, real_sleep):
        self.oc, self.clock, self.limit, self.passes = oc, clock, limit, list(passes)
        self.trace, self.calls, self.stop, self.sessions = "", [], False, None
        self.real_sleep = real_sleep

    async def exec(self, command, **kwargs):
        if command == "time_left":
            left = max(0, int(self.limit - self.clock["t"]))
            return SimpleNamespace(stdout=f"{left // 60} min {left % 60:02d} s left\n")
        if command == self.oc.TRACE_TAIL:
            return SimpleNamespace(stdout=self.trace)
        if command.startswith("pkill -INT"):
            self.stop = True
        if command.startswith("echo") and "opencode_sessions" in command:
            self.sessions = int(command.split()[1])
        return SimpleNamespace(stdout="")

    async def opencode(self, environment, flags, message, append):
        self.calls.append((flags, message, append))
        seconds, events, ok = self.passes.pop(0)
        self.stop = False
        for _ in range(seconds):
            if self.stop:
                return False
            self.clock["t"] += 1
            await self.real_sleep(0)
        self.trace += _trace(events)
        return ok


def _run(oc, monkeypatch, limit, passes):
    clock = {"t": 0.0}
    real_sleep = asyncio.sleep

    async def sleep(seconds):  # the harness's waits and polls take no time on this clock
        await real_sleep(0)

    monkeypatch.setattr(oc.asyncio, "sleep", sleep)
    monkeypatch.setattr(oc.time, "monotonic", lambda: clock["t"])
    sandbox = _Sandbox(oc, clock, limit, passes, real_sleep)
    agent = object.__new__(oc.RolloutOpenCode)
    agent.opencode_written = []

    async def exec_as_agent(environment, command, env=None):
        agent.opencode_written.append((command, env))

    agent.exec_as_agent = exec_as_agent
    agent._opencode = sandbox.opencode
    asyncio.run(agent.run("TASK", sandbox, None))
    return sandbox, agent


def test_a_reminder_stops_the_pass_and_a_cut_off_reply_ends_the_trial(oc, monkeypatch):
    sandbox, agent = _run(oc, monkeypatch, 1200, [
        (700, SIMPLE_STEP, True),  # still working at the first mark (50% of 20 minutes left)
        (30, CUT_OFF_WITH_TOOL, False),
        (30, SIMPLE_STEP, True),  # never reached
    ])

    first, second = sandbox.calls
    assert first == ("", f"TASK\n\n{oc.RULE_NOTE}", False)
    assert second[0] == "--continue" and second[2]
    assert second[1] == oc.REMINDER.format(hours=0, minutes=9)  # stopped at the mark (600 s left), 1 s later
    assert agent.opencode_written == [("write-config", oc.ENV)]
    assert sandbox.sessions == 2


def test_passes_continue_until_5_percent_of_the_time_is_left(oc, monkeypatch):
    monkeypatch.setattr(oc, "REMINDER_FRACTIONS", ())
    special_token_stop = [{"type": "step_start"}, {"type": "text", "part": {"text": "the end token is `"}},
                          {"type": "step_finish", "part": {"reason": "stop"}}]
    sandbox, _ = _run(oc, monkeypatch, 1200, [
        (300, special_token_stop, True),
        (500, SIMPLE_STEP, True),
        (350, SIMPLE_STEP, True),  # ends with 50 s left, under 5% of 1,200 s
    ])

    prompts = [message for _, message, _ in sandbox.calls]
    assert prompts[1] == oc.CUT_HINT.format(hours=0, minutes=15)
    assert prompts[2] == oc.REPROMPT.format(hours=0, minutes=6)
    assert len(prompts) == 3


def test_a_session_that_fails_to_resume_is_replaced(oc, monkeypatch):
    monkeypatch.setattr(oc, "REMINDER_FRACTIONS", ())
    sandbox, _ = _run(oc, monkeypatch, 1200, [
        (100, SIMPLE_STEP, True),
        (2, [], False),
        (2, [], False),  # second quick failure in a row: a new session next
        (1200, SIMPLE_STEP, True),
    ])

    flags = [f for f, _, _ in sandbox.calls]
    assert flags == ["", "--continue", "--continue", ""]
    assert sandbox.calls[3][1].startswith("Note from your harness: your previous session ended with an error")
    assert sandbox.calls[3][1].endswith(f"TASK\n\n{oc.RULE_NOTE}")


def test_time_left_is_read_from_the_task_clock(oc):
    agent = object.__new__(oc.RolloutOpenCode)

    async def left(stdout):
        class Sandbox:
            async def exec(self, command):
                return SimpleNamespace(stdout=stdout)

        return await agent._seconds_left(Sandbox())

    assert asyncio.run(left("12 min 05 s left\n")) == 725
    assert asyncio.run(left("deadline unknown\n")) is None
