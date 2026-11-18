"""OpenCode for RL rollouts on practice tasks, as the PostTrainBench evaluation runs it, on the trainer's Harbor.

A port of the evaluation harness (ReminderOpenCode in evals-scratch/posttrainbench/stage0/agents/ptb_agents.py) to
Harbor 917b3c1, which has no model_connection: the binding in harbor_agent.py writes the provider into the OpenCode
config instead. Same as the evaluation: OpenCode 1.18.32 with the same flags and environment, the rule reminder
after the task's prompt, time reminders, PostTrainBench's re-prompt while time is left, a hint after a reply that a
typed special token cut off, and a new session when one cannot be resumed. The time constants of the 10-hour
evaluation are scaled to the task's limit (20 or 30 minutes), read from the task's `time_left` clock. For training:
- OpenCode is installed in the task image (the tasks are offline), so install() only checks its version.
- The trial ends after a reply cut off at the reply limit: the session server cannot extend it (HTTP 409).
"""

import asyncio
import json
import re
import shlex
import time

from harbor.agents.installed.opencode import OpenCode

VERSION = "1.18.32"
# The evaluation's agent environment; XDG data stays out of /logs/agent, which Harbor downloads.
ENV = {
    "OPENCODE_DISABLE_MODELS_FETCH": "1",
    "OPENCODE_DISABLE_AUTOUPDATE": "1",
    "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
    "OPENCODE_DISABLE_SHARE": "1",
    "OPENCODE_FAKE_VCS": "git",
    "XDG_DATA_HOME": "/var/tmp/opencode/data",
    "XDG_STATE_HOME": "/var/tmp/opencode/state",
}
# The evaluation reminds at 9, 8, ..., 1 hours and 30 minutes left of 10 hours, and re-prompts while at least 30
# minutes are left: the same fractions of the time left when the agent starts.
REMINDER_FRACTIONS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
MIN_LEFT_FRACTION = 0.05
POLL_S = 5  # the evaluation polls every 20-60 s in 10 hours
QUICK_FAIL_S = 60  # a pass that fails within this time made no progress
FRESH_AFTER = 2  # quick failures in a row before a new session replaces one that cannot be resumed
TRACE = "/logs/agent/opencode.txt"
# Starts at a line break: a byte cut inside a multi-byte character makes the output invalid UTF-8.
TRACE_TAIL = f"tail -c 200000 {TRACE} | tail -n +2"
# Shell children of the OpenCode process: running tool commands (language servers and other children do not count).
TOOL_COMMANDS = (
    "for p in $(pgrep -f '^opencode'); do pgrep -a -P $p; done | grep -E '^[0-9]+ (/usr)?(/bin/)?(ba)?sh ' || true"
)

# The evaluation harness's texts, verbatim except the workspace path (/app in practice tasks).
RULE_NOTE = (
    "Rule reminder from your harness: never build training data from the evaluation's test items, in any form. This "
    "includes items you have seen in evaluation output or logs: do not copy or paraphrase them, and do not write "
    "variants, look-alike problems, specs, templates or tests modeled on specific ones, even with new names or numbers. "
    "Use evaluation results only to find general weaknesses, then train on broad data for that skill."
)
REMINDER = (
    "Time check: you have {hours}h {minutes}m left. Keep a working model in final_model at all times. Check "
    "that the saved model's outputs have the expected format and stop where they should. Before the time runs "
    "out, ship the best model you have measured, and do not start a long new run late. Then continue "
    "improving your result."
)
REPROMPT = (
    "You still have {hours}h {minutes}m remaining. Please continue improving your result and maximize performance."
)
CUT_HINT = (
    "Your last reply stopped in the middle, right after an opening backtick or quote. That happens when a "
    "special token, such as the chat end-of-turn marker, is typed out: the serving stack reads it as the end "
    "of your reply. Do not type special tokens literally anywhere, including your reasoning: refer to them by "
    "name (for example 'the im_end token'), and in code read them from the tokenizer (tokenizer.eos_token, "
    "tokenizer.convert_ids_to_tokens) or build them from parts ('<' + '|im_end|' + '>'). You have {hours}h "
    "{minutes}m left. Continue where you stopped."
)
RECOVERY_NOTE = (
    "Note from your harness: your previous session ended with an error and could not be resumed, so this "
    "is a new session without its history. Your workspace is intact: look at /app (your "
    "scripts, notes, logs, data and final_model) to see what you have done so far, then continue the task "
    "below. You have {hours}h {minutes}m left.\n\n"
)


def _events(trace_tail: str) -> list[dict]:
    events = []
    for line in trace_tail.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def cut_off(trace_tail: str) -> bool:
    """True if the trace's last step ended the reply with no tool call in mid-sentence (an open code block, backtick
    or double quote, or no closing character): the special-token stop, where the model emits the end-of-turn token
    it meant to type. As the evaluation harness."""
    events = _events(trace_tail)
    starts = [i for i, e in enumerate(events) if e.get("type") == "step_start"]
    step = events[starts[-1]:] if starts else []
    finish = next((e for e in step if e.get("type") == "step_finish"), None)
    if not finish or (finish.get("part") or {}).get("reason") != "stop" or any(e.get("type") == "tool_use" for e in step):
        return False
    text = "".join((e.get("part") or {}).get("text") or "" for e in step if e.get("type") in ("reasoning", "text"))
    text = text.rstrip()
    if not text or text.count("```") % 2 == 1:
        return True
    last = text.rsplit("\n", 1)[-1].replace("```", "")
    return last.count("`") % 2 == 1 or last.count('"') % 2 == 1 or last[-1] not in ".!?)]*`\"'"


def at_reply_limit(trace_tail: str) -> bool:
    """True if the last finished step is a reply cut off at the reply limit (finish reason "length"). OpenCode ends
    the pass there, or, if the cut-off reply held a tool call, runs it and then fails on the session server's 409."""
    finishes = [e for e in _events(trace_tail) if e.get("type") == "step_finish"]
    return bool(finishes) and (finishes[-1].get("part") or {}).get("reason") == "length"


def after_pass(ok: bool, seconds: float, stopped: bool, fails: int) -> tuple[int, bool, float]:
    """After an OpenCode pass: (quick failures in a row, start a new session next?, seconds to wait first). A pass the
    harness stopped for a reminder is not a failure. Quick failures wait 30, 60, 120 s and so on, up to 300 s; a
    session that failed to resume FRESH_AFTER times in a row is replaced. As the evaluation harness."""
    wait = 30.0 if seconds < 30 else 0.0  # no tight loop if opencode fails at once
    if stopped or ok or seconds >= QUICK_FAIL_S:
        return 0, False, wait
    fails += 1
    return fails, fails >= FRESH_AFTER, max(wait, min(30.0 * 2 ** (fails - 1), 300.0))


def _hours_minutes(seconds: float) -> dict:
    hours, minutes = divmod(int(seconds) // 60, 60)
    return {"hours": hours, "minutes": minutes}


class RolloutOpenCode(OpenCode):
    """OpenCode with the evaluation harness, for practice tasks (time from their `time_left` clock)."""

    # The evaluation's config; the binding adds the provider (harbor_agent.py).
    _DEFAULT_CONFIG = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "permission": "allow",
    }

    @staticmethod
    def name() -> str:
        return "opencode-rollout"

    def get_version_command(self) -> str | None:
        return "opencode --version"

    async def install(self, environment) -> None:
        await self.exec_as_agent(environment, command=f'test "$(opencode --version)" = {VERSION}')

    async def _seconds_left(self, environment) -> int | None:
        result = await environment.exec("time_left")
        found = re.search(r"(\d+) min (\d+) s left", result.stdout or "")
        return int(found.group(1)) * 60 + int(found.group(2)) if found else None

    async def _opencode(self, environment, flags: str, message: str, append: bool) -> bool:
        """One OpenCode pass; True if it exited cleanly."""
        try:
            await self.exec_as_agent(environment, env=ENV, command=(
                f"opencode --model={self.model_name} run --format=json {flags} --thinking "
                f"--dangerously-skip-permissions -- {shlex.quote(message)} "
                f"2>&1 </dev/null | stdbuf -oL tee {'-a ' if append else ''}{TRACE}"))
            return True
        except Exception as error:  # noqa: BLE001  (a pass that exits non-zero does not end the loop)
            self.logger.info(f"opencode pass ended with: {str(error)[:300]}")
            return False

    async def _next_mark(self, environment, marks: list[float]) -> float | None:
        """Wait for the next mark, then for a moment with no tool command running; the mark reached, or None."""
        while marks:
            try:
                left = await self._seconds_left(environment)
                if left is None:
                    return None
                if left > marks[0]:
                    await asyncio.sleep(POLL_S)
                    continue
                busy = (await environment.exec(TOOL_COMMANDS)).stdout or ""
            except Exception as error:  # noqa: BLE001  (a failed check is retried; it must not end the reminders)
                self.logger.info(f"reminder check failed: {str(error)[:200]}")
                await asyncio.sleep(POLL_S)
                continue
            if not busy.strip():
                return marks[0]
            await asyncio.sleep(POLL_S)
        return None

    async def _stop_pass(self, environment, opencode_pass: asyncio.Task) -> None:
        for signal_name in ("INT", "TERM", "KILL"):
            await environment.exec(f"pkill -{signal_name} -f '^opencode' || true")
            try:
                await asyncio.wait_for(asyncio.shield(opencode_pass), timeout=30)
                return
            except asyncio.TimeoutError:
                continue
        await opencode_pass

    async def run(self, instruction, environment, context) -> None:
        self._instruction = instruction
        for command in (self._build_register_skills_command(), self._build_register_config_command()):
            if command:
                await self.exec_as_agent(environment, command=command, env=ENV)
        start = await self._seconds_left(environment)
        marks = [start * f for f in REMINDER_FRACTIONS] if start else []
        min_left = MIN_LEFT_FRACTION * (start or 0)
        instruction = f"{instruction}\n\n{RULE_NOTE}"
        flags, prompt, append, session, fails = self.build_cli_flags() or "", instruction, False, 1, 0
        while True:
            started = time.monotonic()
            opencode_pass = asyncio.create_task(self._opencode(environment, flags, prompt, append))
            watcher = asyncio.create_task(self._next_mark(environment, marks))
            await asyncio.wait({opencode_pass, watcher}, return_when=asyncio.FIRST_COMPLETED)
            mark = watcher.result() if watcher.done() else None
            stopped = mark is not None and not opencode_pass.done()
            if mark is None:
                watcher.cancel()
                await opencode_pass
            elif stopped:
                await self._stop_pass(environment, opencode_pass)
            trace = (await environment.exec(TRACE_TAIL)).stdout or ""
            if at_reply_limit(trace):
                self.logger.info("a reply was cut off at the reply limit: the trial ends")
                break
            fails, fresh, wait = after_pass(opencode_pass.result(), time.monotonic() - started, stopped, fails)
            left = await self._seconds_left(environment)
            if left is None:
                break
            if mark is not None:
                marks = [m for m in marks if m < left]
                prompt = REMINDER.format(**_hours_minutes(left))
                self.logger.info(f"reminder at {mark:.0f} s left: {left} s left")
            elif left < min_left:
                break
            elif cut_off(trace):
                prompt = CUT_HINT.format(**_hours_minutes(left))
                self.logger.info(f"cut-off hint {session}: {left} s left")
            else:
                prompt = REPROMPT.format(**_hours_minutes(left))
                self.logger.info(f"re-prompt {session}: {left} s left")
            session += 1
            flags, append = "--continue", True
            if fresh:  # the session cannot be resumed: start a new one on the same workspace
                flags = self.build_cli_flags() or ""
                prompt = RECOVERY_NOTE.format(**_hours_minutes(left)) + instruction
                self.logger.info(f"session recovery after {fails} failed passes in a row: {left} s left")
            if wait:
                await asyncio.sleep(wait)
        await environment.exec(f"echo {session} > /logs/agent/opencode_sessions.txt")
