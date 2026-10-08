"""mini-swe-agent repository-repair rollouts with command execution in Modal."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import shlex
import tempfile
import threading
import time
import tomllib
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache, partial
from pathlib import Path
from typing import Any

from .sandbox import (
    ModalSWEEnvironment,
    SandboxCommandTimeoutError,
    SandboxTransportError,
    ensure_sandbox_app,
    sandbox_settings,
)

logger = logging.getLogger(__name__)


class _EpisodeCancelled(Exception):
    pass


class _TurnTimeLimitExceeded(Exception):
    """A model request outlived the eval's per-turn time limit
    (``MODAL_SWE_TURN_TIME_LIMIT_SECONDS``): the session server gave up on it."""


@dataclass
class _ActiveEpisode:
    cancelled: threading.Event
    sandbox_id: str | None = None


class _EpisodeAbortState:
    """Thread-safe ownership of the episodes running in one worker process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, _ActiveEpisode] = {}

    def start(self) -> tuple[str, threading.Event]:
        episode_id = uuid.uuid4().hex
        cancelled = threading.Event()
        with self._lock:
            self._active[episode_id] = _ActiveEpisode(cancelled=cancelled)
        return episode_id, cancelled

    def attach_sandbox(self, episode_id: str, sandbox_id: str) -> None:
        with self._lock:
            episode = self._active.get(episode_id)
            if episode is not None:
                episode.sandbox_id = sandbox_id

    def finish(self, episode_id: str) -> None:
        with self._lock:
            self._active.pop(episode_id, None)

    def cancel_all(self) -> list[str]:
        with self._lock:
            episodes = list(self._active.values())
            for episode in episodes:
                episode.cancelled.set()
        return [episode.sandbox_id for episode in episodes if episode.sandbox_id]


# The agent keeps complete command output in its in-memory trajectory. Emitting
# every tool observation (often full test tracebacks) and every successful HTTP
# request to the cluster console adds substantial log I/O without improving
# operations. Keep warnings/errors; rollout metrics and verifier evidence are
# logged separately by this adapter.
def _configure_dependency_logging() -> None:
    for dependency_logger in ("agent", "minisweagent", "litellm", "LiteLLM", "httpx"):
        logging.getLogger(dependency_logger).setLevel(logging.WARNING)


_configure_dependency_logging()

_REWARD_START = "__MILES_SWEGYM_REWARD_START__"
_REWARD_END = "__MILES_SWEGYM_REWARD_END__"
_VERIFIER_LOG_TAIL_CHARS = 4000
_MIMO_CODE_SYSTEM_TEMPLATE = """\
You are an agent, your current working directory is {{cwd}}.

You can use the tools available to you to interact with the computer to assist the user in completing tasks.
"""
_MIMO_CODE_INSTANCE_TEMPLATE = """\
Fix the following issue:

{{task}}

When you are finished, run this exact command to end the episode:

echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
"""
_PYTEST_REPORTER = """\
def pytest_runtest_logreport(report):
    if report.when == "call":
        if report.passed:
            status = "PASSED"
        elif report.failed:
            status = "FAILED"
        else:
            status = "SKIPPED"
        print(f"\\n{status} {report.nodeid}", flush=True)
    elif report.failed:
        print(f"\\nERROR {report.nodeid}", flush=True)
"""

# Preserve mini-swe-agent's existing observation semantics while allowing the
# environment to report that it already bounded output inside the Sandbox.
# Without this branch, mini-swe-agent truncates only after Modal has transported
# the complete output (371 MiB in one observed rollout).
_OBSERVATION_TEMPLATE = """\
{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{% if output.output_truncated | default(false) -%}
{% if output.output_limited | default(false) -%}
<warning>
Your last command was terminated after producing too much output.
Narrow the command or redirect verbose output to a file, then inspect that file selectively with head, tail, sed, or grep.
</warning>
{% else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>
{% endif -%}
<output_head>
{{ output.output_head }}
</output_head>
<elided_bytes>
{{ output.output_elided_bytes }} bytes elided before sandbox transport
</elided_bytes>
<output_tail>
{{ output.output_tail }}
</output_tail>
{% elif output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
</warning>
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ output.output | length - 10000 }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}
"""


def _agent_prompt_overrides() -> dict[str, str]:
    profile = os.getenv("MODAL_SWE_AGENT_PROFILE", "swebench")
    if profile == "swebench":
        return {}
    if profile == "mimo-code-bash":
        return {
            "system_template": _MIMO_CODE_SYSTEM_TEMPLATE,
            "instance_template": _MIMO_CODE_INSTANCE_TEMPLATE,
        }
    raise ValueError(f"Unknown Modal SWE agent profile: {profile!r}")


def _task_dir(metadata: dict[str, Any]) -> Path:
    explicit = metadata.get("task_dir")
    instance_id = metadata.get("instance_id")
    if explicit:
        path = Path(explicit)
    elif instance_id:
        path = (
            Path(os.getenv("MODAL_SWE_TASKS_DIR", "/data/tasks"))
            / str(instance_id).lower()
        )
    else:
        raise ValueError("Modal SWE metadata must contain task_dir or instance_id")
    if not path.is_dir():
        raise FileNotFoundError(f"Modal SWE task directory does not exist: {path}")
    return path


def _task_cwd(metadata: dict[str, Any]) -> str:
    cwd = str(metadata.get("sandbox_cwd", "/testbed"))
    if not cwd.startswith("/") or "\x00" in cwd:
        raise ValueError(
            f"Sandbox working directory must be an absolute path, got {cwd!r}"
        )
    return cwd


def _prepare_environment(
    env: ModalSWEEnvironment,
    task_dir: Path,
) -> None:
    """Run an optional benchmark-owned setup script before policy access."""
    setup_script = task_dir / "environment" / "setup.sh"
    if setup_script.is_file():
        destination = "/tmp/miles-task-environment"
        # Modal readiness probes stop after five minutes. Leave one minute for
        # upload/dispatch and the final marker probe; a larger command timeout
        # could finish successfully but could never transition to Ready.
        timeout = int(os.getenv("MODAL_SWE_SETUP_TIMEOUT", "240"))
        if not 0 < timeout <= 240:
            raise ValueError(
                "MODAL_SWE_SETUP_TIMEOUT must be between 1 and 240 seconds "
                "when setup-complete Modal readiness is enabled"
            )
        env.upload_tree(setup_script.parent, destination)
        return_code, output = env.exec(
            (
                f"bash {shlex.quote(destination + '/setup.sh')}; "
                "status=$?; "
                f"rm -rf {shlex.quote(destination)}; "
                "exit $status"
            ),
            cwd=env.cwd,
            timeout=timeout,
        )
        if return_code != 0:
            raise RuntimeError(
                f"Task environment setup failed with return code {return_code}: "
                f"{output[-_VERIFIER_LOG_TAIL_CHARS:]}"
            )

    env.mark_ready()


def _parse_reward(output: str) -> float | None:
    start = output.rfind(_REWARD_START)
    if start < 0:
        return None
    start += len(_REWARD_START)
    end = output.find(_REWARD_END, start)
    if end < 0:
        return None
    raw = output[start:end].strip()
    if not raw:
        return None

    def binary_reward(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            reward = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(reward) or reward not in {0.0, 1.0}:
            return None
        return reward

    reward = binary_reward(raw)
    if reward is not None:
        return reward

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(decoded, dict):
        if "reward" in decoded:
            decoded = decoded["reward"]
        elif len(decoded) == 1:
            decoded = next(iter(decoded.values()))
        else:
            return None
    return binary_reward(decoded)


def _is_context_limit_error(error: Exception) -> bool:
    """Recognize both Miles and OpenAI/SGLang context-limit error shapes."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ == "ContextWindowExceededError":
            return True
        error_code = getattr(current, "code", None)
        if isinstance(error_code, str) and error_code.lower() in {
            "context_length_exceeded",
            "context_window_exceeded",
        }:
            return True
        message = str(current).lower()
        if any(
            marker in message
            for marker in (
                "tito context limit reached",
                "maximum context length",
                "context window exceeded",
                "context_length_exceeded",
                "exceeds the maximum allowed length",
                "longer than the model's context length",
            )
        ):
            return True
        if "input length" in message and "maximum allowed length" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_sandbox_not_found_error(error: Exception) -> bool:
    """Recognize Modal sandbox disappearance without importing Modal internals."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        error_type = type(current)
        qualified_name = f"{error_type.__module__}.{error_type.__name__}".lower()
        message = str(current).lower()
        if (
            ("modal" in qualified_name and "notfound" in qualified_name)
            or "sandbox not found" in message
            or ("sandbox" in message and "not found" in message)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_infrastructure_error(error: BaseException) -> bool:
    """Return whether an episode failure came from a transient external boundary.

    Unknown exceptions deliberately propagate. Treating every adapter exception
    as an infrastructure abort hides deterministic code, dataset, and protocol
    defects and can silently discard a large fraction of training groups.
    """
    transient_names = {
        "APIConnectionError",
        "APITimeoutError",
        "BadGatewayError",
        "ActorDiedError",
        "InternalServerError",
        "RayActorError",
        "RateLimitError",
        "RemoteProtocolError",
        "ServiceUnavailableError",
        "Timeout",
        "TransportError",
    }
    for current in _exception_chain(error):
        if isinstance(current, (ConnectionError, TimeoutError, SandboxTransportError)):
            return True
        error_type = type(current)
        module = error_type.__module__
        if module == "modal.exception" or module.startswith("modal.exception."):
            return True
        if module.startswith(("httpx", "litellm", "openai", "ray.exceptions")):
            if error_type.__name__ in transient_names:
                return True
    return False


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _is_truncated_generation_error(error: BaseException) -> bool:
    """Recognize the session protocol's terminal response-length condition."""
    marker = "truncated generation cannot be extended"
    for current in _exception_chain(error):
        message = str(current).lower()
        status_code = getattr(current, "status_code", None)
        if marker in message and (status_code == 409 or "error code: 409" in message):
            return True
    return False


def _is_request_deadline_error(error: BaseException) -> bool:
    """Recognize the session server's answer to a request it gave up on."""
    return any(
        "request deadline exceeded" in str(current).lower()
        for current in _exception_chain(error)
    )


def _is_turn_time_limit(error: BaseException) -> bool:
    return any(
        isinstance(current, _TurnTimeLimitExceeded)
        for current in _exception_chain(error)
    )


def _exception_metadata(error: BaseException) -> dict[str, Any]:
    chain = _exception_chain(error)
    root = chain[-1]
    return {
        "error_type": f"{type(error).__module__}.{type(error).__name__}",
        "root_error_type": f"{type(root).__module__}.{type(root).__name__}",
        "error_chain": [
            f"{type(item).__module__}.{type(item).__name__}" for item in chain[:8]
        ],
        "root_error": f"{type(root).__name__}: {root}"[:1000],
    }


def _verifier_timeout(task_dir: Path, configured_timeout: int) -> int:
    """Return the smaller of the operator cap and Harbor's task timeout."""
    if configured_timeout <= 0:
        raise ValueError(f"Verifier timeout must be positive, got {configured_timeout}")
    task_config = task_dir / "task.toml"
    if not task_config.is_file():
        return configured_timeout
    try:
        config = tomllib.loads(task_config.read_text())
        task_timeout = int(config["verifier"]["timeout_sec"])
        if task_timeout <= 0:
            raise ValueError
        return min(configured_timeout, task_timeout)
    except (KeyError, TypeError, ValueError, tomllib.TOMLDecodeError):
        logger.warning(
            "Invalid verifier timeout in %s; using %ss", task_config, configured_timeout
        )
        return configured_timeout


def run_verifier(
    env: ModalSWEEnvironment,
    task_dir: Path,
    *,
    configured_timeout: int,
) -> dict[str, Any]:
    """Run Harbor's verifier and retain enough evidence to audit its verdict."""
    timeout = _verifier_timeout(task_dir, configured_timeout)
    # Verifier tests must not exist in the policy sandbox while the agent is
    # acting. Upload them only after the episode terminates.
    env.upload_tree(task_dir / "tests", "/tests")
    pytest_environment = ""
    if os.getenv("MODAL_SWE_INJECT_PYTEST_REPORTER", "0") == "1":
        # Harbor SWE-Gym's parser needs per-test status lines that its test
        # script otherwise omits. Other benchmarks keep their native output.
        pytest_environment = (
            f"printf %s {shlex.quote(_PYTEST_REPORTER)} "
            "> /tmp/miles_pytest_reporter.py; "
            'export PYTHONPATH="/tmp:${PYTHONPATH:-}"; '
            'export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} '
            '-p miles_pytest_reporter"; '
        )
    verify_command = (
        "mkdir -p /logs/verifier; "
        "rm -f /logs/verifier/reward.txt /logs/verifier/reward.json; "
        f"{pytest_environment}"
        # The sandbox runner bounds output before transport and preserves
        # partial diagnostics when the verifier reaches its deadline.
        "bash /tests/test.sh 2>&1; "
        "status=$?; "
        f"echo {_REWARD_START}; "
        "if [ -f /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
        "elif [ -f /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; fi; "
        f"echo; echo {_REWARD_END}; "
        "exit $status"
    )
    return_code, output = env.exec(verify_command, cwd=env.cwd, timeout=timeout)
    return {
        "reward": _parse_reward(output),
        "return_code": return_code,
        "timeout_sec": timeout,
        "output_tail": output[-_VERIFIER_LOG_TAIL_CHARS:],
    }


# What a fresh-Sandbox grader shares with a task's verifier (for SWE-bench Pro, see
# cookbook.miles_disagg.swebench_pro): the ref naming the tree the agent started from,
# and where the grader puts the policy's patch.
_TASK_BASELINE_REF = "refs/miles/task-baseline"
# The tree the policy starts from: the baseline plus what the image leaves untracked.
_TASK_START_REF = "refs/miles/task-start"
_TASK_START_INDEX = "/tmp/miles-task-start.index"
_GRADE_PATCH_DIR = "/tmp/miles-grade"
_GRADE_PATCH_NAME = "policy.patch"
_CAPTURED_PATCH_PATH = "/tmp/miles_policy_capture.patch"


def _fresh_sandbox_grading() -> bool:
    """Grade in a fresh Sandbox of the task image, with the network open, that receives
    only the policy's patch, as SWE-bench Pro V2 re-grades a captured diff on a pristine
    image. Evaluation turns this on; training grades in the agent's own Sandbox."""
    return os.getenv("MODAL_SWE_GRADE_IN_FRESH_SANDBOX", "0") == "1"


def _fresh_grade_applies_patch() -> bool:
    """Apply the policy's patch to the fresh Sandbox's tree before its verifier runs.

    A SWE-bench Pro verifier applies the patch itself (``swebench_pro.GRADE_PATCH_PATH``);
    a MiMo code verifier tests the tree it finds, so without this a fresh Sandbox would
    grade the untouched baseline. Training on MiMo tasks turns it on with fresh grading."""
    return os.getenv("MODAL_SWE_FRESH_GRADE_APPLY_PATCH", "0") == "1"


def _grade_both_ways() -> bool:
    """Grade each episode in its own Sandbox and in a fresh one, train on the fresh grade,
    and log the own-Sandbox grade beside it, to measure how often the two disagree."""
    return os.getenv("MODAL_SWE_GRADE_BOTH_WAYS", "0") == "1"


def _require_submission() -> bool:
    """Score an episode that ends without the agent's submit command as failed.

    The task tells the agent to run the submit command when it is finished, so an episode
    that stops on repeated format errors or on the step or context limit has not
    completed it, whatever its diff grades to. The diff is still graded and logged as
    ``unsubmitted_diff_passed``."""
    return os.getenv("MODAL_SWE_REQUIRE_SUBMISSION", "0") == "1"


def _time_exceeded_is_infrastructure() -> bool:
    """Drop an episode that reaches its wall-clock budget as an infrastructure failure.

    How fast an episode runs depends on its sampler, not only on the policy; the policy's
    own budgets are the step and context limits, which end the episode as
    ``LimitsExceeded``. With this on, the wall-clock budget is a safety net that should
    almost never fire on a healthy sampler."""
    return os.getenv("MODAL_SWE_TIME_EXCEEDED_IS_INFRA", "0") == "1"


def submission_outcome(
    exit_status: str, graded_reward: float
) -> tuple[float, dict[str, int]]:
    """The reward to train on and the submission metrics, from the episode's exit status
    and its graded reward."""
    submitted = exit_status == "Submitted"
    passed = graded_reward > 0.5
    metrics = {
        "unsubmitted": int(not submitted),
        "unsubmitted_diff_passed": int(not submitted and passed),
    }
    if not submitted and _require_submission():
        return 0.0, metrics
    return graded_reward, metrics


def _apply_grade_patch(
    grade_env: ModalSWEEnvironment, patch: bytes, *, timeout: int
) -> dict[str, Any] | None:
    """Apply the uploaded patch to a fresh Sandbox's tree when the task's verifier will
    not. Returns a missing-reward result when it does not apply, which the episode
    treats as an infrastructure failure: the patch is a diff from this same baseline."""
    if not _fresh_grade_applies_patch() or not patch:
        return None
    return_code, output = grade_env.exec(
        f"git apply --binary --whitespace=nowarn {_GRADE_PATCH_DIR}/{_GRADE_PATCH_NAME}",
        cwd=grade_env.cwd,
        timeout=int(timeout),
    )
    if return_code == 0:
        return None
    return {
        "reward": None,
        "return_code": return_code,
        "timeout_sec": timeout,
        "output_tail": (
            "the policy patch did not apply in the fresh Sandbox: "
            + output[-_VERIFIER_LOG_TAIL_CHARS:]
        ),
    }


def grade_episode(
    env: ModalSWEEnvironment, task_dir: Path, *, settings: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float | int]]:
    """The verifier result an episode trains on, and grading metrics to log beside it."""
    if not _grade_both_ways():
        if _fresh_sandbox_grading():
            return grade_in_fresh_sandbox(env, task_dir, settings=settings), {}
        verifier = run_verifier(
            env, task_dir, configured_timeout=int(settings["verify_timeout"])
        )
        return verifier, {}
    # Capture the patch before the own-Sandbox verifier applies the hidden tests to
    # the tree, so the fresh grade receives only the policy's change.
    patch = capture_policy_patch(env, timeout=int(settings["exec_timeout"]))
    metrics: dict[str, float | int] = {}
    own: dict[str, Any] = {}
    try:
        own = run_verifier(
            env, task_dir, configured_timeout=int(settings["verify_timeout"])
        )
        own_reward = own["reward"]
    except SandboxCommandTimeoutError:
        own_reward = None
        metrics["grade_own_sandbox_timeout"] = 1
    if patch is None:
        fresh: dict[str, Any] = {
            "reward": None,
            "return_code": None,
            "timeout_sec": _verifier_timeout(task_dir, int(settings["verify_timeout"])),
            "output_tail": f"no {_TASK_BASELINE_REF} in the agent's Sandbox",
        }
    else:
        fresh = grade_patch_in_fresh_sandbox(
            patch, task_dir, cwd=env.cwd, settings=settings
        )
    if own_reward is not None:
        metrics["grade_own_sandbox_reward"] = own_reward
        if fresh["reward"] is not None:
            metrics["grade_disagreement"] = int(
                (own_reward > 0.5) != (fresh["reward"] > 0.5)
            )
            if metrics["grade_disagreement"]:
                # Aggregate rates cannot say which grade was wrong; the verifier
                # output of each disagreeing episode can.
                logger.warning(
                    "grades disagree for %s: own Sandbox %s (rc=%s), fresh %s (rc=%s); "
                    "own tail: %s | fresh tail: %s",
                    task_dir.name,
                    own_reward,
                    own.get("return_code"),
                    fresh["reward"],
                    fresh.get("return_code"),
                    str(own.get("output_tail", ""))[-400:].replace("\n", " "),
                    str(fresh.get("output_tail", ""))[-400:].replace("\n", " "),
                )
    return fresh, metrics


def _task_resources(task_dir: Path) -> tuple[float | None, int | None]:
    """The CPU and memory a Harbor task declares for its environment, if any."""
    task_config = task_dir / "task.toml"
    if not task_config.is_file():
        return None, None
    environment = tomllib.loads(task_config.read_text()).get("environment", {})
    cpus, memory = environment.get("cpus"), environment.get("memory_mb")
    return (
        float(cpus) if cpus is not None else None,
        int(memory) if memory is not None else None,
    )


def snapshot_task_start(env: ModalSWEEnvironment, *, timeout: int) -> None:
    """Record the tree the policy starts from as ``_TASK_START_REF``: the task baseline
    plus whatever the image leaves untracked, such as an installed .venv or a generated
    lockfile. The captured patch is the policy's change from this tree, so it applies
    to a fresh Sandbox of the same image, which holds the same untracked files; a diff
    from the baseline alone would re-add them there and fail to apply. A temporary
    index leaves the agent's own index and status untouched."""
    index = f"GIT_INDEX_FILE={_TASK_START_INDEX}"
    return_code, output = env.exec(
        f"rm -f {_TASK_START_INDEX} && "
        f"base=$(git rev-parse --verify {_TASK_BASELINE_REF}) && "
        f'{index} git read-tree "$base" && '
        f"{index} git add -A . && "
        f"tree=$({index} git write-tree) && "
        "commit=$(git -c user.name=miles -c user.email=miles@example.invalid "
        'commit-tree "$tree" -p "$base" -m task-start) && '
        f'git update-ref {_TASK_START_REF} "$commit"; '
        f"status=$?; rm -f {_TASK_START_INDEX}; exit $status",
        cwd=env.cwd,
        timeout=timeout,
    )
    if return_code != 0:
        raise RuntimeError(
            f"recording the task start tree failed with return code {return_code}: "
            f"{output[-_VERIFIER_LOG_TAIL_CHARS:]}"
        )


def capture_policy_patch(env: ModalSWEEnvironment, *, timeout: int) -> bytes | None:
    """The policy's change, as its diff from the tree it started from (the task start,
    else the baseline) with untracked files included. None when both refs are gone, so
    there is nothing to grade."""
    return_code, output = env.exec(
        f"ref=$(git rev-parse --verify --quiet {_TASK_START_REF} || "
        f"git rev-parse --verify --quiet {_TASK_BASELINE_REF}) && "
        "{ git add -N . >/dev/null 2>&1 || true; } && "
        f'git diff --binary "$ref" -- . > {_CAPTURED_PATCH_PATH} && '
        f"sha256sum {_CAPTURED_PATCH_PATH}",
        cwd=env.cwd,
        timeout=timeout,
    )
    if return_code != 0 or not output.strip():
        return None
    patch = env.download_file(_CAPTURED_PATCH_PATH)
    if hashlib.sha256(patch).hexdigest() != output.split()[0]:
        raise SandboxTransportError("the policy patch changed in transfer")
    return patch


def grade_in_fresh_sandbox(
    agent_env: ModalSWEEnvironment,
    task_dir: Path,
    *,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """``run_verifier``'s result for the policy's patch, graded in a fresh Sandbox.

    The fresh Sandbox runs the task's setup, so its tree is the baseline the agent
    started from; it receives the patch and nothing else of the agent's Sandbox. Its
    network is open, as a benchmark's verifier phase may download dependencies. It
    takes the task's declared CPU and memory, which the benchmark grades with.
    """
    started = time.perf_counter()
    patch = capture_policy_patch(agent_env, timeout=int(settings["exec_timeout"]))
    if patch is None:
        return {
            "reward": None,
            "return_code": None,
            "timeout_sec": _verifier_timeout(task_dir, int(settings["verify_timeout"])),
            "output_tail": f"no {_TASK_BASELINE_REF} in the agent's Sandbox",
        }
    return grade_patch_in_fresh_sandbox(
        patch, task_dir, cwd=agent_env.cwd, settings=settings, started=started
    )


def grade_patch_in_fresh_sandbox(
    patch: bytes,
    task_dir: Path,
    *,
    cwd: str,
    settings: dict[str, Any],
    started: float | None = None,
) -> dict[str, Any]:
    """``grade_in_fresh_sandbox`` for a patch already captured, such as one an eval
    stored, so a sample can be re-graded without rerunning its agent."""
    if started is None:
        started = time.perf_counter()
    timeout = _verifier_timeout(task_dir, int(settings["verify_timeout"]))
    cpu, memory_mib = _task_resources(task_dir)
    grade_env = ModalSWEEnvironment(
        task_dir,
        cwd=cwd,
        lifetime=timeout + 900,
        exec_timeout=int(settings["exec_timeout"]),
        app_name=str(settings["app_name"]),
        cpu=cpu,
        memory_mib=memory_mib,
        block_network=False,
    )
    try:
        _prepare_environment(grade_env, task_dir)
        with tempfile.TemporaryDirectory() as staging:
            (Path(staging) / _GRADE_PATCH_NAME).write_bytes(patch)
            grade_env.upload_tree(staging, _GRADE_PATCH_DIR)
        verifier = _apply_grade_patch(grade_env, patch, timeout=timeout)
        if verifier is None:
            verifier = run_verifier(
                grade_env, task_dir, configured_timeout=int(settings["verify_timeout"])
            )
    except SandboxCommandTimeoutError as error:
        # A verifier timeout is still graded (zero), so keep what it graded.
        error.policy_patch = patch
        raise
    finally:
        _stop_environment(grade_env)
    verifier["policy_patch"] = patch
    verifier["grade_metrics"] = {
        "grade_sandbox_boot_time": grade_env.boot_time,
        "grade_time": time.perf_counter() - started,
        "policy_patch_bytes": len(patch),
    }
    return verifier


def _graded_outputs(
    verifier: dict[str, Any] | None, patch: bytes | None
) -> dict[str, Any]:
    """What a fresh-Sandbox grade keeps beside the reward, so the sample can be re-graded
    or diagnosed without rerunning the agent: the exact patch it graded (base64) and the
    verifier's output tail. Grading in the agent's own Sandbox keeps neither."""
    if patch is None:
        return {}
    outputs = {"policy_patch_b64": base64.b64encode(patch).decode("ascii")}
    if verifier is not None:
        outputs["verifier_output_tail"] = verifier["output_tail"]
    return outputs


def _failure(
    reason: str,
    *,
    infrastructure: bool = True,
    total_time: float | None = None,
    agent_queue_time: float | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    metrics: dict[str, float | int] = {
        "infra_error": int(infrastructure),
        "policy_failure": int(not infrastructure),
    }
    if total_time is not None:
        metrics["total_time"] = total_time
    if agent_queue_time is not None:
        metrics["agent_queue_time"] = agent_queue_time
    metadata = dict(metadata)
    if isinstance(metadata.get("agent_metrics"), dict):
        metrics.update(metadata.pop("agent_metrics"))
    result = {
        "exit_status": reason,
        "eval_report": {},
        "agent_metrics": metrics,
        **metadata,
    }
    if infrastructure:
        result["_miles_abort"] = True
    else:
        # Preserve recorded policy tokens and train the terminal outcome with
        # zero reward. The normal reward function reads this field.
        result["reward"] = 0.0
    return result


def _validated_reward(sample: Any) -> float:
    """Read a verifier reward without turning missing infrastructure into zero."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if "reward" not in metadata:
        raise ValueError(
            "Modal SWE sample has no verifier reward; the adapter must abort "
            "infrastructure failures before reward evaluation"
        )
    reward = metadata["reward"]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError(
            f"Modal SWE reward must be numeric, got {type(reward).__name__}"
        )
    reward = float(reward)
    if not math.isfinite(reward):
        raise ValueError(f"Modal SWE reward must be finite, got {reward}")
    if reward not in {0.0, 1.0}:
        raise ValueError(f"Modal SWE reward must be binary, got {reward}")
    return reward


async def reward_func(
    args: Any,
    samples: Any,
    **kwargs: Any,
) -> float | list[float]:
    """Custom RM hook for rewards already computed by Harbor's verifier.

    This is intentionally strict. A missing reward means the verifier contract
    was broken; silently mapping it to zero would train on an infrastructure
    failure as though it were a policy failure.
    """
    del args, kwargs
    if isinstance(samples, list):
        return [_validated_reward(sample) for sample in samples]
    return _validated_reward(samples)


def pick_latest_leaf(samples: list[Any], session_metadata: dict[str, Any]) -> list[Any]:
    """Select the final committed branch for one agent execution.

    Session v2 retains a trajectory tree so tree-RL and subagent workloads can
    train multiple terminal leaves. Modal SWE currently has a one-execution,
    one-training-sample contract, so older retry/rollback branches are not
    separate rollouts. ``node_id`` is the session server's monotonic commit
    order; the greatest leaf id is the last successfully committed trajectory.
    """
    del session_metadata
    if not samples:
        return []

    def node_id(sample: Any) -> int:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        leaf = metadata.get("leaf")
        value = leaf.get("node_id") if isinstance(leaf, dict) else None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "Modal SWE session-v2 sample is missing an integer leaf.node_id"
            )
        return value

    return [max(samples, key=node_id)]


def postprocess_samples(
    samples: list[Any], session_metadata: dict[str, Any]
) -> list[Any]:
    """Apply the v2 default policy, then exclude infrastructure failures.

    ``_miles_abort`` is private adapter control metadata: it must affect sample
    status, but must never become model-visible training metadata. Keeping this
    in the session-v2 postprocessor hook avoids a Modal-specific failure
    protocol in Miles' generic agentic rollout path.
    """
    from miles.rollout.session.v2.postprocessor_hub.default_postprocess import (
        default_postprocess,
    )
    from miles.utils.types import Sample

    processed = default_postprocess(samples, session_metadata)
    agent_metadata = session_metadata.get("agent") or {}
    if agent_metadata.get("_miles_abort"):
        for sample in processed:
            sample.status = Sample.Status.ABORTED
            sample.reward = None
            sample.metadata.pop("_miles_abort", None)
        # Miles drops aborted samples before the metrics hook, so per-pool abort
        # rates are counted from this line. An episode whose first request failed
        # has no response to name its pool.
        sources = {s.metadata.get("rollout_source") for s in processed} - {None}
        logger.info(
            "aborted episode source=%s reason=%s phase=%s error=%s",
            ",".join(sorted(map(str, sources))) or "unknown",
            agent_metadata.get("exit_status"),
            agent_metadata.get("failure_phase"),
            str(agent_metadata.get("agent_error", ""))[:200].replace("\n", " "),
        )
    return processed


@dataclass(frozen=True)
class _EnvironmentSnapshot:
    command_count: int
    exec_time: float
    exec_remote_time: float
    upload_time: float
    upload_bytes: int
    command_timeout_count: int
    output_bytes: int
    transferred_bytes: int
    output_truncated_count: int
    output_hard_limit_count: int

    @classmethod
    def empty(cls) -> _EnvironmentSnapshot:
        return cls(
            command_count=0,
            exec_time=0.0,
            exec_remote_time=0.0,
            upload_time=0.0,
            upload_bytes=0,
            command_timeout_count=0,
            output_bytes=0,
            transferred_bytes=0,
            output_truncated_count=0,
            output_hard_limit_count=0,
        )

    @classmethod
    def capture(cls, env: ModalSWEEnvironment) -> _EnvironmentSnapshot:
        return cls(
            command_count=env.command_count,
            exec_time=env.exec_time,
            exec_remote_time=env.exec_remote_time,
            upload_time=env.upload_time,
            upload_bytes=env.upload_bytes,
            command_timeout_count=env.command_timeout_count,
            output_bytes=env.command_output_bytes,
            transferred_bytes=env.command_transferred_bytes,
            output_truncated_count=env.command_output_truncated_count,
            output_hard_limit_count=env.command_output_hard_limit_count,
        )

    def since(self, previous: _EnvironmentSnapshot) -> _EnvironmentSnapshot:
        return _EnvironmentSnapshot(
            **{
                field: max(0, getattr(self, field) - getattr(previous, field))
                for field in self.__dataclass_fields__
            }
        )


def _environment_metrics(
    env: ModalSWEEnvironment,
    *,
    agent_queue_time: float,
    agent_dispatch_queue_time: float = 0.0,
    total_time: float,
    agent_start_snapshot: _EnvironmentSnapshot | None = None,
    agent_snapshot: _EnvironmentSnapshot | None = None,
) -> dict[str, float | int]:
    """Return sandbox/tool timing components and an overlap-free total."""
    agent_start_snapshot = agent_start_snapshot or _EnvironmentSnapshot.empty()
    post_agent_snapshot = agent_snapshot or _EnvironmentSnapshot.capture(env)
    agent_snapshot = post_agent_snapshot.since(agent_start_snapshot)

    agent_slice = slice(
        agent_start_snapshot.command_count,
        post_agent_snapshot.command_count,
    )
    agent_durations = env.exec_durations[agent_slice]
    agent_remote_durations = env.exec_remote_durations[agent_slice]
    agent_transport_durations = env.exec_transport_durations[agent_slice]
    agent_input_sizes = env.command_input_sizes[agent_slice]
    setup_exec_time = agent_start_snapshot.exec_time
    setup_upload_time = agent_start_snapshot.upload_time
    verifier_exec_time = max(0.0, env.exec_time - post_agent_snapshot.exec_time)
    verifier_exec_remote_time = max(
        0.0,
        env.exec_remote_time - post_agent_snapshot.exec_remote_time,
    )
    verifier_upload_time = max(
        0.0,
        env.upload_time - post_agent_snapshot.upload_time,
    )
    if env.boot_time is None or env.readiness_time is None:
        raise RuntimeError("Sandbox metrics requested before task setup became ready")
    setup_time = setup_exec_time + setup_upload_time
    # Boot/readiness now includes task setup. Keep the setup metric as an
    # overlapping breakdown, but subtract it from the remaining command time
    # so total_tool_time does not count setup twice.
    sandbox_non_generation_time = env.boot_time + max(
        0.0,
        env.exec_time + env.upload_time - setup_time,
    )
    return {
        "agent_queue_time": agent_queue_time,
        "agent_dispatch_queue_time": agent_dispatch_queue_time,
        "sandbox_boot_time": env.boot_time,
        "sandbox_schedule_time": env.schedule_time,
        "sandbox_readiness_time": env.readiness_time,
        "sandbox_setup_time": setup_time,
        "tool_calls": agent_snapshot.command_count,
        "agent_tool_exec_time": agent_snapshot.exec_time,
        "agent_tool_remote_exec_time": agent_snapshot.exec_remote_time,
        "agent_tool_transport_time": max(
            0.0,
            agent_snapshot.exec_time - agent_snapshot.exec_remote_time,
        ),
        "agent_tool_exec_mean": (
            sum(agent_durations) / len(agent_durations) if agent_durations else 0.0
        ),
        "agent_tool_remote_exec_mean": (
            sum(agent_remote_durations) / len(agent_remote_durations)
            if agent_remote_durations
            else 0.0
        ),
        "agent_tool_transport_mean": (
            sum(agent_transport_durations) / len(agent_transport_durations)
            if agent_transport_durations
            else 0.0
        ),
        "agent_tool_exec_p90": (
            sorted(agent_durations)[round((len(agent_durations) - 1) * 0.90)]
            if agent_durations
            else 0.0
        ),
        "agent_tool_exec_max": max(agent_durations, default=0.0),
        "agent_tool_input_mib": sum(agent_input_sizes) / (1024 * 1024),
        "agent_tool_input_mean_bytes": (
            sum(agent_input_sizes) / len(agent_input_sizes)
            if agent_input_sizes
            else 0.0
        ),
        "agent_tool_input_max_bytes": max(agent_input_sizes, default=0),
        "agent_tool_input_over_64k_count": sum(
            size > 65536 for size in agent_input_sizes
        ),
        "agent_tool_input_over_64k_ratio": (
            sum(size > 65536 for size in agent_input_sizes) / len(agent_input_sizes)
            if agent_input_sizes
            else 0.0
        ),
        "tool_timeout_count": agent_snapshot.command_timeout_count,
        "agent_tool_output_mib": agent_snapshot.output_bytes / (1024 * 1024),
        "agent_tool_transferred_mib": agent_snapshot.transferred_bytes / (1024 * 1024),
        "agent_tool_output_truncated_count": agent_snapshot.output_truncated_count,
        "agent_tool_output_truncated_ratio": (
            agent_snapshot.output_truncated_count / agent_snapshot.command_count
            if agent_snapshot.command_count
            else 0.0
        ),
        "agent_tool_output_hard_limit_count": agent_snapshot.output_hard_limit_count,
        "agent_tool_output_hard_limit_ratio": (
            agent_snapshot.output_hard_limit_count / agent_snapshot.command_count
            if agent_snapshot.command_count
            else 0.0
        ),
        "verifier_upload_time": verifier_upload_time,
        "verifier_upload_mib": max(
            0,
            env.upload_bytes - post_agent_snapshot.upload_bytes,
        )
        / (1024 * 1024),
        "verifier_exec_time": verifier_exec_time,
        "verifier_remote_exec_time": verifier_exec_remote_time,
        "verifier_transport_time": max(
            0.0,
            verifier_exec_time - verifier_exec_remote_time,
        ),
        "verifier_output_mib": max(
            0,
            env.command_output_bytes - post_agent_snapshot.output_bytes,
        )
        / (1024 * 1024),
        "verifier_transferred_mib": max(
            0,
            env.command_transferred_bytes - post_agent_snapshot.transferred_bytes,
        )
        / (1024 * 1024),
        "total_tool_time": sandbox_non_generation_time,
        "total_time": total_time,
    }


def _attach_client_model_timings(
    metrics: dict[str, Any],
    durations: list[float],
) -> dict[str, Any]:
    """Attach exact timings; the rollout log hook computes batch summaries."""
    metrics["client_model_request_durations_seconds"] = list(durations)
    model_request_time = sum(durations)
    total_time = float(metrics.get("total_time", 0.0))
    metrics["model_request_count"] = len(durations)
    metrics["model_request_time"] = model_request_time
    interaction_time = max(0.0, total_time - model_request_time)
    sandbox_time = min(
        interaction_time,
        max(0.0, float(metrics.get("total_tool_time", 0.0))),
    )
    metrics["interaction_time"] = interaction_time
    metrics["interaction_sandbox_time"] = sandbox_time
    # Agent setup, prompt rendering, Python scheduling, and any other time not
    # spent in a model HTTP request or a measured Sandbox operation. Keeping
    # this residual explicit makes instrumentation gaps visible.
    metrics["interaction_unattributed_time"] = max(
        0.0,
        interaction_time - sandbox_time,
    )
    if total_time > 0:
        metrics["generation_time_ratio"] = min(1.0, model_request_time / total_time)
        metrics["interaction_time_ratio"] = max(
            0.0, 1.0 - metrics["generation_time_ratio"]
        )
        metrics["generation_bound"] = int(metrics["generation_time_ratio"] > 0.5)
    return metrics


def _model_token_metrics(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize API usage from the responses retained by mini-swe-agent."""
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    for message in messages:
        extra = message.get("extra")
        if not isinstance(extra, dict):
            continue
        response = extra.get("response")
        if not isinstance(response, dict):
            continue
        usage = response.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt >= 0:
            prompt_tokens.append(prompt)
        if (
            isinstance(completion, int)
            and not isinstance(completion, bool)
            and completion >= 0
        ):
            completion_tokens.append(completion)
    return {
        "model_prompt_tokens_total": sum(prompt_tokens),
        "model_prompt_tokens_max": max(prompt_tokens, default=0),
        "model_completion_tokens_total": sum(completion_tokens),
        "model_completion_tokens_max": max(completion_tokens, default=0),
    }


# Off unless set: a directory each finished episode writes its trajectory to, as one
# JSON file, for the sampler benchmark's replay (cookbook/miles_disagg/bench).
TRAJECTORY_DUMP_ENV = "MODAL_SWE_TRAJECTORY_DUMP_DIR"
TRAJECTORY_FORMAT = "modal-swe-trajectory/v1"


def _token_count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _model_call_fields(response: Any) -> dict[str, Any]:
    """What a retained model response says about its request: token counts from its
    usage and the finish reason. A response mini-swe-agent could only keep as a repr
    still marks the message as a model call."""
    fields: dict[str, Any] = {"model_call": True}
    if not isinstance(response, dict):
        return fields
    usage = response.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens"):
            if (count := _token_count(usage.get(key))) is not None:
                fields[key] = count
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            if (cached := _token_count(details.get("cached_tokens"))) is not None:
                fields["cached_tokens"] = cached
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        if (reason := choices[0].get("finish_reason")) is not None:
            fields["finish_reason"] = reason
    return fields


def _trajectory_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return {
        "id": call.get("id"),
        "type": call.get("type") or "function",
        "function": {
            "name": function.get("name"),
            "arguments": function.get("arguments"),
        },
    }


def _trajectory_message(message: dict[str, Any]) -> dict[str, Any]:
    """One message as the dump keeps it: what the API saw, plus the request's token
    counts on every message that carries a model response. That is each assistant
    turn, and the format-error message mini-swe-agent sends instead of a response it
    could not parse (whose output never re-enters the context)."""
    record: dict[str, Any] = {
        "role": message.get("role"),
        "content": message.get("content"),
    }
    for key in ("reasoning_content", "tool_call_id", "name"):
        if message.get(key) is not None:
            record[key] = message[key]
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        record["tool_calls"] = [
            _trajectory_tool_call(call) for call in tool_calls if isinstance(call, dict)
        ]
    extra = message.get("extra")
    if isinstance(extra, dict):
        if "response" in extra:
            record.update(_model_call_fields(extra["response"]))
            if isinstance(extra.get("timestamp"), (int, float)):
                record["timestamp"] = float(extra["timestamp"])
        if record["role"] == "exit" and "exit_status" in extra:
            record["exit_status"] = extra["exit_status"]
    return record


def trajectory_record(
    messages: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    result: dict[str, Any],
    durations: list[float],
    request_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """An episode's trajectory for replay: its ordered messages, and the duration of
    every model HTTP request in order. Each duration is also set on its model call
    when the two counts match; a resent or failed request leaves them unpaired."""
    records = [
        _trajectory_message(message)
        for message in messages
        if isinstance(message, dict)
    ]
    calls = [record for record in records if record.get("model_call")]
    durations = [float(duration) for duration in durations]
    if len(calls) == len(durations):
        for record, duration in zip(calls, durations, strict=True):
            record["request_seconds"] = duration
    return {
        "format": TRAJECTORY_FORMAT,
        "instance_id": metadata.get("instance_id"),
        # Stamped by the eval's generate hook; training samples have none.
        "sample_index": metadata.get("eval_sample_index"),
        "exit_status": result.get("exit_status"),
        "reward": result.get("reward"),
        "model_calls": len(calls),
        "model_request_durations_seconds": durations,
        "request_kwargs": dict(request_kwargs or {}),
        "tools": tools,
        "messages": records,
    }


def write_trajectory(directory: Path, record: dict[str, Any]) -> Path:
    """Write one trajectory under a unique name, atomically, so a reader never sees a
    partial file and two episodes of one task never collide."""
    directory.mkdir(parents=True, exist_ok=True)
    instance = "".join(
        char if char.isalnum() or char in "-_." else "_"
        for char in str(record.get("instance_id") or "unknown")
    )
    index = record.get("sample_index")
    name = f"{instance}.s{'x' if index is None else index}.{uuid.uuid4().hex[:12]}"
    path = directory / f"{name}.json"
    partial_path = directory / f"{name}.partial"
    partial_path.write_text(json.dumps(record, default=str))
    os.replace(partial_path, path)
    return path


def _request_tools(model: Any) -> list[dict[str, Any]] | None:
    """The tools every request of ``model`` carries: mini-swe-agent's LiteLLM model
    sends its bash tool with each one, and the chat template renders it."""
    try:
        from minisweagent.models.litellm_model import LitellmModel
        from minisweagent.models.utils.actions_toolcall import BASH_TOOL
    except ImportError:
        return None
    return [BASH_TOOL] if isinstance(model, LitellmModel) else None


@dataclass
class _TrajectoryCapture:
    """Collects an episode's agent while it runs and writes its trajectory at the end."""

    directory: Path
    metadata: dict[str, Any]
    request_kwargs: dict[str, Any]
    agent: Any = None
    model: Any = None
    durations: list[float] = field(default_factory=list)

    @classmethod
    def from_environment(
        cls, metadata: dict[str, Any], request_kwargs: dict[str, Any]
    ) -> _TrajectoryCapture | None:
        directory = os.getenv(TRAJECTORY_DUMP_ENV, "").strip()
        if not directory:
            return None
        return cls(Path(directory), dict(metadata or {}), dict(request_kwargs or {}))

    def attach(self, agent: Any, model: Any, durations: list[float]) -> None:
        self.agent, self.model, self.durations = agent, model, durations

    def write(self, result: Any) -> Path | None:
        """Write the trajectory, or nothing when the agent never started. A failed
        write is logged and never fails the episode."""
        if self.agent is None or not isinstance(result, dict):
            return None
        try:
            record = trajectory_record(
                list(getattr(self.agent, "messages", None) or []),
                metadata=self.metadata,
                result=result,
                durations=self.durations,
                request_kwargs=self.request_kwargs,
                tools=_request_tools(self.model),
            )
            return write_trajectory(self.directory, record)
        except Exception:
            logger.warning(
                "Failed to write the trajectory of %s to %s",
                self.metadata.get("instance_id"),
                self.directory,
                exc_info=True,
            )
            return None


def _instrument_model_requests(
    model: Any,
    durations: list[float],
    phase_callback: Callable[[str], None] | None = None,
    cancelled: threading.Event | None = None,
    turn_time_limit: bool = False,
) -> None:
    """Measure each real LiteLLM HTTP attempt as perceived by the agent. With
    ``turn_time_limit``, a request the session server gave up on raises
    ``_TurnTimeLimitExceeded``, which the model never retries."""
    query = getattr(model, "_query", None)
    if not callable(query):
        logger.warning(
            "Model %s has no callable _query; client request timing unavailable",
            type(model).__name__,
        )
        return

    def timed_query(*args: Any, **kwargs: Any) -> Any:
        _raise_if_cancelled(cancelled)
        started = time.perf_counter()
        if phase_callback is not None:
            phase_callback("model_generation")
        try:
            result = query(*args, **kwargs)
            _raise_if_cancelled(cancelled)
            return result
        except Exception as error:
            if turn_time_limit and _is_request_deadline_error(error):
                raise _TurnTimeLimitExceeded(str(error)[:500]) from error
            raise
        finally:
            durations.append(time.perf_counter() - started)
            if phase_callback is not None:
                phase_callback("interaction")

    try:
        model._query = timed_query
    except (AttributeError, TypeError):
        logger.warning(
            "Model %s does not allow _query instrumentation; client request timing unavailable",
            type(model).__name__,
        )


def _raise_if_cancelled(cancelled: threading.Event | None) -> None:
    if cancelled is not None and cancelled.is_set():
        raise _EpisodeCancelled


async def _terminate_sandboxes(sandbox_ids: list[str]) -> None:
    if not sandbox_ids:
        return

    import modal

    async def terminate(sandbox_id: str) -> None:
        sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
        await sandbox.terminate.aio()

    results = await asyncio.gather(
        *(terminate(sandbox_id) for sandbox_id in sandbox_ids),
        return_exceptions=True,
    )
    for sandbox_id, result in zip(sandbox_ids, results, strict=True):
        if isinstance(result, Exception):
            logger.warning(
                "Failed to terminate cancelled Modal Sandbox %s: %s: %s",
                sandbox_id,
                type(result).__name__,
                result,
            )


def _stop_environment(env: ModalSWEEnvironment) -> None:
    try:
        env.stop()
    except Exception:
        logger.warning("Failed to clean up Modal SWE environment", exc_info=True)


def _run_episode_sync(
    *,
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
    queued_at: float,
    dispatch_queue_time: float = 0.0,
    phase_callback: Callable[[str], None] | None = None,
    cancelled: threading.Event | None = None,
    sandbox_started: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one episode; with ``MODAL_SWE_TRAJECTORY_DUMP_DIR`` set, also write its
    trajectory there once it ends. The dump never changes the episode's result."""
    trajectory = _TrajectoryCapture.from_environment(metadata, request_kwargs)
    result = _run_episode(
        base_url=base_url,
        prompt=prompt,
        request_kwargs=request_kwargs,
        metadata=metadata,
        queued_at=queued_at,
        dispatch_queue_time=dispatch_queue_time,
        phase_callback=phase_callback,
        cancelled=cancelled,
        sandbox_started=sandbox_started,
        trajectory=trajectory,
    )
    if trajectory is not None:
        trajectory.write(result)
    return result


def _run_episode(
    *,
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
    queued_at: float,
    dispatch_queue_time: float = 0.0,
    phase_callback: Callable[[str], None] | None = None,
    cancelled: threading.Event | None = None,
    sandbox_started: Callable[[str], None] | None = None,
    trajectory: _TrajectoryCapture | None = None,
) -> dict[str, Any]:
    from minisweagent.agents import get_agent
    from minisweagent.config import get_config_from_spec
    from minisweagent.models import get_model

    # mini-swe-agent configures its logger during import, so enforce the
    # production console levels once more after those imports complete.
    _configure_dependency_logging()

    downstream_phase_callback = phase_callback
    current_phase = "executor_queue"

    def set_phase(phase: str) -> None:
        nonlocal current_phase
        current_phase = phase
        if downstream_phase_callback is not None:
            downstream_phase_callback(phase)

    agent_queue_time = time.perf_counter() - queued_at
    settings = sandbox_settings()
    task_dir = _task_dir(metadata)
    task_cwd = _task_cwd(metadata)
    verifier_timeout = _verifier_timeout(task_dir, int(settings["verify_timeout"]))
    started = time.perf_counter()
    env: ModalSWEEnvironment | None = None

    try:
        _raise_if_cancelled(cancelled)
        set_phase("sandbox_boot")
        env = ModalSWEEnvironment(
            task_dir,
            cwd=task_cwd,
            lifetime=int(settings["episode_timeout"]) + verifier_timeout + 300,
            exec_timeout=int(settings["exec_timeout"]),
            app_name=str(settings["app_name"]),
        )
        if sandbox_started is not None:
            sandbox_started(str(env.sandbox.object_id))
        _raise_if_cancelled(cancelled)
        set_phase("sandbox_setup")
        _prepare_environment(env, task_dir)
        if _fresh_sandbox_grading() or _grade_both_ways():
            snapshot_task_start(env, timeout=int(settings["exec_timeout"]))
        _raise_if_cancelled(cancelled)
        set_phase("agent_setup")
        agent_start_snapshot = _EnvironmentSnapshot.capture(env)
        config = get_config_from_spec("swebench")
        client_model_request_durations: list[float] = []
        model_kwargs = {
            **request_kwargs,
            "api_base": f"{base_url.rstrip('/')}/v1",
            "api_key": "EMPTY",
            "drop_params": True,
            # LiteLLM's shorter implicit default can expire a valid long-context
            # decode while the episode itself still has ample wall-clock budget.
            "timeout": float(os.getenv("MODAL_SWE_MODEL_REQUEST_TIMEOUT", "1800")),
            # A hidden client retry can overlap the original request after a
            # transport disconnect. The Miles session server also serializes
            # same-session requests, but disabling LiteLLM's inner retry avoids
            # duplicate model work. Production configs also set mini-SWE-Agent's
            # outer attempt limit to one.
            "num_retries": 0,
        }
        model_config = {
            **config.get("model", {}),
            "model_name": f"openai/{os.getenv('AGENT_MODEL_NAME', 'model')}",
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
            "observation_template": _OBSERVATION_TEMPLATE,
        }
        agent_config = {
            **config.get("agent", {}),
            **_agent_prompt_overrides(),
            "step_limit": int(os.getenv("MODAL_SWE_MAX_STEPS", "100")),
            "wall_time_limit_seconds": int(settings["episode_timeout"]),
            "cost_limit": 0.0,
            "output_path": None,
        }
        model = get_model(config=model_config)
        _instrument_model_requests(
            model,
            client_model_request_durations,
            set_phase,
            cancelled,
            turn_time_limit=bool(os.getenv("MODAL_SWE_TURN_TIME_LIMIT_SECONDS")),
        )
        # BadRequest errors are deterministic for a fixed request. Retrying a
        # TITO validation or context-limit 400 ten times only burns rollout
        # slots. Transient transport/server errors remain retryable.
        try:
            import litellm

            # LiteLLM prints an issue URL and debugging hint directly to stderr
            # for every mapped client error, bypassing Python logger levels.
            # Context exhaustion is an expected terminal condition for these
            # long trajectories, so retain it in aggregate outcome metrics
            # without flooding the distributed job log.
            litellm.suppress_debug_info = True
            if hasattr(model, "abort_exceptions"):
                # A cancelled episode raises from inside the request, and must
                # end there even when an eval allows the request a retry.
                model.abort_exceptions = list(
                    dict.fromkeys(
                        [
                            *model.abort_exceptions,
                            litellm.exceptions.BadRequestError,
                            _EpisodeCancelled,
                            _TurnTimeLimitExceeded,
                        ]
                    )
                )
        except (AttributeError, ImportError):
            logger.warning("Unable to mark LiteLLM BadRequestError as non-retryable")
        agent = get_agent(model, env, agent_config, default_type="default")
        if trajectory is not None:
            trajectory.attach(agent, model, client_model_request_durations)
        # DefaultAgent logs every message (including complete command output)
        # at DEBUG. Miles configures the process root logger independently, so
        # logger levels alone can be reset by initialization order; disabling
        # this trajectory logger is deterministic. The messages remain in
        # ``agent.messages`` and therefore in the returned training sample.
        agent.logger.disabled = True
        context_limit_exceeded = False
        generation_limit_exceeded = False
        try:
            _raise_if_cancelled(cancelled)
            set_phase("interaction")
            result = agent.run(str(prompt))
        except SandboxCommandTimeoutError:
            logger.warning("Modal SWE command timed out for %s", task_dir.name)
            elapsed = time.perf_counter() - started
            agent_metrics = _environment_metrics(
                env,
                agent_queue_time=agent_queue_time,
                agent_dispatch_queue_time=dispatch_queue_time,
                total_time=elapsed,
                agent_start_snapshot=agent_start_snapshot,
            )
            _attach_client_model_timings(
                agent_metrics,
                client_model_request_durations,
            )
            return _failure(
                "command_timeout",
                infrastructure=False,
                total_time=elapsed,
                agent_queue_time=agent_queue_time,
                agent_metrics=agent_metrics,
            )
        except Exception as error:
            if cancelled is not None and cancelled.is_set():
                raise _EpisodeCancelled from None
            if _is_turn_time_limit(error):
                # A turn still generating at the eval's time limit is a policy
                # outcome: the episode fails with zero reward instead of resending
                # or rerunning. Checked before the infrastructure test, which the
                # underlying gateway error would otherwise match.
                elapsed = time.perf_counter() - started
                agent_metrics = _environment_metrics(
                    env,
                    agent_queue_time=agent_queue_time,
                    agent_dispatch_queue_time=dispatch_queue_time,
                    total_time=elapsed,
                    agent_start_snapshot=agent_start_snapshot,
                )
                _attach_client_model_timings(
                    agent_metrics,
                    client_model_request_durations,
                )
                return _failure(
                    "TurnTimeLimit",
                    infrastructure=False,
                    total_time=elapsed,
                    agent_queue_time=agent_queue_time,
                    agent_metrics=agent_metrics,
                )
            if _is_context_limit_error(error):
                # Reaching the configured context budget is a normal policy
                # limit, not an infrastructure failure. Grade the current
                # sandbox state just like mini-SWE-Agent's step limit.
                context_limit_exceeded = True
                result = {
                    "exit_status": "LimitsExceeded",
                    "submission": "",
                }
                logger.debug(
                    "Modal SWE context limit reached for %s; running verifier",
                    task_dir.name,
                )
            elif _is_truncated_generation_error(error):
                # A response that reached its generation cap is a valid policy
                # terminal. The session protocol cannot extend that leaf, so
                # grade the sandbox state produced up to the truncation.
                generation_limit_exceeded = True
                result = {
                    "exit_status": "LimitsExceeded",
                    "submission": "",
                }
                logger.debug(
                    "Modal SWE generation limit reached for %s; running verifier",
                    task_dir.name,
                )
            elif _is_infrastructure_error(error):
                logger.warning(
                    "Modal SWE agent failed for %s: %s: %s",
                    task_dir.name,
                    type(error).__name__,
                    str(error)[:500],
                )
                elapsed = time.perf_counter() - started
                reason = (
                    "sandbox_not_found"
                    if _is_sandbox_not_found_error(error)
                    else "agent_error"
                )
                agent_metrics = _environment_metrics(
                    env,
                    agent_queue_time=agent_queue_time,
                    agent_dispatch_queue_time=dispatch_queue_time,
                    total_time=elapsed,
                    agent_start_snapshot=agent_start_snapshot,
                )
                _attach_client_model_timings(
                    agent_metrics,
                    client_model_request_durations,
                )
                return _failure(
                    reason,
                    total_time=elapsed,
                    agent_queue_time=agent_queue_time,
                    agent_error=f"{type(error).__name__}: {error}"[:1000],
                    failure_phase=current_phase,
                    agent_metrics=agent_metrics,
                    **_exception_metadata(error),
                    **env.lifecycle_diagnostics(),
                )
            else:
                raise

        exit_status = result.get("exit_status", "completed")
        if exit_status == "TimeExceeded" and _time_exceeded_is_infrastructure():
            logger.warning(
                "Modal SWE episode reached its wall-clock budget for %s; "
                "dropped as an infrastructure failure",
                task_dir.name,
            )
            elapsed = time.perf_counter() - started
            agent_metrics = _environment_metrics(
                env,
                agent_queue_time=agent_queue_time,
                agent_dispatch_queue_time=dispatch_queue_time,
                total_time=elapsed,
                agent_start_snapshot=agent_start_snapshot,
            )
            _attach_client_model_timings(
                agent_metrics,
                client_model_request_durations,
            )
            return _failure(
                "time_exceeded",
                total_time=elapsed,
                agent_queue_time=agent_queue_time,
                failure_phase="interaction",
                agent_metrics=agent_metrics,
            )
        agent_snapshot = _EnvironmentSnapshot.capture(env)
        verify_started = time.perf_counter()
        _raise_if_cancelled(cancelled)
        set_phase("verification")
        try:
            verifier, grade_metrics = grade_episode(env, task_dir, settings=settings)
        except SandboxCommandTimeoutError as error:
            diagnostic = ""
            if error.result is not None:
                diagnostic = (
                    error.result.output_tail
                    if error.result.output_truncated
                    else error.result.output
                )[-_VERIFIER_LOG_TAIL_CHARS:]
            logger.warning(
                "Modal SWE verifier timed out for %s after %ss",
                task_dir.name,
                verifier_timeout,
            )
            elapsed = time.perf_counter() - started
            metrics = _environment_metrics(
                env,
                agent_queue_time=agent_queue_time,
                agent_dispatch_queue_time=dispatch_queue_time,
                total_time=elapsed,
                agent_start_snapshot=agent_start_snapshot,
                agent_snapshot=agent_snapshot,
            )
            metrics["verifier_timeout"] = 1
            metrics["context_limit_exceeded"] = int(context_limit_exceeded)
            metrics["generation_limit_exceeded"] = int(generation_limit_exceeded)
            _attach_client_model_timings(
                metrics,
                client_model_request_durations,
            )
            return _failure(
                "verifier_timeout",
                # The verifier executes the repository state produced by the
                # policy. Making tests exceed the benchmark deadline is a
                # policy outcome, not missing infrastructure; preserve the
                # trajectory and train it with zero reward.
                infrastructure=False,
                total_time=elapsed,
                agent_queue_time=agent_queue_time,
                verifier_timeout_sec=_verifier_timeout(
                    task_dir, int(settings["verify_timeout"])
                ),
                verifier_output_tail=diagnostic,
                agent_metrics=metrics,
                **_graded_outputs(None, getattr(error, "policy_patch", None)),
            )
        verify_time = time.perf_counter() - verify_started
        reward = verifier["reward"]
        if reward is None:
            logger.error(
                "Modal SWE verifier produced no reward for %s (rc=%s, timeout=%ss)",
                task_dir.name,
                verifier["return_code"],
                verifier["timeout_sec"],
            )
            elapsed = time.perf_counter() - started
            metrics = _environment_metrics(
                env,
                agent_queue_time=agent_queue_time,
                agent_dispatch_queue_time=dispatch_queue_time,
                total_time=elapsed,
                agent_start_snapshot=agent_start_snapshot,
                agent_snapshot=agent_snapshot,
            )
            metrics["verifier_return_code"] = verifier["return_code"]
            metrics["verifier_reward_missing"] = 1
            metrics.update(grade_metrics)
            metrics["context_limit_exceeded"] = int(context_limit_exceeded)
            metrics["generation_limit_exceeded"] = int(generation_limit_exceeded)
            _attach_client_model_timings(
                metrics,
                client_model_request_durations,
            )
            return _failure(
                "verifier_infra_error",
                total_time=elapsed,
                agent_queue_time=agent_queue_time,
                verifier_return_code=verifier["return_code"],
                verifier_timeout_sec=verifier["timeout_sec"],
                verifier_output_tail=verifier["output_tail"],
                agent_metrics=metrics,
                # The tail is already passed above; keep only the patch.
                **_graded_outputs(None, verifier.get("policy_patch")),
            )

        elapsed = time.perf_counter() - started
        agent_metrics = _environment_metrics(
            env,
            agent_queue_time=agent_queue_time,
            agent_dispatch_queue_time=dispatch_queue_time,
            total_time=elapsed,
            agent_start_snapshot=agent_start_snapshot,
            agent_snapshot=agent_snapshot,
        )
        agent_metrics["turns"] = agent.n_calls
        agent_metrics.update(_model_token_metrics(agent.messages))
        agent_metrics["eval_time"] = verify_time
        agent_metrics["context_limit_exceeded"] = int(context_limit_exceeded)
        agent_metrics["generation_limit_exceeded"] = int(generation_limit_exceeded)
        agent_metrics["verifier_return_code"] = verifier["return_code"]
        agent_metrics["verifier_reward_missing"] = 0
        agent_metrics["verifier_timeout"] = 0
        agent_metrics.update(verifier.get("grade_metrics", {}))
        agent_metrics.update(grade_metrics)
        reward, submission_metrics = submission_outcome(exit_status, reward)
        agent_metrics.update(submission_metrics)
        _attach_client_model_timings(
            agent_metrics,
            client_model_request_durations,
        )
        return {
            "reward": reward,
            "exit_status": exit_status,
            "eval_report": {
                "reward": reward,
                "verifier_return_code": verifier["return_code"],
            },
            "verifier_return_code": verifier["return_code"],
            "verifier_timeout_sec": verifier["timeout_sec"],
            "agent_metrics": agent_metrics,
            **_graded_outputs(verifier, verifier.get("policy_patch")),
        }
    except _EpisodeCancelled:
        return _failure(
            "rollout_cancelled",
            total_time=time.perf_counter() - started,
            agent_queue_time=agent_queue_time,
            failure_phase=current_phase,
        )
    except Exception as error:
        if cancelled is not None and cancelled.is_set():
            return _failure(
                "rollout_cancelled",
                total_time=time.perf_counter() - started,
                agent_queue_time=agent_queue_time,
                failure_phase=current_phase,
            )
        if not _is_infrastructure_error(error):
            raise
        logger.warning(
            "Modal SWE episode failed during %s for %s: %s: %s",
            current_phase,
            task_dir.name,
            type(error).__name__,
            str(error)[:500],
        )
        return _failure(
            "sandbox_infra_error",
            total_time=time.perf_counter() - started,
            agent_queue_time=agent_queue_time,
            failure_phase=current_phase,
            sandbox_error=f"{type(error).__name__}: {error}"[:1000],
            **_exception_metadata(error),
            **(env.lifecycle_diagnostics() if env is not None else {}),
        )
    finally:
        set_phase("cleanup")
        if env is not None:
            _stop_environment(env)


def _threads_per_agent_process() -> int:
    return int(
        os.getenv(
            "MODAL_SWE_AGENT_THREADS_PER_PROCESS",
            os.getenv("MODAL_SWE_AGENT_THREADS", "32"),
        )
    )


def _agent_process_count() -> int:
    return int(os.getenv("MODAL_SWE_AGENT_PROCESSES", "1"))


class _StartRamp:
    """Spreads the first wave of episodes over ``MODAL_SWE_START_RAMP_SECONDS``.

    A fresh rollout process starts every concurrent session at once, so all of
    them create sandboxes and prefill their prompts together. Episode ``i`` of the
    first wave (one per concurrent session) instead waits ``ramp * i / wave``;
    later episodes, which replace finished ones, start at once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = 0

    def next_delay(self) -> float:
        ramp = float(os.getenv("MODAL_SWE_START_RAMP_SECONDS", "0"))
        wave = _agent_process_count() * _threads_per_agent_process()
        with self._lock:
            index = self._started
            self._started += 1
        if ramp <= 0 or index >= wave:
            return 0.0
        return ramp * index / wave


_start_ramp = _StartRamp()


@cache
def _local_agent_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=_threads_per_agent_process(),
        thread_name_prefix="modal-swe",
    )


class _AgentWorker:
    """One Ray process with its own Modal SDK event loop and bounded thread fan-out."""

    def __init__(self, worker_index: int, threads: int) -> None:
        self.worker_index = worker_index
        self.threads = threads
        self.executor = ThreadPoolExecutor(
            max_workers=threads,
            thread_name_prefix=f"modal-swe-{worker_index}",
        )
        self._phase_lock = threading.Lock()
        self._episode_phases: dict[str, tuple[str, float, float]] = {}
        self._abort_state = _EpisodeAbortState()
        self._abort_generation = 0

    async def ping(self) -> dict[str, Any]:
        """Prove the controller actor imported and started before rollout."""
        return {
            "worker_index": self.worker_index,
            "threads": self.threads,
            "pid": os.getpid(),
        }

    async def stats(self) -> dict[str, Any]:
        """Return a cheap live phase snapshot while executor threads are busy."""
        now = time.monotonic()
        with self._phase_lock:
            entries = list(self._episode_phases.values())
        phase_counts = Counter(phase for phase, _, _ in entries)
        oldest_phase_seconds: dict[str, float] = {}
        for phase, phase_started, _ in entries:
            oldest_phase_seconds[phase] = max(
                oldest_phase_seconds.get(phase, 0.0),
                now - phase_started,
            )
        return {
            "worker_index": self.worker_index,
            "active": len(entries),
            "phase_counts": dict(phase_counts),
            "oldest_phase_seconds": oldest_phase_seconds,
            "oldest_episode_seconds": max(
                (now - episode_started for _, _, episode_started in entries),
                default=0.0,
            ),
        }

    async def run_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        generation = int(payload.pop("_abort_generation", 0))
        if generation != self._abort_generation:
            return _failure("rollout_cancelled")
        dispatch_queue_time = max(
            0.0,
            time.time() - float(payload.pop("submitted_at_unix")),
        )
        queued_at = time.perf_counter()
        episode_id, cancelled = self._abort_state.start()
        episode_started = time.monotonic()
        current_phase = "executor_queue"
        phase_started = episode_started
        phase_durations: Counter[str] = Counter()

        def set_phase(phase: str) -> None:
            nonlocal current_phase, phase_started
            now = time.monotonic()
            phase_durations[current_phase] += max(0.0, now - phase_started)
            current_phase = phase
            phase_started = now
            with self._phase_lock:
                self._episode_phases[episode_id] = (
                    phase,
                    phase_started,
                    episode_started,
                )

        with self._phase_lock:
            self._episode_phases[episode_id] = (
                current_phase,
                phase_started,
                episode_started,
            )
        result = None
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                partial(
                    _run_episode_sync,
                    **payload,
                    queued_at=queued_at,
                    dispatch_queue_time=dispatch_queue_time,
                    phase_callback=set_phase,
                    cancelled=cancelled,
                    sandbox_started=partial(
                        self._abort_state.attach_sandbox,
                        episode_id,
                    ),
                ),
            )
        except Exception as error:
            if not _is_infrastructure_error(error):
                raise RuntimeError(
                    f"Modal SWE episode failed during {current_phase}: "
                    f"{type(error).__name__}: {error}"
                ) from None
            logger.warning(
                "Modal SWE worker episode failed during %s: %s: %s",
                current_phase,
                type(error).__name__,
                str(error)[:500],
            )
            result = _failure(
                "sandbox_infra_error",
                failure_phase=current_phase,
                sandbox_error=f"{type(error).__name__}: {error}"[:1000],
                **_exception_metadata(error),
            )
        finally:
            now = time.monotonic()
            phase_durations[current_phase] += max(0.0, now - phase_started)
            with self._phase_lock:
                self._episode_phases.pop(episode_id, None)
            self._abort_state.finish(episode_id)
        if isinstance(result, dict):
            metrics = result.setdefault("agent_metrics", {})
            metrics["agent_worker_index"] = self.worker_index
            for phase, duration in phase_durations.items():
                metrics[f"phase_{phase}_seconds"] = duration
            metrics["phase_accounted_seconds"] = sum(phase_durations.values())
        return result

    async def abort_episodes(self, generation: int) -> None:
        if generation <= self._abort_generation:
            return
        self._abort_generation = generation
        await _terminate_sandboxes(self._abort_state.cancel_all())


class _RayAgentWorkerPool:
    """Load-balanced handles for independent Modal-controller processes."""

    def __init__(self, workers: list[Any], per_worker_capacity: int = 1) -> None:
        self.workers = workers
        self.in_flight = [0] * len(workers)
        self.next_tie_break = 0
        self.progress_reporter: asyncio.Task | None = None
        self.capacity = len(workers) * per_worker_capacity
        self._available = asyncio.Semaphore(self.capacity)
        self.generation = 0

    def _acquire(self) -> tuple[int, Any]:
        minimum = min(self.in_flight)
        for offset in range(len(self.workers)):
            index = (self.next_tie_break + offset) % len(self.workers)
            if self.in_flight[index] == minimum:
                self.in_flight[index] += 1
                self.next_tie_break = (index + 1) % len(self.workers)
                return index, self.workers[index]
        raise AssertionError("Agent worker pool is unexpectedly empty")

    def _release(self, index: int) -> None:
        self.in_flight[index] -= 1
        assert self.in_flight[index] >= 0

    async def run_episode(
        self,
        payload: dict[str, Any],
        *,
        generation: int | None = None,
    ) -> dict[str, Any]:
        generation = self.generation if generation is None else generation
        # Keep excess episode coroutines in this process. Submitting them to
        # Ray would put them ahead of lightweight stats RPCs in each actor's
        # mailbox and eventually exhaust max_pending_calls.
        await self._available.acquire()
        if generation != self.generation:
            self._available.release()
            return _failure("rollout_cancelled")
        index, worker = self._acquire()
        try:
            future = asyncio.ensure_future(
                worker.run_episode.remote({**payload, "_abort_generation": generation})
            )
        except Exception:
            self._release(index)
            self._available.release()
            raise

        def release_when_finished(completed: asyncio.Future) -> None:
            # Retrieve failures even when the caller was cancelled so asyncio
            # does not report an unobserved task exception.
            if not completed.cancelled():
                completed.exception()
            self._release(index)
            self._available.release()

        future.add_done_callback(release_when_finished)
        try:
            # Cancelling an asyncio waiter cannot stop the executor thread in
            # the Ray actor. Keep the remote call alive and retain its capacity
            # accounting until the episode actually exits.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            raise

    async def abort(self) -> None:
        self.generation += 1
        await asyncio.gather(
            *(worker.abort_episodes.remote(self.generation) for worker in self.workers)
        )

    def ensure_progress_reporter(self) -> None:
        if self.progress_reporter is None or self.progress_reporter.done():
            self.progress_reporter = asyncio.create_task(self._report_progress())

    async def _report_progress(self) -> None:
        """Log one aggregate phase heartbeat for the complete controller pool."""
        while True:
            await asyncio.sleep(30)
            stats_tasks = [
                asyncio.ensure_future(worker.stats.remote()) for worker in self.workers
            ]
            done, pending = await asyncio.wait(stats_tasks, timeout=5)
            for task in pending:
                task.cancel()
            snapshots = []
            for task in done:
                try:
                    snapshots.append(task.result())
                except Exception as error:
                    logger.warning(
                        "One Modal agent-pool stats shard failed: %s",
                        error,
                    )
            if not snapshots:
                logger.warning(
                    "Modal agent-pool phase metrics unavailable from all %d workers",
                    len(self.workers),
                )
                continue

            phases: Counter[str] = Counter()
            oldest_by_phase: dict[str, float] = {}
            for snapshot in snapshots:
                phases.update(snapshot["phase_counts"])
                for phase, seconds in snapshot["oldest_phase_seconds"].items():
                    oldest_by_phase[phase] = max(
                        oldest_by_phase.get(phase, 0.0),
                        float(seconds),
                    )
            dispatched = sum(self.in_flight)
            logger.info(
                "Modal agent-pool progress: responsive_workers=%d/%d dispatched=%d "
                "active=%d capacity=%d phases=%s oldest_phase_seconds=%s "
                "oldest_episode=%.1fs",
                len(snapshots),
                len(self.workers),
                dispatched,
                sum(phases.values()),
                self.capacity,
                dict(sorted(phases.items())),
                {
                    phase: round(seconds, 1)
                    for phase, seconds in sorted(oldest_by_phase.items())
                },
                max(
                    (
                        float(snapshot["oldest_episode_seconds"])
                        for snapshot in snapshots
                    ),
                    default=0.0,
                ),
            )


@cache
def _ray_agent_pool() -> _RayAgentWorkerPool:
    import ray

    process_count = _agent_process_count()
    threads = _threads_per_agent_process()
    if process_count <= 1:
        raise ValueError("Ray agent pool requires more than one process")
    remote_worker = ray.remote(_AgentWorker)
    workers = [
        remote_worker.options(
            num_cpus=1,
            # Reserve actor concurrency for ping/stats while all executor
            # slots are occupied by episodes. This does not increase
            # episode concurrency; the thread pool remains the hard cap.
            max_concurrency=threads + 2,
            max_pending_calls=(threads + 2) * 2,
            scheduling_strategy="SPREAD",
        ).remote(index, threads)
        for index in range(process_count)
    ]
    # Actor construction is asynchronous. Resolve one cheap call on every
    # worker so an import/scheduling failure is surfaced once at pool startup
    # rather than converting early trajectories into infrastructure failures.
    ready = ray.get([worker.ping.remote() for worker in workers])
    pool = _RayAgentWorkerPool(workers, per_worker_capacity=threads)
    logger.info(
        "Started %s Modal agent-controller processes with %s threads each (pids=%s)",
        process_count,
        threads,
        [item["pid"] for item in ready],
    )
    return pool


@cache
def _local_abort_state() -> _EpisodeAbortState:
    return _EpisodeAbortState()


async def abort(_args: Any) -> None:
    """Stop every episode owned by this agent integration."""
    if _agent_process_count() > 1:
        await _ray_agent_pool().abort()
        return
    await _terminate_sandboxes(_local_abort_state().cancel_all())


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Run one repository-repair episode without blocking the rollout event loop."""
    del kwargs
    if (delay := _start_ramp.next_delay()) > 0:
        await asyncio.sleep(delay)
    queued_at = time.perf_counter()
    payload = {
        "base_url": base_url,
        "prompt": prompt,
        "request_kwargs": request_kwargs or {},
        "metadata": metadata or {},
        "submitted_at_unix": time.time(),
    }
    agent_pool = None
    local_episode = None
    if _agent_process_count() > 1:
        agent_pool = _ray_agent_pool()
        generation = agent_pool.generation
    else:
        local_state = _local_abort_state()
        episode_id, cancelled = local_state.start()
        local_episode = (local_state, episode_id)
    try:
        await ensure_sandbox_app(str(sandbox_settings()["app_name"]))
        if agent_pool is not None:
            agent_pool.ensure_progress_reporter()
            episode = agent_pool.run_episode(payload, generation=generation)
        else:
            payload.pop("submitted_at_unix")
            episode = asyncio.get_running_loop().run_in_executor(
                _local_agent_executor(),
                partial(
                    _run_episode_sync,
                    **payload,
                    queued_at=queued_at,
                    cancelled=cancelled,
                    sandbox_started=partial(
                        local_state.attach_sandbox,
                        episode_id,
                    ),
                ),
            )
        # The mini-swe-agent wall limit, per-command/verifier deadlines, and
        # Modal Sandbox lifetime already bound every blocking phase. A second
        # asyncio timeout cannot stop the worker thread and would release pool
        # accounting while its sandbox kept running.
        return await episode
    except Exception as error:
        if not _is_infrastructure_error(error):
            raise
        logger.warning(
            "Modal SWE episode failed: %s: %s",
            type(error).__name__,
            str(error)[:500],
        )
        return _failure(
            "sandbox_infra_error",
            total_time=time.perf_counter() - queued_at,
            sandbox_error=f"{type(error).__name__}: {error}"[:1000],
            **_exception_metadata(error),
        )
    finally:
        if local_episode is not None:
            local_state, episode_id = local_episode
            local_state.finish(episode_id)
