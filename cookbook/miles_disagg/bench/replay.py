"""Closed-loop replay of recorded agent trajectories against OpenAI-compatible SGLang
engines.

Each session replays one trajectory at a time with zero think time: it sends a turn's
request, reads the response, and sends the next turn at once. Turn ``k`` carries the
conversation so far as the agent would build it: the recorded system, user and tool
messages, plus the assistant outputs this replay generated, so each engine's prefix
cache sees what it would see under the agent. Every request decodes exactly the
recorded number of tokens (``max_tokens`` = the recorded completion length, with
``ignore_eos``), whatever the sampled text.

A new session starts at a random turn of a random trajectory, its context so far taken
as recorded, so long contexts are present from the first minute; once a trajectory
ends, the session moves on to another from its first turn. Mid-trajectory starts share
recorded prefixes that no two real episodes share, so each trajectory run tags its
first user message with a unique line: sessions share the system prompt and tools, as
real episodes do, and nothing after them.

With ``TokenPrompts``, each request also carries its prompt as token IDs, built the way
the training session server builds them (token in, token out): the conversation is
rendered and tokenized once when a session starts a trajectory, and each later turn adds
the returned completion IDs and the newly appended messages, tokenized on their own. The
engine then never re-renders or re-tokenizes the conversation, which at agent context
lengths would cap it at a few requests per second on its tokenizer process alone.

Responses stream by default, which times each request's first token and decode. With
``ReplayConfig.stream`` off, each response arrives whole, as the training agent's do,
which costs the client one parse per request instead of one per token: a single replay
process can then load many fast engines without becoming the bottleneck.

Statistics accrue to a measurement ``Window`` while one is open: output tokens as they
stream in (or as each unstreamed response arrives), and every other per-request
statistic over the requests that finish inside the window. Pure Python plus httpx;
nothing here imports Modal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

try:  # A request carries its whole context's token IDs: encode them fast when possible.
    import orjson

    def _dumps(value: Any) -> bytes:
        return orjson.dumps(value)

    _loads = orjson.loads
except ImportError:  # pragma: no cover - the driver image installs orjson

    def _dumps(value: Any) -> bytes:
        return json.dumps(value).encode()

    _loads = json.loads

logger = logging.getLogger(__name__)

# Written by cookbook.miles_disagg.modal_swe.agent (MODAL_SWE_TRAJECTORY_DUMP_DIR).
TRAJECTORY_FORMAT = "modal-swe-trajectory/v1"
API_ROLES = frozenset({"system", "user", "assistant", "tool"})
# Modal Flash keeps a session on one replica when the gateway routes requests.
DEFAULT_AFFINITY_HEADER = "Modal-Session-ID"
# How long a stopping session may take to unwind. A cancelled httpx stream can sit in
# a shielded socket read that swallows the cancellation (httpcore's close runs under
# an anyio shield), so a stop never waits on it unbounded; closing the client ends it.
STOP_TIMEOUT_S = 5.0


# ── Trajectories ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelCall:
    """One model request of a recorded episode. ``before`` holds the messages the agent
    added since the previous request. ``output`` is the recorded assistant message, or
    ``None`` when the agent dropped the response (a format error) so that it never
    entered the context."""

    before: tuple[dict[str, Any], ...]
    output: dict[str, Any] | None
    completion_tokens: int
    prompt_tokens: int | None = None


@dataclass(frozen=True)
class Trajectory:
    name: str
    calls: tuple[ModelCall, ...]
    tools: tuple[dict[str, Any], ...] | None = None

    def prefix(self, start: int) -> list[dict[str, Any]]:
        """Call ``start``'s messages as recorded: every earlier call's context and
        recorded output, then its own context."""
        messages: list[dict[str, Any]] = []
        for call in self.calls[:start]:
            messages.extend(call.before)
            if call.output is not None:
                messages.append(call.output)
        messages.extend(self.calls[start].before)
        return [dict(message) for message in messages]


def valid_tool_calls(calls: Iterable[Any]) -> list[dict[str, Any]]:
    """Tool calls the server can render back into a prompt: a function name and JSON
    object arguments. A call cut off by ``max_tokens`` has neither and is dropped."""
    valid = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name, arguments = function.get("name"), function.get("arguments")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        else:
            parsed = arguments
        if not isinstance(parsed, dict):
            continue
        valid.append(
            {
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(parsed)},
            }
        )
    return valid


def api_message(message: dict[str, Any]) -> dict[str, Any]:
    """A dumped message as an OpenAI chat message."""
    role = message["role"]
    out: dict[str, Any] = {"role": role, "content": message.get("content")}
    if role == "assistant":
        if out["content"] is None:
            out["content"] = ""
        if message.get("reasoning_content"):
            out["reasoning_content"] = message["reasoning_content"]
        if tool_calls := valid_tool_calls(message.get("tool_calls") or ()):
            out["tool_calls"] = tool_calls
    if role == "tool" and message.get("tool_call_id"):
        out["tool_call_id"] = message["tool_call_id"]
    return out


def parse_trajectory(record: dict[str, Any], name: str) -> Trajectory:
    """Split a dumped episode into its model calls. A message carrying a response's
    token counts is a call; an assistant one is its output, and any other (mini-swe-
    agent's format-error message) follows a call whose output was dropped."""
    if record.get("format") != TRAJECTORY_FORMAT:
        raise ValueError(f"{name}: not a {TRAJECTORY_FORMAT} trajectory")
    calls: list[ModelCall] = []
    pending: list[dict[str, Any]] = []
    for message in record.get("messages") or ():
        role = message.get("role")
        if role not in API_ROLES:
            continue
        tokens = message.get("completion_tokens")
        is_call = (
            bool(message.get("model_call"))
            and isinstance(tokens, int)
            and not isinstance(tokens, bool)
        )
        if not is_call:
            pending.append(api_message(message))
            continue
        prompt = message.get("prompt_tokens")
        prompt = prompt if isinstance(prompt, int) else None
        if role == "assistant":
            calls.append(
                ModelCall(tuple(pending), api_message(message), max(1, tokens), prompt)
            )
            pending = []
        else:
            calls.append(ModelCall(tuple(pending), None, max(1, tokens), prompt))
            pending = [api_message(message)]
    tools = record.get("tools")
    return Trajectory(
        name=name,
        calls=tuple(calls),
        tools=tuple(tools) if isinstance(tools, list) and tools else None,
    )


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Every dumped trajectory under ``path`` (a file or a directory, recursively)
    with at least one model call, ordered by file name."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.json"))
    trajectories = []
    for file in files:
        try:
            trajectory = parse_trajectory(json.loads(file.read_text()), file.stem)
        except (ValueError, json.JSONDecodeError) as error:
            logger.warning("skipping %s: %s", file, error)
            continue
        if trajectory.calls:
            trajectories.append(trajectory)
    return trajectories


def tag_first_user_message(messages: list[dict[str, Any]], tag: str) -> None:
    """Prefix the first user message with ``tag`` on its own line, in place."""
    for index, message in enumerate(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            messages[index] = {**message, "content": f"{tag}\n{message['content']}"}
            return


# ── Statistics ───────────────────────────────────────────────────────────────────────


def template_message(message: dict[str, Any]) -> dict[str, Any]:
    """``message`` as a chat template renders it: tool-call arguments as objects."""
    calls = message.get("tool_calls")
    if not calls:
        return message
    rendered = []
    for call in calls:
        function = dict(call.get("function") or {})
        if isinstance(function.get("arguments"), str):
            try:
                function["arguments"] = json.loads(function["arguments"])
            except json.JSONDecodeError:
                pass
        rendered.append({**call, "function": function})
    return {**message, "tool_calls": rendered}


class TokenPrompts:
    """Prompt token IDs as the training session server builds them (Miles' TITO).

    A trajectory's context is rendered and tokenized once (``render``). A request's
    prompt is that context plus ``suffix(appended)``: the messages appended since,
    rendered after a placeholder turn and tokenized on their own, then the generation
    prompt. A kept response extends the context by its completion IDs and the
    template's end of turn (``close``); a dropped one leaves it as it was.

    ``tokenizer`` needs ``apply_chat_template(..., tokenize=False)`` and ``encode``, as
    a Hugging Face tokenizer has."""

    _PLACEHOLDER = "\u2063placeholder\u2063"

    def __init__(self, tokenizer: Any, template_kwargs: dict[str, Any] | None = None):
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})
        # A minimal conversation to append after (Qwen3.6's template needs a user turn).
        self._base = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": self._PLACEHOLDER},
        ]
        self._base_text = self._render(self._base, generation=False)
        tail = self._base_text[
            self._base_text.rindex(self._PLACEHOLDER) + len(self._PLACEHOLDER) :
        ]
        self.end_of_turn = self._encode(tail)

    def _render(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        generation: bool,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> str:
        return self.tokenizer.apply_chat_template(
            [template_message(message) for message in messages],
            tools=list(tools) if tools else None,
            add_generation_prompt=generation,
            tokenize=False,
            **self.template_kwargs,
        )

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def render(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> list[int]:
        """The whole conversation, without the generation prompt."""
        return self._encode(self._render(messages, generation=False, tools=tools))

    def suffix(self, appended: Sequence[dict[str, Any]]) -> list[int]:
        """``appended`` as they follow an assistant turn, then the generation prompt."""
        text = self._render([*self._base, *appended], generation=True)
        if not text.startswith(self._base_text):
            raise ValueError("the chat template re-renders earlier turns")
        return self._encode(text[len(self._base_text) :])

    def close(self, completion: Sequence[int]) -> list[int]:
        """A completion with the template's end of turn, which a response capped by
        ``max_tokens`` lacks and a stopped one carries only in part."""
        tail = self.end_of_turn
        if completion and tail and completion[-1] == tail[0]:
            return [*completion, *tail[1:]]
        return [*completion, *tail]


def percentile(values: Sequence[float], q: float) -> float | None:
    """The ``q``-th percentile (0..100), linearly interpolated, or None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


class Window:
    """What one measurement window saw. Output tokens count as they arrive; every
    other statistic covers the requests that finished inside the window."""

    def __init__(self, started: float, sessions: int) -> None:
        self.started = started
        self.ended: float | None = None
        self.sessions = sessions
        self.output_tokens = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completed = 0
        self.ttft: list[float] = []
        self.latency: list[float] = []
        self.decode_speed: list[float] = []
        self.e2e_speed: list[float] = []
        self.prompt_lengths: list[int] = []
        # Over requests whose recorded prompt length is known: what the engine saw, and
        # what the training run recorded.
        self.served_prompt_tokens = 0
        self.recorded_prompt_tokens = 0
        self.completion_lengths: list[int] = []
        self.errors: Counter[str] = Counter()
        self.in_flight: list[int] = []
        self.loop_lag: list[float] = []
        self.server: dict[str, float] = {}

    def summary(self) -> dict[str, Any]:
        if self.ended is None:
            raise RuntimeError("the window is still open")
        duration = max(self.ended - self.started, 1e-9)
        return {
            "window_s": duration,
            "sessions": self.sessions,
            "output_tokens": self.output_tokens,
            "output_tok_s": self.output_tokens / duration,
            "prompt_tokens": self.prompt_tokens,
            "prompt_tok_s": self.prompt_tokens / duration,
            "uncached_prompt_tok_s": (self.prompt_tokens - self.cached_tokens)
            / duration,
            "requests": self.completed,
            "requests_s": self.completed / duration,
            "decode_tok_s_p10": percentile(self.decode_speed, 10),
            "decode_tok_s_p50": percentile(self.decode_speed, 50),
            "decode_tok_s_p90": percentile(self.decode_speed, 90),
            "decode_samples": len(self.decode_speed),
            "e2e_tok_s_p50": percentile(self.e2e_speed, 50),
            "ttft_s_p50": percentile(self.ttft, 50),
            "ttft_s_p90": percentile(self.ttft, 90),
            "latency_s_p50": percentile(self.latency, 50),
            "latency_s_p90": percentile(self.latency, 90),
            "prompt_tokens_p50": percentile(self.prompt_lengths, 50),
            "prompt_vs_recorded": (
                self.served_prompt_tokens / self.recorded_prompt_tokens
                if self.recorded_prompt_tokens
                else None
            ),
            "prompt_tokens_p90": percentile(self.prompt_lengths, 90),
            "completion_tokens_p50": percentile(self.completion_lengths, 50),
            "errors": sum(self.errors.values()),
            "error_kinds": dict(self.errors),
            "in_flight_mean": _mean(self.in_flight),
            "loop_lag_s_p90": percentile(self.loop_lag, 90),
            "loop_lag_s_max": max(self.loop_lag, default=0.0),
            "server": dict(self.server),
        }


def parse_prometheus(text: str) -> dict[str, float]:
    """Prometheus text exposition as {metric name: value}. Label sets of one name
    collapse to their maximum: with ``--enable-metrics-for-all-schedulers`` every TP
    rank reports the same batch, so a sum would double it. Histogram buckets are
    dropped; their ``_sum`` and ``_count`` stay."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, _, rest = line.partition("{")
            rest = rest.rpartition("}")[2]
        else:
            name, _, rest = line.partition(" ")
        if name.endswith("_bucket") or name.endswith("_created"):
            continue
        try:
            value = float(rest.split()[0])
        except (IndexError, ValueError):
            continue
        if math.isfinite(value):
            values[name] = max(value, values.get(name, value))
    return values


def _is_counter(name: str) -> bool:
    return name.endswith(("_total", "_sum", "_count"))


def server_metrics(
    before: Sequence[tuple[float, dict[str, float] | None]],
    after: Sequence[tuple[float, dict[str, float] | None]],
    samples: Sequence[Sequence[tuple[float, dict[str, float] | None]]],
) -> dict[str, float]:
    """Engine metrics over a window, from per-engine scrapes ``(time, values)``:
    counters as per-second rates between the scrapes around the window, summed over
    engines (``rate:<name>``); gauges as their mean over the window's scrapes,
    averaged over engines (``mean:<name>``). Engines without metrics add nothing."""
    rates: dict[str, float] = {}
    gauges: dict[str, list[float]] = {}
    for engine, ((t0, first), (t1, last)) in enumerate(zip(before, after, strict=True)):
        if first is None or last is None or t1 <= t0:
            continue
        for name, value in last.items():
            if _is_counter(name) and name in first:
                rate = (value - first[name]) / (t1 - t0)
                rates[f"rate:{name}"] = rates.get(f"rate:{name}", 0.0) + rate
        engine_gauges: dict[str, list[float]] = {}
        for scrape in [*(sample[engine] for sample in samples), (t1, last)]:
            values = scrape[1]
            for name, value in (values or {}).items():
                if not _is_counter(name):
                    engine_gauges.setdefault(name, []).append(value)
        for name, observed in engine_gauges.items():
            gauges.setdefault(name, []).append(sum(observed) / len(observed))
    return {
        **rates,
        **{f"mean:{name}": sum(v) / len(v) for name, v in gauges.items()},
    }


# ── Replay ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Target:
    """One engine to load: its base URL and any headers that address it (a Modal
    Flash replica is addressed through its pool's URL by ``modal-flash-upstream``)."""

    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    name: str = ""


@dataclass(frozen=True)
class ReplayConfig:
    temperature: float = 1.0
    top_p: float = 0.97
    top_k: int = 64
    # Rollout requests return every token's log-probability; so do these. Score-centering
    # rollouts also return each position's top candidates (Miles: score_centering_top_k).
    logprobs: bool = True
    top_logprobs: int = 0
    # Stream each response (times its first token and decode) or read it whole, as the
    # training agent does (latency only, and far less work for this client).
    stream: bool = True
    # The served model name; by default each target's first /v1/models entry.
    model: str | None = None
    seed: int = 0
    random_start: bool = True
    unique_sessions: bool = True
    max_tokens_cap: int | None = None
    # Shorter responses are too short to time their decode.
    min_decode_tokens: int = 16
    connect_timeout_s: float = 30.0
    # Longest silence while streaming, which covers a long wait for prefill.
    read_timeout_s: float = 900.0
    error_backoff_s: float = 1.0
    affinity_header: str | None = DEFAULT_AFFINITY_HEADER
    metrics_path: str | None = "/metrics"
    metrics_interval_s: float = 15.0
    tick_s: float = 1.0


@dataclass
class _Session:
    index: int
    target: int
    rng: random.Random
    first_done: asyncio.Event = field(default_factory=asyncio.Event)
    runs: int = 0
    task: asyncio.Task | None = None
    stopped: bool = False


@dataclass
class _Stream:
    """One response as it arrives: streamed chunk by chunk, or whole."""

    sent: float
    streamed: bool = True
    recorded_prompt_tokens: int | None = None
    completion_ids: list[int] | None = None
    tokens: int = 0
    tokens_at_first: int = 0
    first_token: float | None = None
    last_token: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int = 0
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)

    def merge_tool_call(self, delta: dict[str, Any]) -> None:
        index = delta.get("index", 0)
        call = self.tool_calls.setdefault(
            index if isinstance(index, int) else 0,
            {"id": None, "function": {"name": "", "arguments": ""}},
        )
        if delta.get("id"):
            call["id"] = delta["id"]
        function = delta.get("function") or {}
        if function.get("name") and not call["function"]["name"]:
            call["function"]["name"] = function["name"]
        if isinstance(function.get("arguments"), str):
            call["function"]["arguments"] += function["arguments"]

    def assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self.content),
        }
        if self.reasoning:
            message["reasoning_content"] = "".join(self.reasoning)
        calls = valid_tool_calls(self.tool_calls[i] for i in sorted(self.tool_calls))
        if calls:
            message["tool_calls"] = calls
        return message


class Replay:
    """Sessions replaying ``trajectories`` against ``targets``; session ``i`` always
    loads ``targets[i % len(targets)]``. Use as an async context manager, then
    ``resize``, ``warmup`` and ``measure``."""

    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        targets: Sequence[Target],
        config: ReplayConfig | None = None,
        *,
        prompts: TokenPrompts | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not trajectories:
            raise ValueError("no trajectories to replay")
        if not targets:
            raise ValueError("no targets to load")
        self.trajectories = list(trajectories)
        self.targets = list(targets)
        self.config = config or ReplayConfig()
        if prompts is not None and self.config.stream:
            raise ValueError("token prompts read each response whole: set stream=False")
        self.prompts = prompts
        self._clock = clock
        self._client: httpx.AsyncClient | None = None
        self._models: list[str] = []
        self._sessions: list[_Session] = []
        self._window: Window | None = None
        self._in_flight = 0
        self._stragglers: set[asyncio.Task] = set()
        self.requests_sent = 0
        self.errors: Counter[str] = Counter()

    async def __aenter__(self) -> Replay:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    @property
    def sessions(self) -> int:
        return len(self._sessions)

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout_s,
                read=self.config.read_timeout_s,
                write=60.0,
                pool=None,
            ),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            trust_env=False,
        )
        self._models = [
            self.config.model or await self._served_model(target)
            for target in self.targets
        ]

    async def retarget(self, targets: Sequence[Target]) -> None:
        """Load ``targets`` instead, one for one (a replica's replacement after a
        preemption has a new address). Call with no sessions running."""
        if self._sessions:
            raise RuntimeError("stop every session before retargeting")
        if len(targets) != len(self.targets):
            raise ValueError(f"need {len(self.targets)} targets, got {len(targets)}")
        self.targets = list(targets)
        self._models = [
            self.config.model or await self._served_model(target)
            for target in self.targets
        ]

    async def close(self) -> None:
        await self.resize(0)
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        # Closing the client closes every connection, which ends any stuck read.
        stragglers, self._stragglers = self._stragglers, set()
        for task in stragglers:
            task.cancel()
        await self._settle(stragglers, keep=False)

    async def _settle(
        self, tasks: Iterable[asyncio.Task], *, keep: bool = True
    ) -> None:
        """Wait up to ``STOP_TIMEOUT_S`` for stopping tasks; keep any still running
        for ``close`` to end."""
        tasks = [task for task in tasks if not task.done()]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=STOP_TIMEOUT_S)
        if pending:
            logger.warning("%d stopped sessions are still unwinding", len(pending))
            if keep:
                self._stragglers.update(pending)

    async def _served_model(self, target: Target) -> str:
        assert self._client is not None
        try:
            response = await self._client.get(
                f"{target.base_url}/v1/models", headers=target.headers, timeout=30.0
            )
            return str(response.json()["data"][0]["id"])
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
            logger.warning(
                "no model list at %s; sending model=default", target.base_url
            )
            return "default"

    async def resize(self, sessions: int) -> None:
        """Run ``sessions`` sessions: new ones start mid-trajectory; surplus ones stop,
        newest first, abandoning their in-flight requests."""
        if sessions < 0:
            raise ValueError(f"sessions must be >= 0, got {sessions}")
        stopping = []
        while len(self._sessions) > sessions:
            session = self._sessions.pop()
            session.stopped = True
            if session.task is not None:
                session.task.cancel()
                stopping.append(session.task)
        await self._settle(stopping)
        while len(self._sessions) < sessions:
            index = len(self._sessions)
            session = _Session(
                index=index,
                target=index % len(self.targets),
                rng=random.Random(f"{self.config.seed}:{index}"),
            )
            session.task = asyncio.create_task(self._run_session(session))
            self._sessions.append(session)

    async def warmup(self, min_s: float, max_s: float) -> float:
        """Wait at least ``min_s``, then until every session has finished its first
        request (its long first prefill), for at most ``max_s`` in all. Returns the
        seconds waited."""
        started = self._clock()
        await asyncio.sleep(min_s)
        waiting = [
            s.first_done.wait() for s in self._sessions if not s.first_done.is_set()
        ]
        remaining = max_s - (self._clock() - started)
        if waiting and remaining > 0:
            try:
                await asyncio.wait_for(asyncio.gather(*waiting), timeout=remaining)
            except TimeoutError:
                logger.warning("warm-up ended before every session finished a request")
        return self._clock() - started

    async def measure(self, window_s: float) -> dict[str, Any]:
        """Open a window for ``window_s`` seconds and summarize it, with the engines'
        metrics around it when they export them."""
        before = await self._scrape_all()
        window = Window(self._clock(), len(self._sessions))
        samples: list[list[tuple[float, dict[str, float] | None]]] = []
        self._window = window
        background = [
            asyncio.create_task(self._tick(window)),
            asyncio.create_task(self._sample_metrics(samples)),
        ]
        try:
            await asyncio.sleep(window_s)
        finally:
            self._window = None
            window.ended = self._clock()
            for task in background:
                task.cancel()
            # A scrape cancelled mid-request may unwind slowly; never wait on it long.
            await asyncio.wait(background, timeout=STOP_TIMEOUT_S)
        after = await self._scrape_all()
        window.server = server_metrics(before, after, samples)
        return window.summary()

    # ── sessions ──

    async def _run_session(self, session: _Session) -> None:
        first = True
        while not session.stopped:
            trajectory = session.rng.choice(self.trajectories)
            start = 0
            if first and self.config.random_start:
                start = session.rng.randrange(len(trajectory.calls))
            first = False
            session.runs += 1
            run_id = f"bench-{self.config.seed}-{session.index}-{session.runs}"
            messages = trajectory.prefix(start)
            if self.config.unique_sessions:
                tag_first_user_message(messages, f"[replay {run_id}]")
            # Token prompts: the context's IDs through the last kept response, and the
            # messages appended since (a dropped response's context stays pending).
            context: list[int] = []
            pending: list[dict[str, Any]] = []
            if self.prompts is not None:
                context = await asyncio.to_thread(
                    self.prompts.render, messages, trajectory.tools
                )
            for position in range(start, len(trajectory.calls)):
                call = trajectory.calls[position]
                if position > start:
                    appended = [dict(message) for message in call.before]
                    messages.extend(appended)
                    pending.extend(appended)
                if session.stopped:
                    return
                input_ids = (
                    context + self.prompts.suffix(pending)
                    if self.prompts is not None
                    else None
                )
                result = await self._request(
                    session, run_id, trajectory, call, messages, input_ids
                )
                session.first_done.set()
                if session.stopped:
                    return
                if result is None:
                    await asyncio.sleep(self.config.error_backoff_s)
                    break
                output, completion_ids = result
                if call.output is not None:
                    messages.append(output)
                    if input_ids is not None:
                        context = input_ids + self.prompts.close(completion_ids or [])
                        pending = []

    def request_body(
        self,
        trajectory: Trajectory,
        call: ModelCall,
        messages: list[dict[str, Any]],
        model: str,
        input_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        max_tokens = call.completion_tokens
        if self.config.max_tokens_cap is not None:
            max_tokens = min(max_tokens, self.config.max_tokens_cap)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "stream": self.config.stream,
        }
        if self.config.stream:
            body["stream_options"] = {
                "include_usage": True,
                "continuous_usage_stats": True,
            }
        if trajectory.tools:
            body["tools"] = list(trajectory.tools)
        if self.config.logprobs or input_ids is not None:
            body["logprobs"] = True
        if self.config.top_logprobs:
            body["top_logprobs"] = self.config.top_logprobs
        if input_ids is not None:
            # As Miles' session server sends a turn: the messages, their token IDs (the
            # engine skips its own rendering), and the completion IDs in meta_info.
            body["input_ids"] = input_ids
            body["return_meta_info"] = True
            body["no_stop_trim"] = False
        return body

    async def _request(
        self,
        session: _Session,
        run_id: str,
        trajectory: Trajectory,
        call: ModelCall,
        messages: list[dict[str, Any]],
        input_ids: list[int] | None = None,
    ) -> tuple[dict[str, Any], list[int] | None] | None:
        """Send one turn and read its response. Returns the generated assistant
        message and, for a token prompt, its completion IDs; None after an error."""
        if self._client is None or session.stopped:
            return None
        target = self.targets[session.target]
        body = self.request_body(
            trajectory, call, messages, self._models[session.target], input_ids
        )
        headers = dict(target.headers)
        if self.config.affinity_header:
            headers[self.config.affinity_header] = run_id
        stream = _Stream(
            sent=self._clock(),
            streamed=self.config.stream,
            recorded_prompt_tokens=call.prompt_tokens,
        )
        self._in_flight += 1
        self.requests_sent += 1
        try:
            url = f"{target.base_url}/v1/chat/completions"
            if self.config.stream:
                async with self._client.stream(
                    "POST", url, json=body, headers=headers
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        self._error(f"http_{response.status_code}")
                        return None
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)
                        if not isinstance(chunk, dict) or "error" in chunk:
                            self._error("stream_error")
                            return None
                        self._on_chunk(stream, chunk)
            else:
                response = await self._client.post(
                    url,
                    content=_dumps(body),
                    headers={**headers, "content-type": "application/json"},
                )
                if response.status_code != 200:
                    self._error(f"http_{response.status_code}")
                    return None
                payload = _loads(response.content)
                if not isinstance(payload, dict) or "error" in payload:
                    self._error("response_error")
                    return None
                self._on_response(stream, payload)
        except asyncio.CancelledError:
            raise
        except httpx.TimeoutException:
            if not session.stopped:
                self._error("timeout")
            return None
        except (httpx.HTTPError, json.JSONDecodeError, OSError, RuntimeError) as error:
            # A stopped session's request ends when the client closes under it.
            if not session.stopped:
                self._error(type(error).__name__)
            return None
        finally:
            self._in_flight -= 1
        if stream.first_token is None:
            self._error("no_tokens")
            return None
        if input_ids is not None and stream.completion_ids is None:
            self._error("no_token_ids")
            return None
        self._on_done(stream)
        return stream.assistant_message(), stream.completion_ids

    def _on_chunk(self, stream: _Stream, chunk: dict[str, Any]) -> None:
        now = self._clock()
        counted = stream.tokens
        text = False
        logprob_tokens = 0
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                stream.reasoning.append(delta["reasoning_content"])
                text = True
            if delta.get("content"):
                stream.content.append(delta["content"])
                text = True
            for tool_call in delta.get("tool_calls") or ():
                if isinstance(tool_call, dict):
                    stream.merge_tool_call(tool_call)
                    text = True
            logprobs = choice.get("logprobs")
            if isinstance(logprobs, dict) and isinstance(logprobs.get("content"), list):
                logprob_tokens += len(logprobs["content"])
        usage = chunk.get("usage")
        reported = None
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                stream.prompt_tokens = usage["prompt_tokens"]
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict) and isinstance(
                details.get("cached_tokens"), int
            ):
                stream.cached_tokens = details["cached_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                reported = usage["completion_tokens"]
        # The server's running count when it streams one; else one token per logprob
        # entry, or per chunk of text. A final usage chunk corrects any estimate.
        if reported is not None:
            counted = max(counted, reported)
        elif logprob_tokens:
            counted += logprob_tokens
        elif text:
            counted += 1
        added = counted - stream.tokens
        if added <= 0:
            return
        if stream.first_token is None:
            stream.first_token = now
            stream.tokens_at_first = counted
        stream.last_token = now
        stream.tokens = counted
        if self._window is not None:
            self._window.output_tokens += added

    def _on_response(self, stream: _Stream, payload: dict[str, Any]) -> None:
        """A whole response: its message, and its token counts from ``usage``."""
        now = self._clock()
        for choice in payload.get("choices") or ():
            message = choice.get("message") or {}
            if message.get("reasoning_content"):
                stream.reasoning.append(message["reasoning_content"])
            if message.get("content"):
                stream.content.append(message["content"])
            for index, tool_call in enumerate(message.get("tool_calls") or ()):
                if isinstance(tool_call, dict):
                    stream.merge_tool_call({"index": index, **tool_call})
            meta = choice.get("meta_info")
            if isinstance(meta, dict) and isinstance(
                meta.get("output_token_logprobs"), list
            ):
                stream.completion_ids = [
                    entry[1] for entry in meta["output_token_logprobs"]
                ]
        usage = payload.get("usage")
        if not isinstance(usage, dict) or not isinstance(
            usage.get("completion_tokens"), int
        ):
            return
        if isinstance(usage.get("prompt_tokens"), int):
            stream.prompt_tokens = usage["prompt_tokens"]
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
            stream.cached_tokens = details["cached_tokens"]
        if usage["completion_tokens"] <= 0:
            return
        stream.first_token = stream.last_token = now
        stream.tokens = usage["completion_tokens"]
        if self._window is not None:
            self._window.output_tokens += stream.tokens

    def _on_done(self, stream: _Stream) -> None:
        window = self._window
        if window is None:
            return
        assert stream.first_token is not None and stream.last_token is not None
        done = self._clock()
        window.completed += 1
        window.latency.append(done - stream.sent)
        prompt = stream.prompt_tokens or 0
        window.prompt_tokens += prompt
        window.cached_tokens += min(stream.cached_tokens, prompt)
        window.prompt_lengths.append(prompt)
        if stream.recorded_prompt_tokens and prompt:
            window.served_prompt_tokens += prompt
            window.recorded_prompt_tokens += stream.recorded_prompt_tokens
        window.completion_lengths.append(stream.tokens)
        if done > stream.sent:
            window.e2e_speed.append(stream.tokens / (done - stream.sent))
        if not stream.streamed:
            return
        window.ttft.append(stream.first_token - stream.sent)
        decode_tokens = stream.tokens - stream.tokens_at_first
        decode_s = stream.last_token - stream.first_token
        if decode_tokens >= self.config.min_decode_tokens and decode_s > 0:
            window.decode_speed.append(decode_tokens / decode_s)

    def _error(self, kind: str) -> None:
        self.errors[kind] += 1
        if self._window is not None:
            self._window.errors[kind] += 1

    # ── background ──

    async def _tick(self, window: Window) -> None:
        """Sample the requests in flight and how late this event loop wakes, which
        exposes a load generator too slow for its sessions."""
        while True:
            before = self._clock()
            await asyncio.sleep(self.config.tick_s)
            window.loop_lag.append(
                max(0.0, self._clock() - before - self.config.tick_s)
            )
            window.in_flight.append(self._in_flight)

    async def _scrape(self, target: Target) -> tuple[float, dict[str, float] | None]:
        assert self._client is not None
        if not self.config.metrics_path:
            return self._clock(), None
        try:
            response = await self._client.get(
                f"{target.base_url}{self.config.metrics_path}",
                headers=target.headers,
                timeout=15.0,
            )
        except httpx.HTTPError:
            return self._clock(), None
        if response.status_code != 200:
            return self._clock(), None
        return self._clock(), parse_prometheus(response.text)

    async def _scrape_all(self) -> list[tuple[float, dict[str, float] | None]]:
        return list(await asyncio.gather(*(self._scrape(t) for t in self.targets)))

    async def _sample_metrics(
        self, samples: list[list[tuple[float, dict[str, float] | None]]]
    ) -> None:
        if not self.config.metrics_path:
            return
        while True:
            await asyncio.sleep(self.config.metrics_interval_s)
            samples.append(await self._scrape_all())
