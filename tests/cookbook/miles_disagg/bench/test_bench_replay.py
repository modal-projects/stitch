"""The trajectory replay, against an in-process fake engine that streams at a set rate."""

import asyncio
import json
import re
from collections import defaultdict

import pytest

from cookbook.miles_disagg.bench import replay
from cookbook.miles_disagg.bench.fake_server import BASH_TOOL, FakeEngine

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _record(name, completions, *, drop=(), task=None):
    """A dumped trajectory whose messages name their call: call k's output says
    ``recorded <name> <k>`` and the tool result after it ``out <name> <k>``. A call in
    ``drop`` was a format error: its output was replaced by a user message."""
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": task or f"Fix {name}."},
    ]
    for k, tokens in enumerate(completions):
        usage = {"model_call": True, "completion_tokens": tokens, "prompt_tokens": 10}
        if k in drop:
            messages.append(
                {"role": "user", "content": f"format error {name} {k}", **usage}
            )
            continue
        messages.append(
            {
                "role": "assistant",
                "content": f"recorded {name} {k}",
                "reasoning_content": f"thought {k}",
                "tool_calls": [
                    {
                        "id": f"{name}-{k}",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                    }
                ],
                **usage,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"{name}-{k}",
                "content": f"out {name} {k}",
            }
        )
    messages.append(
        {"role": "exit", "content": "Submitted", "exit_status": "Submitted"}
    )
    return {
        "format": replay.TRAJECTORY_FORMAT,
        "instance_id": name,
        "tools": [BASH_TOOL],
        "messages": messages,
    }


def _trajectories(*records):
    return [replay.parse_trajectory(r, r["instance_id"]) for r in records]


def _call_index(trajectory, messages):
    """Which call of ``trajectory`` a request's messages are for, from its last one."""
    last = messages[-1]["content"]
    if last.startswith("[replay "):
        return 0
    *_, k = last.split()
    return int(k) + 1


def _by_run(engine):
    runs = defaultdict(list)
    for request in engine.requests:
        runs[request["headers"]["modal-session-id"]].append(request["body"])
    return runs


def test_parse_splits_calls_and_keeps_dropped_outputs_out_of_context():
    (trajectory,) = _trajectories(_record("t", [5, 6, 7], drop={1}))

    assert [c.completion_tokens for c in trajectory.calls] == [5, 6, 7]
    assert [c.output is None for c in trajectory.calls] == [False, True, False]
    assert [m["role"] for m in trajectory.calls[0].before] == ["system", "user"]
    assert [m["content"] for m in trajectory.calls[1].before] == ["out t 0"]
    # The format-error message is the next call's context; the dropped output is not.
    assert [m["content"] for m in trajectory.calls[2].before] == ["format error t 1"]
    assert [m["content"] for m in trajectory.prefix(2)] == [
        "You are an agent.",
        "Fix t.",
        "recorded t 0",
        "out t 0",
        "format error t 1",
    ]
    recorded = trajectory.calls[0].output
    assert recorded["reasoning_content"] == "thought 0"
    assert json.loads(recorded["tool_calls"][0]["function"]["arguments"]) == {
        "command": "ls"
    }
    assert trajectory.tools[0]["function"]["name"] == "bash"


def test_load_skips_partial_files_and_foreign_formats(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_record("a", [3])))
    (tmp_path / "b.partial").write_text("{")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.json").write_text(json.dumps({"format": "other"}))
    (tmp_path / "nested" / "d.json").write_text(json.dumps(_record("d", [])))

    loaded = replay.load_trajectories(tmp_path)

    assert [t.name for t in loaded] == ["a"]


def test_tool_calls_cut_off_mid_arguments_are_dropped():
    calls = [
        {"function": {"name": "bash", "arguments": '{"command": "ls"}'}},
        {"function": {"name": "bash", "arguments": '{"command": "l'}},
        {"function": {"name": "", "arguments": "{}"}},
        {"function": {"name": "bash", "arguments": "[1]"}},
    ]

    (valid,) = replay.valid_tool_calls(calls)

    assert valid == {
        "id": "call_0",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }


async def _replay_for(engine, trajectories, sessions, seconds, **config):
    settings = {"metrics_path": None, "seed": 7, **config}
    async with replay.Replay(
        trajectories, [replay.Target(engine.url)], replay.ReplayConfig(**settings)
    ) as running:
        await running.resize(sessions)
        await asyncio.sleep(seconds)
        return running


def test_requests_carry_the_recorded_length_with_ignore_eos_and_sampling():
    async def scenario() -> None:
        trajectories = _trajectories(_record("a", [5, 9, 13]), _record("b", [4, 8]))
        async with FakeEngine(token_rate=2000) as engine:
            await _replay_for(engine, trajectories, sessions=4, seconds=0.5)

        assert len(engine.requests) > 8
        by_name = {t.name: t for t in trajectories}
        for request in engine.requests:
            body = request["body"]
            assert body["ignore_eos"] is True
            assert body["stream"] is True
            assert body["stream_options"] == {
                "include_usage": True,
                "continuous_usage_stats": True,
            }
            assert (body["temperature"], body["top_p"], body["top_k"]) == (
                1.0,
                0.97,
                64,
            )
            assert body["logprobs"] is True
            assert body["model"] == "fake-model"
            assert body["tools"] == [BASH_TOOL]
            name = body["messages"][1]["content"].split("Fix ")[1].rstrip(".")
            trajectory = by_name[name]
            call = trajectory.calls[_call_index(trajectory, body["messages"])]
            assert body["max_tokens"] == call.completion_tokens

    asyncio.run(scenario())


def test_sessions_start_mid_trajectory_from_the_recorded_context():
    async def scenario() -> None:
        trajectories = _trajectories(_record("long", list(range(5, 25))))
        async with FakeEngine(token_rate=5000) as engine:
            await _replay_for(engine, trajectories, sessions=12, seconds=0.3)
        (trajectory,) = trajectories
        runs = _by_run(engine)

        first_runs = [r for run, r in runs.items() if run.endswith("-1")]
        starts = [_call_index(trajectory, r[0]["messages"]) for r in first_runs]
        assert len(first_runs) == 12
        assert len(set(starts)) > 3 and max(starts) > 0
        for requests, start in zip(first_runs, starts, strict=True):
            expected = trajectory.prefix(start)
            sent = requests[0]["messages"]
            # Recorded context, recorded outputs included, with the run tag on the task.
            assert sent[0] == expected[0]
            assert sent[1]["content"].startswith("[replay bench-7-")
            assert sent[1]["content"].endswith("\nFix long.")
            assert sent[2:] == expected[2:]

    asyncio.run(scenario())


def test_context_grows_with_the_generated_outputs():
    async def scenario() -> None:
        trajectories = _trajectories(_record("g", [6, 7, 8, 9], drop={2}))
        async with FakeEngine(token_rate=5000) as engine:
            await _replay_for(engine, trajectories, sessions=3, seconds=0.4)
        (trajectory,) = trajectories

        checked = 0
        for requests in _by_run(engine).values():
            for previous, current in zip(requests, requests[1:], strict=False):
                sent, before = current["messages"], previous["messages"]
                k = _call_index(trajectory, before)
                assert sent[: len(before)] == before
                appended = sent[len(before) :]
                call = trajectory.calls[k]
                if call.output is None:
                    # A dropped output never enters the context.
                    assert appended == list(trajectory.calls[k + 1].before)
                else:
                    generated, *rest = appended
                    assert generated["role"] == "assistant"
                    assert not generated["content"].startswith("recorded")
                    assert generated["content"].startswith("c")
                    assert generated["reasoning_content"].startswith("r0 ")
                    (tool_call,) = generated["tool_calls"]
                    assert json.loads(tool_call["function"]["arguments"]) == {
                        "command": "ls"
                    }
                    assert rest == list(trajectory.calls[k + 1].before)
                checked += 1
        assert checked > 5

    asyncio.run(scenario())


def test_finished_trajectories_hand_over_to_new_ones_from_their_first_turn():
    async def scenario() -> None:
        trajectories = _trajectories(_record("x", [3, 3]), _record("y", [3, 3, 3]))
        async with FakeEngine(token_rate=5000) as engine:
            running = await _replay_for(engine, trajectories, sessions=2, seconds=0.4)
        runs = _by_run(engine)

        later = {run: r for run, r in runs.items() if not run.endswith("-1")}
        assert len(later) >= 4
        for requests in later.values():
            assert requests[0]["messages"][0] == {
                "role": "system",
                "content": "You are an agent.",
            }
            assert len(requests[0]["messages"]) == 2
        tags = {r[0]["messages"][1]["content"].split("\n")[0] for r in runs.values()}
        assert len(tags) == len(runs)
        assert running.errors == {}

    asyncio.run(scenario())


def test_a_failed_request_is_counted_and_the_session_moves_on():
    async def scenario() -> None:
        trajectories = _trajectories(
            _record("ok", [4, 4, 4]), _record("bad", [4, 4], task="poison")
        )

        def reject(body):
            return 400 if body["messages"][1]["content"].endswith("poison") else None

        async with FakeEngine(token_rate=5000, reject=reject) as engine:
            async with replay.Replay(
                trajectories,
                [replay.Target(engine.url)],
                replay.ReplayConfig(metrics_path=None, error_backoff_s=0.01),
            ) as running:
                await running.resize(4)
                summary = await running.measure(0.5)

        assert summary["errors"] > 0
        assert set(summary["error_kinds"]) == {"http_400"}
        assert summary["requests"] > 0
        assert running.errors["http_400"] >= summary["errors"]

    asyncio.run(scenario())


def test_window_measures_decode_speed_ttft_and_throughput():
    async def scenario() -> None:
        trajectories = _trajectories(*(_record(f"w{i}", [40] * 30) for i in range(4)))
        async with FakeEngine(token_rate=100, ttft_s=0.05) as engine:
            async with replay.Replay(
                trajectories,
                [replay.Target(engine.url)],
                replay.ReplayConfig(metrics_interval_s=0.5, seed=1),
            ) as running:
                await running.resize(4)
                waited = await running.warmup(0.2, 5.0)
                summary = await running.measure(2.0)

        assert waited >= 0.2
        assert summary["sessions"] == 4
        assert summary["requests"] >= 4 and summary["errors"] == 0
        assert 70 <= summary["decode_tok_s_p50"] <= 110
        assert summary["decode_tok_s_p10"] <= summary["decode_tok_s_p50"]
        assert summary["decode_tok_s_p50"] <= summary["decode_tok_s_p90"]
        assert 0.04 <= summary["ttft_s_p50"] <= 0.2
        # Four sessions at ~100 tok/s each, less time spent waiting for first tokens.
        assert 4 * 100 * 0.5 <= summary["output_tok_s"] <= 4 * 100 * 1.1
        assert summary["completion_tokens_p50"] == 40
        assert summary["prompt_tokens"] > 0 and summary["prompt_tok_s"] > 0
        assert summary["latency_s_p50"] >= summary["ttft_s_p50"]
        assert summary["in_flight_mean"] > 3
        server = summary["server"]
        assert server["rate:sglang:generation_tokens_total"] == pytest.approx(
            summary["output_tok_s"], rel=0.3
        )
        assert 3 <= server["mean:sglang:num_running_reqs"] <= 4

    asyncio.run(scenario())


def test_whole_responses_measure_latency_and_throughput_and_carry_the_turn():
    async def scenario() -> None:
        trajectories = _trajectories(*(_record(f"n{i}", [40] * 30) for i in range(4)))
        async with FakeEngine(token_rate=100, ttft_s=0.05) as engine:
            async with replay.Replay(
                trajectories,
                [replay.Target(engine.url)],
                replay.ReplayConfig(metrics_path=None, seed=1, stream=False),
            ) as running:
                await running.resize(4)
                await running.warmup(0.2, 5.0)
                summary = await running.measure(2.0)

        assert summary["requests"] >= 4 and summary["errors"] == 0
        assert all(r["body"]["stream"] is False for r in engine.requests)
        assert all("stream_options" not in r["body"] for r in engine.requests)
        # 40 tokens at 100 tok/s plus the first-token wait.
        assert 0.4 <= summary["latency_s_p50"] <= 0.7
        assert summary["completion_tokens_p50"] == 40
        assert summary["prompt_tokens"] > 0
        # Whole responses time no first token or decode.
        assert summary["ttft_s_p50"] is None and summary["decode_tok_s_p50"] is None
        assert 4 * 100 * 0.5 <= summary["output_tok_s"] <= 4 * 100 * 1.1
        assert summary["loop_lag_s_p90"] is not None
        # Each next turn carries the previous whole response, tool call included.
        later = [
            r["body"]["messages"]
            for r in engine.requests
            if any(m["role"] == "assistant" for m in r["body"]["messages"][2:])
        ]
        assert later
        generated = [
            m
            for messages in later
            for m in messages
            if m.get("content", "").startswith("c ")
        ]
        assert generated
        assert all(m["reasoning_content"].startswith("r ") for m in generated)
        assert all(
            m["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'
            for m in generated
        )

    asyncio.run(scenario())


def test_tokens_are_counted_without_running_usage_and_settled_by_the_final_usage():
    async def scenario() -> None:
        trajectories = _trajectories(_record("u", [30] * 5))
        async with FakeEngine(token_rate=3000, continuous_usage=False) as engine:
            async with replay.Replay(
                trajectories,
                [replay.Target(engine.url)],
                replay.ReplayConfig(metrics_path=None, logprobs=False),
            ) as running:
                await running.resize(1)
                await running.warmup(0.0, 2.0)
                summary = await running.measure(0.5)

        assert summary["requests"] >= 1
        assert summary["completion_tokens_p50"] == 30
        assert all(
            "continuous_usage_stats" in r["body"]["stream_options"]
            for r in engine.requests
        )
        assert all("logprobs" not in r["body"] for r in engine.requests)

    asyncio.run(scenario())


def test_streamed_chunks_count_tokens_three_ways():
    running = replay.Replay(_trajectories(_record("c", [3])), [replay.Target("x")])
    window = replay.Window(0.0, 1)
    running._window = window

    by_usage = replay._Stream(sent=0.0)
    running._on_chunk(
        by_usage,
        {
            "choices": [{"delta": {"role": "assistant"}}],
            "usage": {"completion_tokens": 1},
        },
    )
    running._on_chunk(
        by_usage,
        {"choices": [{"delta": {"content": "ab"}}], "usage": {"completion_tokens": 3}},
    )
    by_logprobs = replay._Stream(sent=0.0)
    running._on_chunk(
        by_logprobs,
        {"choices": [{"delta": {"content": "x"}, "logprobs": {"content": [{}, {}]}}]},
    )
    by_chunks = replay._Stream(sent=0.0)
    running._on_chunk(
        by_chunks, {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
    )
    running._on_chunk(by_chunks, {"choices": [{"delta": {"reasoning_content": "r"}}]})
    running._on_chunk(by_chunks, {"choices": [], "usage": {"completion_tokens": 5}})

    assert (by_usage.tokens, by_usage.tokens_at_first) == (3, 1)
    assert by_logprobs.tokens == 2
    assert (by_chunks.tokens, by_chunks.tokens_at_first) == (5, 1)
    assert window.output_tokens == 3 + 2 + 5


def test_window_summary_percentiles_and_rates():
    window = replay.Window(10.0, 2)
    window.output_tokens = 500
    window.prompt_tokens, window.cached_tokens, window.completed = 1000, 600, 4
    window.decode_speed = [10.0, 20.0, 30.0, 40.0, 50.0]
    window.ttft = [0.1, 0.3]
    window.errors["timeout"] += 1
    window.ended = 20.0

    summary = window.summary()

    assert summary["output_tok_s"] == 50.0
    assert summary["uncached_prompt_tok_s"] == 40.0
    assert summary["requests_s"] == 0.4
    assert summary["decode_tok_s_p50"] == 30.0
    assert summary["decode_tok_s_p10"] == pytest.approx(14.0)
    assert summary["ttft_s_p50"] == pytest.approx(0.2)
    assert summary["latency_s_p50"] is None
    assert summary["error_kinds"] == {"timeout": 1}


def test_prometheus_parsing_and_window_rates():
    text = (
        "# TYPE sglang:generation_tokens_total counter\n"
        'sglang:generation_tokens_total{tp_rank="0"} 100\n'
        'sglang:generation_tokens_total{tp_rank="1"} 100\n'
        'sglang:num_running_reqs{tp_rank="0"} 4\n'
        'sglang:ttft_seconds_bucket{le="1"} 3\n'
        "sglang:ttft_seconds_sum 2.5\n"
        "sglang:cache_hit_rate NaN\n"
    )

    parsed = replay.parse_prometheus(text)

    assert parsed == {
        "sglang:generation_tokens_total": 100.0,
        "sglang:num_running_reqs": 4.0,
        "sglang:ttft_seconds_sum": 2.5,
    }
    before = [(0.0, {"x_total": 10.0, "g": 1.0}), (0.0, None)]
    after = [(10.0, {"x_total": 60.0, "g": 3.0}), (10.0, None)]
    samples = [[(5.0, {"g": 2.0}), (5.0, None)]]
    assert replay.server_metrics(before, after, samples) == {
        "rate:x_total": 5.0,
        "mean:g": 2.5,
    }


# ── Token prompts (TITO) ──


class QwenLikeTokenizer:
    """A Qwen-style chat template over one token per character (special markers whole),
    so encodings concatenate exactly: a token prompt built turn by turn must equal a
    full re-render of the same conversation."""

    SPECIAL = re.compile(r"<\|[a-z_]+\|>|.", re.S)

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    ):
        assert tokenize is False
        out = (
            ["<|im_start|>tools\n" + json.dumps(tools) + "<|im_end|>\n"]
            if tools
            else []
        )
        index = 0
        while index < len(messages):
            message = messages[index]
            if message["role"] == "tool":  # consecutive tool results share one turn
                block = []
                while index < len(messages) and messages[index]["role"] == "tool":
                    block.append(f"<tool_response>{messages[index]['content']}")
                    index += 1
                out.append("<|im_start|>user\n" + "\n".join(block) + "<|im_end|>\n")
                continue
            body = message.get("content") or ""
            if message["role"] == "assistant":
                body = f"<think>{message.get('reasoning_content', '')}</think>{body}"
                for call in message.get("tool_calls") or ():
                    assert isinstance(call["function"]["arguments"], dict)
                    body += f"<tool_call>{call['function']['name']}"
            out.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
            index += 1
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)

    def encode(self, text, add_special_tokens=False):
        return [
            self.vocab.setdefault(p, len(self.vocab))
            for p in self.SPECIAL.findall(text)
        ]


def _assistant_ids(tokenizer, message):
    """The token IDs a model emits for ``message``: its rendered body, then <|im_end|>."""
    text = tokenizer.apply_chat_template([replay.template_message(message)])
    body = text.split("assistant\n", 1)[1]
    return tokenizer.encode(body[: body.index("<|im_end|>") + len("<|im_end|>")])


def test_token_prompts_built_turn_by_turn_equal_a_full_render():
    tokenizer = QwenLikeTokenizer()
    prompts = replay.TokenPrompts(tokenizer)
    (trajectory,) = _trajectories(_record("t", [5, 6, 7, 8, 9], drop=(2,)))
    full = lambda messages: tokenizer.encode(  # noqa: E731
        tokenizer.apply_chat_template(
            [replay.template_message(m) for m in messages],
            tools=list(trajectory.tools),
            add_generation_prompt=True,
        )
    )

    for start in range(len(trajectory.calls)):
        messages = trajectory.prefix(start)
        context, pending = prompts.render(messages, trajectory.tools), []
        for position in range(start, len(trajectory.calls)):
            call = trajectory.calls[position]
            if position > start:
                messages.extend(call.before)
                pending.extend(call.before)
            ids = context + prompts.suffix(pending)
            assert ids == full(messages), (start, position)
            if call.output is not None:  # the agent kept it: it joins the context
                messages.append(call.output)
                context = ids + prompts.close(_assistant_ids(tokenizer, call.output))
                pending = []


def test_a_capped_completion_gets_the_whole_end_of_turn_a_stopped_one_only_the_rest():
    tokenizer = QwenLikeTokenizer()
    prompts = replay.TokenPrompts(tokenizer)
    end, newline = tokenizer.encode("<|im_end|>\n")

    assert prompts.end_of_turn == [end, newline]
    assert prompts.close([7, 8]) == [7, 8, end, newline]
    assert prompts.close([7, end]) == [7, end, newline]


def test_token_prompts_carry_the_context_ids_and_extend_them_by_each_completion():
    async def scenario() -> None:
        from cookbook.miles_disagg.bench.fake_server import COMPLETION_ID_BASE

        tokenizer = QwenLikeTokenizer()
        prompts = replay.TokenPrompts(tokenizer)
        trajectories = _trajectories(_record("k", [4, 5, 6, 7]))
        config = replay.ReplayConfig(
            metrics_path=None,
            seed=2,
            stream=False,
            random_start=False,
            top_logprobs=128,
        )
        async with FakeEngine(token_rate=4000) as engine:
            async with replay.Replay(
                trajectories, [replay.Target(engine.url)], config, prompts=prompts
            ) as running:
                await running.resize(1)
                await running.warmup(0.0, 3.0)
                summary = await running.measure(0.5)

        bodies = [request["body"] for request in engine.requests]
        assert summary["errors"] == 0 and len(bodies) >= 3
        assert all(
            b["return_meta_info"] is True and b["logprobs"] is True for b in bodies
        )
        assert all(b["top_logprobs"] == 128 and b["stream"] is False for b in bodies)
        first, second = bodies[0], bodies[1]
        completion = [COMPLETION_ID_BASE + i for i in range(first["max_tokens"])]
        tool_turn = prompts.suffix(trajectories[0].calls[1].before)
        assert second["input_ids"] == (
            first["input_ids"] + prompts.close(completion) + tool_turn
        )
        # The engine saw exactly the token prompt.
        assert summary["prompt_tokens"] > 0

    asyncio.run(scenario())


def test_token_prompts_need_whole_responses():
    with pytest.raises(ValueError, match="stream=False"):
        replay.Replay(
            _trajectories(_record("x", [3])),
            [replay.Target("http://unused")],
            replay.ReplayConfig(stream=True),
            prompts=replay.TokenPrompts(QwenLikeTokenizer()),
        )


def test_a_window_compares_served_prompt_lengths_with_the_recorded_ones():
    window = replay.Window(0.0, 1)
    window.served_prompt_tokens, window.recorded_prompt_tokens = 990, 1000
    window.ended = 1.0

    assert window.summary()["prompt_vs_recorded"] == pytest.approx(0.99)
