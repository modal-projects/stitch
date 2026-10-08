"""A stand-in for an SGLang engine: OpenAI-compatible chat completions, streamed or whole,
that decode at a set rate, for testing the replay and the sweep without a GPU.

Raw asyncio HTTP/1.1, one response per connection: httpx's in-process ASGI transport
buffers a whole response, which would hide every timing the replay measures.

    python -m cookbook.miles_disagg.bench.fake_server --port 8000 --rate 80 \\
        --capacity 16 --synthetic-traces /tmp/bench-traces
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cookbook.miles_disagg.bench.replay import TRAJECTORY_FORMAT

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


# Completion token IDs the fake engine returns: COMPLETION_ID_BASE + position.
COMPLETION_ID_BASE = 100_000


def _words(value: Any) -> int:
    return len(json.dumps(value).split()) if value is not None else 0


class FakeEngine:
    """Streams ``max_tokens`` tokens per request (as ``ignore_eos`` would), the first
    after ``ttft_s`` and then ``token_rate`` per second. With ``capacity``, a request
    decodes at ``token_rate * capacity / active`` once more than ``capacity`` run, so
    per-request speed falls as load rises. ``reject(body)`` may return an HTTP status
    to fail a request with. Every request's headers and body are kept in ``requests``."""

    def __init__(
        self,
        *,
        token_rate: float = 200.0,
        ttft_s: float = 0.0,
        capacity: int | None = None,
        model: str = "fake-model",
        continuous_usage: bool = True,
        reject: Callable[[dict[str, Any]], int | None] | None = None,
    ) -> None:
        self.token_rate = token_rate
        self.ttft_s = ttft_s
        self.capacity = capacity
        self.model = model
        self.continuous_usage = continuous_usage
        self.reject = reject
        self.requests: list[dict[str, Any]] = []
        self.active = 0
        self.generated = 0
        self.prompt_total = 0
        self.url = ""
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> FakeEngine:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        self._server = await asyncio.start_server(self._handle, host, port)
        bound = self._server.sockets[0].getsockname()
        self.url = f"http://{bound[0]}:{bound[1]}"
        return self.url

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), timeout=1.0)
            self._server = None

    def rate(self) -> float:
        if self.capacity and self.active > self.capacity:
            return self.token_rate * self.capacity / self.active
        return self.token_rate

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            method, path, _ = request_line.decode().split(" ", 2)
            headers: dict[str, str] = {}
            while (line := await reader.readline()) not in (b"\r\n", b"\n", b""):
                name, _, value = line.decode().partition(":")
                headers[name.strip().lower()] = value.strip()
            body = await reader.readexactly(int(headers.get("content-length") or 0))
            if method == "GET" and path == "/v1/models":
                await self._send_json(
                    writer, 200, {"object": "list", "data": [{"id": self.model}]}
                )
            elif method == "GET" and path == "/health":
                await self._send_json(writer, 200, {})
            elif method == "GET" and path == "/metrics":
                await self._send(writer, 200, "text/plain", self.metrics().encode())
            elif method == "POST" and path == "/v1/chat/completions":
                await self._chat(writer, headers, json.loads(body))
            else:
                await self._send_json(writer, 404, {"error": "not found"})
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def metrics(self) -> str:
        labels = f'{{model_name="{self.model}"}}'
        return (
            "# TYPE sglang:generation_tokens_total counter\n"
            f"sglang:generation_tokens_total{labels} {self.generated}\n"
            f"sglang:prompt_tokens_total{labels} {self.prompt_total}\n"
            "# TYPE sglang:num_running_reqs gauge\n"
            f"sglang:num_running_reqs{labels} {self.active}\n"
        )

    async def _send(
        self, writer: asyncio.StreamWriter, status: int, media: str, body: bytes
    ) -> None:
        writer.write(
            f"HTTP/1.1 {status} X\r\nContent-Type: {media}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()

    async def _send_json(
        self, writer: asyncio.StreamWriter, status: int, payload: Any
    ) -> None:
        await self._send(
            writer, status, "application/json", json.dumps(payload).encode()
        )

    async def _chat(
        self,
        writer: asyncio.StreamWriter,
        headers: dict[str, str],
        body: dict[str, Any],
    ) -> None:
        self.requests.append({"headers": headers, "body": body})
        if self.reject is not None and (status := self.reject(body)):
            await self._send_json(writer, status, {"error": {"message": "rejected"}})
            return
        tokens = max(1, int(body.get("max_tokens") or 16))
        if isinstance(body.get("input_ids"), list):
            prompt = len(body["input_ids"])
        else:
            prompt = _words(body.get("messages")) + _words(body.get("tools"))
        if body.get("stream") is False:
            await self._chat_whole(writer, body, tokens, prompt)
            return
        options = body.get("stream_options") or {}
        continuous = self.continuous_usage and options.get("continuous_usage_stats")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()

        async def emit(chunk: dict[str, Any]) -> None:
            writer.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await writer.drain()

        def usage(completion: int) -> dict[str, Any]:
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "prompt_tokens_details": {"cached_tokens": 0},
            }

        self.active += 1
        self.prompt_total += prompt
        try:
            await asyncio.sleep(self.ttft_s)
            for index in range(tokens):
                if index:
                    await asyncio.sleep(1.0 / self.rate())
                delta: dict[str, Any] = {}
                if index == 0:
                    delta["role"] = "assistant"
                if body.get("tools") and tokens >= 4 and index == tokens - 2:
                    delta["tool_calls"] = [
                        {
                            "index": 0,
                            "id": f"call_{len(self.requests)}",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": '},
                        }
                    ]
                elif body.get("tools") and tokens >= 4 and index == tokens - 1:
                    delta["tool_calls"] = [
                        {"index": 0, "function": {"arguments": '"ls"}'}}
                    ]
                elif index < tokens // 2:
                    delta["reasoning_content"] = f"r{index} "
                else:
                    delta["content"] = f"c{index} "
                choice: dict[str, Any] = {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None,
                }
                if body.get("logprobs"):
                    choice["logprobs"] = {
                        "content": [{"token": f"t{index}", "logprob": -0.5}]
                    }
                chunk: dict[str, Any] = {
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [choice],
                }
                if continuous:
                    chunk["usage"] = usage(index + 1)
                self.generated += 1
                await emit(chunk)
            await emit(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
            )
            if options.get("include_usage"):
                await emit({"choices": [], "usage": usage(tokens)})
            writer.write(b"data: [DONE]\n\n")
            await writer.drain()
        finally:
            self.active -= 1

    async def _chat_whole(
        self,
        writer: asyncio.StreamWriter,
        body: dict[str, Any],
        tokens: int,
        prompt: int,
    ) -> None:
        """A non-streamed completion: decode at the same rate, then answer whole."""
        self.active += 1
        self.prompt_total += prompt
        try:
            await asyncio.sleep(self.ttft_s)
            for _ in range(1, tokens):
                await asyncio.sleep(1.0 / self.rate())
            self.generated += tokens
        finally:
            self.active -= 1
        message: dict[str, Any] = {
            "role": "assistant",
            "reasoning_content": "r " * (tokens // 2),
            "content": "c " * (tokens - tokens // 2),
        }
        if body.get("tools") and tokens >= 4:
            message["tool_calls"] = [
                {
                    "id": f"call_{len(self.requests)}",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                }
            ]
        choice: dict[str, Any] = {
            "index": 0,
            "message": message,
            "finish_reason": "length",
        }
        if body.get("logprobs"):
            choice["logprobs"] = {
                "content": [{"token": f"t{i}", "logprob": -0.5} for i in range(tokens)]
            }
        if body.get("return_meta_info"):
            # SGLang's (logprob, token id, text) per output token.
            choice["meta_info"] = {
                "output_token_logprobs": [
                    [-0.5, COMPLETION_ID_BASE + i, None] for i in range(tokens)
                ],
                "completion_tokens": tokens,
            }
        await self._send_json(
            writer,
            200,
            {
                "object": "chat.completion",
                "model": self.model,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": tokens,
                    "total_tokens": prompt + tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )


def synthetic_trajectory(
    name: str,
    *,
    calls: int,
    completion_tokens: int = 32,
    tool_output_words: int = 50,
    seed: int = 0,
) -> dict[str, Any]:
    """A dumped trajectory as the agent writes one: system, task, then ``calls``
    assistant turns each followed by a tool result."""
    rng = random.Random(f"{name}:{seed}")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": f"Fix issue {name}."},
    ]
    for index in range(calls):
        messages.append(
            {
                "role": "assistant",
                "content": f"step {index}",
                "reasoning_content": "thinking",
                "tool_calls": [
                    {
                        "id": f"{name}-{index}",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": f"step {index}"}),
                        },
                    }
                ],
                "model_call": True,
                "prompt_tokens": 100 * (index + 1),
                "completion_tokens": max(
                    1, int(completion_tokens * rng.uniform(0.5, 1.5))
                ),
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"{name}-{index}",
                "content": " ".join(f"w{i}" for i in range(tool_output_words)),
            }
        )
    messages.append(
        {"role": "exit", "content": "Submitted", "exit_status": "Submitted"}
    )
    return {
        "format": TRAJECTORY_FORMAT,
        "instance_id": name,
        "sample_index": 0,
        "exit_status": "Submitted",
        "reward": 1.0,
        "model_calls": calls,
        "model_request_durations_seconds": [],
        "request_kwargs": {},
        "tools": [BASH_TOOL],
        "messages": messages,
    }


def write_synthetic_traces(directory: Path, count: int = 8, calls: int = 12) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        name = f"synthetic-{index}"
        record = synthetic_trajectory(name, calls=calls + index)
        (directory / f"{name}.json").write_text(json.dumps(record))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rate", type=float, default=80.0, help="tok/s per request")
    parser.add_argument("--ttft", type=float, default=0.05)
    parser.add_argument("--capacity", type=int, help="requests at full rate")
    parser.add_argument("--synthetic-traces", type=Path, help="write traces here first")
    args = parser.parse_args()
    if args.synthetic_traces:
        write_synthetic_traces(args.synthetic_traces)

    async def serve() -> None:
        engine = FakeEngine(
            token_rate=args.rate, ttft_s=args.ttft, capacity=args.capacity
        )
        print(f"fake engine at {await engine.start(args.host, args.port)}", flush=True)
        await asyncio.Event().wait()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
