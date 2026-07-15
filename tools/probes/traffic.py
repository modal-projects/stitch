"""Reward-free traffic shapes against a pool gateway.

Shapes stress the serving/sync protocol, not the model — what matters is prompt and
decode length, concurrency, and session structure. ``agentic`` is the headline shape
(the top real use case): multi-turn sessions whose context grows every turn with a
synthetic tool-result blob, pinned to one replica via the session-affinity header —
which makes per-publish KV-namespace rotation (``extra_key``) directly measurable as
turn-latency inflation right after a version flip.

Responses carry ``weight_version_start``/``weight_version_end`` (top-level on OpenAI
routes), so the generator doubles as the straddle-attribution collector.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AFFINITY_HEADER = "Modal-Session-ID"  # what the cookbook configs use
RETRY_409_SLEEP = 1.0
RETRY_409_LIMIT = 120
_WORDS = ("the model weights version pool replica delta chain anchor policy rollout "
          "context token prefill decode publish commit stage pointer session request").split()


@dataclass(frozen=True)
class Shape:
    prompt_tokens: tuple[int, int]
    max_tokens: tuple[int, int]
    turns: tuple[int, int] = (1, 1)
    tool_tokens: tuple[int, int] = (0, 0)  # per-turn context growth (agentic)


SHAPES: dict[str, Shape] = {
    "long_decode": Shape(prompt_tokens=(200, 800), max_tokens=(4096, 12288)),
    "long_prefill": Shape(prompt_tokens=(8_000, 24_000), max_tokens=(256, 1024)),
    "agentic": Shape(prompt_tokens=(1_000, 3_000), max_tokens=(256, 1536), turns=(4, 12), tool_tokens=(500, 4_000)),
}
MIXED_WEIGHTS = {"long_decode": 0.4, "long_prefill": 0.2, "agentic": 0.4}


def _filler(rng: random.Random, tokens: int) -> str:
    return " ".join(rng.choices(_WORDS, k=max(1, int(tokens * 0.75))))  # ~0.75 words/token


async def run(
    gateway: str,
    model: str,
    *,
    shape: str = "mixed",
    concurrency: int = 16,
    duration: float = 600.0,
    lag: int | None = None,  # floor requests at (gateway-observed version - lag); None = unconstrained
    out_path: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    import httpx

    rows: list[dict[str, Any]] = []
    floor = _VersionFloor(gateway) if lag is not None else None
    deadline = time.time() + duration
    async with httpx.AsyncClient(timeout=3600.0, trust_env=False) as client:
        if floor:
            await floor.start(client)
        workers = [
            asyncio.create_task(_worker(client, gateway, model, shape, deadline, rows, floor, lag, i, random.Random(seed + i)))
            for i in range(concurrency)
        ]
        await asyncio.gather(*workers)
        if floor:
            floor.stop()
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return summarize(rows)


async def _worker(client, gateway, model, shape_name, deadline, rows, floor, lag, worker_id, rng) -> None:  # noqa: ANN001
    n = 0  # one worker = sequential sessions
    while time.time() < deadline:
        name = shape_name if shape_name != "mixed" else rng.choices(*zip(*MIXED_WEIGHTS.items()))[0]
        await _session(client, gateway, model, name, SHAPES[name], rng, rows, floor, lag, session_id=f"w{worker_id}-{n}")
        n += 1


async def _session(client, gateway, model, name, spec, rng, rows, floor, lag, session_id) -> None:  # noqa: ANN001
    messages = [{"role": "user", "content": _filler(rng, rng.randint(*spec.prompt_tokens))}]
    headers = {AFFINITY_HEADER: session_id}
    for turn in range(rng.randint(*spec.turns)):
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": rng.randint(*spec.max_tokens),
            "temperature": 0.8,
        }
        if floor is not None and floor.version is not None:
            payload["weight_version"] = {"min_version": max(0, floor.version - (lag or 0))}
        row = {"t": time.time(), "shape": name, "session": session_id, "turn": turn, "retries_409": 0}
        data = await _post_with_retry(client, f"{gateway}/v1/chat/completions", payload, headers, row)
        rows.append(row)
        if data is None:
            return  # session dies with its failed request
        row.update(
            wv_start=data.get("weight_version_start"),
            wv_end=data.get("weight_version_end"),
            straddled=data.get("weight_version_start") != data.get("weight_version_end"),
        )
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        messages.append({"role": "assistant", "content": content})
        if spec.tool_tokens[1]:  # agentic: the "tool result" grows the context every turn
            messages.append({"role": "user", "content": f"tool result:\n{_filler(rng, rng.randint(*spec.tool_tokens))}\ncontinue."})
        else:
            break


async def _post_with_retry(client, url, payload, headers, row) -> dict[str, Any] | None:  # noqa: ANN001
    start = time.time()
    for _ in range(RETRY_409_LIMIT):
        try:
            resp = await client.post(url, json=payload, headers=headers)
        except Exception as exc:  # noqa: BLE001
            row.update(latency=time.time() - start, error=str(exc)[:200])
            return None
        if resp.status_code == 409:  # version not ready: the retryable staleness signal
            row["retries_409"] += 1
            await asyncio.sleep(RETRY_409_SLEEP)
            continue
        row.update(latency=time.time() - start, status=resp.status_code)
        if resp.status_code != 200:
            row["error"] = resp.text[:200]
            return None
        return resp.json()
    row.update(latency=time.time() - start, error="409 retry budget exhausted")
    return None


class _VersionFloor:
    """Track the pool's applied version via the gateway's /server_info (answers from an
    arbitrary replica — probe-grade, not exact)."""

    def __init__(self, gateway: str) -> None:
        self.gateway = gateway
        self.version: int | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self, client) -> None:  # noqa: ANN001
        from stitch.versions import VersionRef

        async def loop() -> None:
            while True:
                try:
                    info = (await client.get(f"{self.gateway}/server_info")).json()
                    if info.get("applied"):
                        self.version = VersionRef.parse(info["applied"]).version
                except Exception:  # noqa: BLE001
                    pass
                await asyncio.sleep(2.0)

        self._task = asyncio.create_task(loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_shape.setdefault(r["shape"], []).append(r)
    out: dict[str, Any] = {"requests": len(rows)}
    for name, rs in sorted(by_shape.items()):
        lat = sorted(r["latency"] for r in rs if "latency" in r)
        out[name] = {
            "n": len(rs),
            "latency_p50": lat[len(lat) // 2] if lat else None,
            "latency_p95": lat[int(len(lat) * 0.95)] if lat else None,
            "straddled": sum(bool(r.get("straddled")) for r in rs),
            "retries_409": sum(r.get("retries_409", 0) for r in rs),
            "errors": sum("error" in r for r in rs),
        }
    return out
