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


def _estimate_tokens(messages: list[dict[str, str]]) -> int:
    return int(sum(len(m["content"].split()) for m in messages) / 0.75)


async def run(
    gateway: str,
    model: str,
    *,
    shape: str = "mixed",
    concurrency: int = 16,
    duration: float = 600.0,
    lag: int | None = None,  # floor requests at (gateway-observed version - lag); None = unconstrained
    context_limit: int = 16384,  # engine --context-length; sessions stop growing before it
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
            asyncio.create_task(
                _worker(client, gateway, model, shape, deadline, rows, floor, lag, context_limit, i, random.Random(seed + i))
            )
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


async def _worker(client, gateway, model, shape_name, deadline, rows, floor, lag, context_limit, worker_id, rng) -> None:  # noqa: ANN001
    n = 0  # one worker = sequential sessions
    while time.time() < deadline:
        name = shape_name if shape_name != "mixed" else rng.choices(*zip(*MIXED_WEIGHTS.items()))[0]
        await _session(
            client, gateway, model, name, SHAPES[name], rng, rows, floor, lag, context_limit,
            session_id=f"w{worker_id}-{n}",
        )
        n += 1


async def _session(client, gateway, model, name, spec, rng, rows, floor, lag, context_limit, session_id) -> None:  # noqa: ANN001
    prompt_tokens = min(rng.randint(*spec.prompt_tokens), context_limit - max(spec.max_tokens) - 256)
    messages = [{"role": "user", "content": _filler(rng, prompt_tokens)}]
    headers = {AFFINITY_HEADER: session_id}
    for turn in range(rng.randint(*spec.turns)):
        max_tokens = rng.randint(*spec.max_tokens)
        if _estimate_tokens(messages) + max_tokens + 256 > context_limit:
            break  # the session has outgrown the engine's context window
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
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


async def run_rl_replay(
    gateway: str,
    model: str,
    *,
    dataset: str = "zhuzilin/gsm8k",
    split: str = "train",
    messages_key: str = "messages",
    batch_size: int = 64,          # prompts per step (slime rollout_batch_size)
    group_size: int = 8,           # GRPO samples per prompt (n_samples_per_prompt)
    max_tokens: int = 16,          # small on purpose: E2E ≈ RTT+queue+prefill ≈ TTFT
    temperature: float = 1.0,
    duration: float = 600.0,
    out_path: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Replay the RL rollout workload without a trainer: GRPO-shaped bursts of the
    training dataset's own prompts.

    Each "step" mirrors slime's rollout phase — ``batch_size`` prompts sampled from
    the dataset, ``group_size`` concurrent requests per prompt (identical messages,
    one affinity session per group), all ``batch_size*group_size`` fired at once,
    then a barrier before the next step (slime is step-synchronous). With a small
    ``max_tokens`` the measured latency isolates the routing-sensitive path (network
    + queue + prefill); the group structure supplies the natural GRPO prefix reuse.
    """
    import httpx
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    deadline = time.time() + duration
    step = 0
    # The whole burst must be genuinely concurrent — the default 100-connection
    # pool would serialize a 512-request step client-side and pollute latency.
    limits = httpx.Limits(max_connections=batch_size * group_size + 32)
    async with httpx.AsyncClient(timeout=3600.0, trust_env=False, limits=limits) as client:
        while time.time() < deadline:
            prompt_rows = [ds[i] for i in rng.sample(range(len(ds)), batch_size)]

            async def one(step: int, pidx: int, sample: int, messages: list[dict]) -> None:
                payload = {"model": model, "messages": messages,
                           "max_tokens": max_tokens, "temperature": temperature}
                headers = {AFFINITY_HEADER: f"s{step}-g{pidx}"}
                row = {"t": time.time(), "shape": "rl_replay", "step": step,
                       "session": f"s{step}-g{pidx}", "turn": sample, "retries_409": 0}
                data = await _post_with_retry(client, f"{gateway}/v1/chat/completions", payload, headers, row)
                if data is not None:
                    row.update(
                        wv_start=data.get("weight_version_start"),
                        wv_end=data.get("weight_version_end"),
                        straddled=data.get("weight_version_start") != data.get("weight_version_end"),
                    )
                rows.append(row)

            await asyncio.gather(*(
                one(step, pidx, sample, list(pr[messages_key]))
                for pidx, pr in enumerate(prompt_rows)
                for sample in range(group_size)
            ))
            step += 1
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"steps": step, **summarize(rows)}


async def run_bfcl_replay(
    gateway: str,
    model: str,
    *,
    trajectories_path: str,      # bfcl_prep.py output: {episode_id, tools, steps[]} per line
    concurrency: int = 48,       # episodes in flight (each runs its steps sequentially)
    max_tokens: int = 256,       # one tool call per step; bounded, TTFT-dominated
    temperature: float = 0.0,
    duration: float = 600.0,
    out_path: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Teacher-forced BFCL multi-turn replay: workers run episodes end-to-end,
    sending each step's ground-truth prefix (+ OpenAI ``tools``) and advancing on
    ground truth regardless of the model's reply. Steps within an episode are
    strictly sequential, so trajectory wall time = sum of per-step latencies —
    the compounding quantity a routing policy can shrink. The workload is
    deterministic, so episodes pair exactly across routing-policy arms.
    """
    import httpx

    episodes = [json.loads(line) for line in Path(trajectories_path).read_text().splitlines() if line]
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    deadline = time.time() + duration
    counter = {"instance": 0}

    async def worker(worker_id: int, client: httpx.AsyncClient) -> None:
        while time.time() < deadline:
            ep = episodes[rng.randrange(len(episodes))]
            counter["instance"] += 1
            instance = counter["instance"]
            session = f"ep-{ep['episode_id']}-{instance}"
            t_start = time.time()
            completed = 0
            for k, messages in enumerate(ep["steps"]):
                if time.time() > deadline:
                    break
                payload = {
                    "model": model, "messages": messages, "tools": ep["tools"],
                    "max_tokens": max_tokens, "temperature": temperature,
                }
                row = {"t": time.time(), "shape": "bfcl", "episode": ep["episode_id"],
                       "instance": instance, "step": k, "total_steps": len(ep["steps"]),
                       "session": session, "retries_409": 0}
                data = await _post_with_retry(
                    client, f"{gateway}/v1/chat/completions", payload, {AFFINITY_HEADER: session}, row
                )
                rows.append(row)
                if data is None:
                    break  # trajectory dies with its failed step
                completed += 1
                row.update(
                    wv_start=data.get("weight_version_start"),
                    wv_end=data.get("weight_version_end"),
                )
            rows.append({
                "t": time.time(), "shape": "bfcl_trajectory", "episode": ep["episode_id"],
                "instance": instance, "steps_completed": completed,
                "total_steps": len(ep["steps"]),
                "trajectory_seconds": time.time() - t_start,
                "complete": completed == len(ep["steps"]),
            })

    def flush() -> None:
        if out_path:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))

    async def flusher() -> None:
        # Periodic flush so a preempted container loses ≤2 min of rows (the
        # restarted container writes a fresh uniquely-named file; see app.py).
        while time.time() < deadline:
            await asyncio.sleep(120)
            flush()

    limits = httpx.Limits(max_connections=concurrency + 16)
    async with httpx.AsyncClient(timeout=3600.0, trust_env=False, limits=limits) as client:
        await asyncio.gather(flusher(), *(worker(i, client) for i in range(concurrency)))
    flush()
    trajectories = [r for r in rows if r["shape"] == "bfcl_trajectory" and r["complete"]]
    lat = sorted(r["trajectory_seconds"] for r in trajectories)
    return {
        "steps": sum(1 for r in rows if r["shape"] == "bfcl"),
        "trajectories_complete": len(trajectories),
        "trajectory_p50": lat[len(lat) // 2] if lat else None,
        "trajectory_p95": lat[int(len(lat) * 0.95)] if lat else None,
        **summarize([r for r in rows if r["shape"] == "bfcl"]),
    }


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
