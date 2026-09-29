"""Compare rollout numerics by scoring identical tokens on two engines."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Sequence
from typing import Any

READY_TIMEOUT = 30 * 60


async def compare(
    reference_gateway: str,
    candidate_gateway: str,
    model: str,
    *,
    samples: int = 16,
    output_tokens: int = 256,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate on ``candidate`` and teacher-force its tokens on both engines."""
    import httpx

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=3600.0, trust_env=False) as client:
        await asyncio.gather(
            _wait_for_ready(client, reference_gateway),
            _wait_for_ready(client, candidate_gateway),
        )
        for index in range(samples):
            prompt_ids = await _tokenize(
                client,
                candidate_gateway,
                model,
                _prompt(random.Random(seed + index)),
            )
            output_ids, decode_logprobs = await _generate(
                client,
                candidate_gateway,
                prompt_ids,
                output_tokens=output_tokens,
                seed=seed + index,
            )
            tokens = [*prompt_ids, *output_ids]
            candidate_logprobs, reference_logprobs = await asyncio.gather(
                _score(client, candidate_gateway, tokens, len(prompt_ids)),
                _score(client, reference_gateway, tokens, len(prompt_ids)),
            )
            rows.append(
                {
                    "sample": index,
                    "tokens": len(output_ids),
                    "candidate_decode_vs_reference": _pair_metrics(
                        decode_logprobs, reference_logprobs
                    ),
                    "candidate_prefill_vs_reference": _pair_metrics(
                        candidate_logprobs, reference_logprobs
                    ),
                    "candidate_decode_vs_prefill": _pair_metrics(
                        decode_logprobs, candidate_logprobs
                    ),
                }
            )
    return {"samples": rows, "summary": summarize(rows)}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(rows),
        "tokens": sum(row["tokens"] for row in rows),
    }
    for name in (
        "candidate_decode_vs_reference",
        "candidate_prefill_vs_reference",
        "candidate_decode_vs_prefill",
    ):
        result[name] = {
            "mean_abs_diff": _mean([row[name]["mean_abs_diff"] for row in rows]),
            "p95_abs_diff": _percentile(
                [row[name]["p95_abs_diff"] for row in rows], 0.95
            ),
            "max_abs_diff": max(
                (row[name]["max_abs_diff"] for row in rows), default=0.0
            ),
            "mean_token_ess": _mean([row[name]["token_ess"] for row in rows]),
        }
    return result


async def _wait_for_ready(client: Any, gateway: str) -> None:
    deadline = asyncio.get_running_loop().time() + READY_TIMEOUT
    while True:
        try:
            response = await client.get(f"{gateway}/server_info", timeout=10.0)
            if response.status_code == 200 and response.json().get("ready") is True:
                return
        except Exception:  # noqa: BLE001 - readiness is retried to the deadline
            pass
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"rollout pool did not become ready: {gateway}")
        await asyncio.sleep(1.0)


def _prompt(rng: random.Random) -> list[dict[str, str]]:
    values = [rng.randint(1, 10_000) for _ in range(64)]
    return [
        {
            "role": "user",
            "content": (
                "Analyze this integer sequence, explain two visible patterns, and write "
                f"a Python function that validates them:\n{values}"
            ),
        }
    ]


async def _tokenize(
    client: Any, gateway: str, model: str, messages: list[dict[str, str]]
) -> list[int]:
    response = await client.post(
        f"{gateway}/tokenize", json={"model": model, "messages": messages}
    )
    response.raise_for_status()
    return response.json()["tokens"]


async def _generate(
    client: Any,
    gateway: str,
    prompt_ids: list[int],
    *,
    output_tokens: int,
    seed: int,
) -> tuple[list[int], list[float]]:
    response = await client.post(
        f"{gateway}/generate",
        json={
            "input_ids": prompt_ids,
            "sampling_params": {
                "max_new_tokens": output_tokens,
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 1024,
                "sampling_seed": seed,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        },
    )
    response.raise_for_status()
    body = response.json()
    items = body["meta_info"]["output_token_logprobs"]
    output_ids = [int(item[1]) for item in items]
    return output_ids, [float(item[0]) for item in items]


async def _score(
    client: Any, gateway: str, tokens: list[int], prompt_length: int
) -> list[float]:
    response = await client.post(
        f"{gateway}/generate",
        json={
            "input_ids": tokens,
            "sampling_params": {
                "max_new_tokens": 0,
                "temperature": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": prompt_length - 1,
        },
    )
    response.raise_for_status()
    items = response.json()["meta_info"]["input_token_logprobs"]
    tail = items[-(len(tokens) - prompt_length) :]
    expected = tokens[prompt_length:]
    actual = [int(item[1]) for item in tail]
    if actual != expected:
        raise ValueError("teacher-forced token alignment mismatch")
    return [float(item[0]) for item in tail]


def _pair_metrics(
    candidate: Sequence[float], reference: Sequence[float]
) -> dict[str, float]:
    if len(candidate) != len(reference):
        raise ValueError("logprob vectors have different lengths")
    deltas = [ref - cand for cand, ref in zip(candidate, reference, strict=True)]
    absolute = [abs(delta) for delta in deltas]
    ratios = [math.exp(max(-40.0, min(40.0, delta))) for delta in deltas]
    ratio_sum = sum(ratios)
    ess = ratio_sum * ratio_sum / (len(ratios) * sum(ratio * ratio for ratio in ratios))
    return {
        "mean_diff": _mean(deltas),
        "mean_abs_diff": _mean(absolute),
        "p50_abs_diff": _percentile(absolute, 0.50),
        "p95_abs_diff": _percentile(absolute, 0.95),
        "p99_abs_diff": _percentile(absolute, 0.99),
        "max_abs_diff": max(absolute, default=0.0),
        "token_ess": ess,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]
