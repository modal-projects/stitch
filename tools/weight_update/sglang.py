"""SGLang protocol and correctness checks for weight-update validation."""

from __future__ import annotations

import hashlib
import os
import signal
import struct
import threading
import time
from pathlib import Path
from typing import Any


def _post(
    client: Any,
    url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    response = client.post(f"{url}{path}", json=payload, timeout=timeout)
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"{path} returned HTTP {response.status_code}: {response.text[:500]}"
        ) from exc
    if response.status_code != 200 or body.get("success") is False:
        raise RuntimeError(f"{path} failed with HTTP {response.status_code}: {body}")
    return body


def _generate(
    url: str,
    *,
    fingerprint: bool = False,
    fingerprint_logprobs: bool = True,
) -> dict[str, Any]:
    import httpx

    started = time.perf_counter()
    models_response = httpx.get(
        f"{url}/v1/models",
        timeout=30,
        trust_env=False,
    )
    models_response.raise_for_status()
    model = models_response.json()["data"][0]["id"]
    messages = [
        {
            "role": "user",
            "content": "Explain in exactly three short clauses why the Moon has phases.",
        }
    ]
    if not fingerprint or not fingerprint_logprobs:
        request = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 96 if fingerprint else 80,
        }
        if fingerprint:
            # Compare one deterministic DP worker rather than treating normal
            # cross-rank numerical drift as a weight-update failure.
            request["routed_dp_rank"] = 0
        response = httpx.post(
            f"{url}/v1/chat/completions",
            json=request,
            timeout=300,
            trust_env=False,
        )
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
        if len(text.split()) < 5 or sum(character.isalpha() for character in text) < 20:
            raise RuntimeError(f"completion was not plausibly fluent: {text!r}")
        result = {
            "wall_s": round(time.perf_counter() - started, 6),
            "text": text,
            "weight_version": (body.get("metadata") or {}).get("weight_version"),
        }
        if fingerprint:
            result["output_ids"] = []
            result["output_logprobs"] = []
        return result

    response = httpx.post(
        f"{url}/generate",
        json={
            "text": "Explain why the Moon has phases in exactly three short clauses.",
            "routed_dp_rank": 0,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 48 if fingerprint else 80,
                "ignore_eos": fingerprint,
            },
            "return_logprob": fingerprint and fingerprint_logprobs,
            "return_text_in_logprobs": False,
            "top_logprobs_num": 1 if fingerprint and fingerprint_logprobs else 0,
            "logprob_start_len": -1,
            "stream": False,
        },
        timeout=300,
        trust_env=False,
    )
    response.raise_for_status()
    body = response.json()
    text = body.get("text") or ""
    result = {
        "wall_s": round(time.perf_counter() - started, 6),
        "text": text,
        "weight_version": (body.get("meta_info") or {}).get("weight_version"),
    }
    if fingerprint:
        raw_logprobs = (body.get("meta_info") or {}).get("output_token_logprobs") or []
        result["output_ids"] = body.get("output_ids") or []
        result["output_logprobs"] = [
            float(item[0] if isinstance(item, (list, tuple)) else item)
            for item in raw_logprobs
        ]
        if fingerprint_logprobs and not result["output_ids"]:
            raise RuntimeError("fingerprint token IDs are incomplete")
        if fingerprint_logprobs and len(result["output_ids"]) != len(
            result["output_logprobs"]
        ):
            raise RuntimeError("fingerprint logprobs are incomplete")
    return result


def _weight_checksums(client: Any, url: str) -> dict[str, Any]:
    result = _post(client, url, "/weights_checker", {"action": "checksum"})
    ranks = result.get("ranks") or []
    if not ranks or not result.get("per_engine_checksum"):
        raise RuntimeError(f"weight checksum response is incomplete: {result}")
    return result


def _flush_inference_cache(url: str) -> None:
    """Give each exact fingerprint the same empty-cache execution state."""

    import httpx

    response = httpx.post(
        f"{url}/flush_cache",
        timeout=300,
        trust_env=False,
    )
    response.raise_for_status()
    if not response.text.startswith("Cache flushed."):
        raise RuntimeError(f"SGLang did not flush its inference cache: {response.text}")


def _fingerprint(
    url: str,
    *,
    fingerprint_logprobs: bool = True,
) -> dict[str, Any]:
    _flush_inference_cache(url)
    return _generate(
        url,
        fingerprint=True,
        fingerprint_logprobs=fingerprint_logprobs,
    )


def _weight_checksum_differences(
    live: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Describe exact per-rank tensor differences without retaining full maps."""

    live_ranks = live["ranks"]
    reference_ranks = reference["ranks"]
    rank_differences = []
    for rank_index in range(max(len(live_ranks), len(reference_ranks))):
        live_rank = live_ranks[rank_index] if rank_index < len(live_ranks) else None
        reference_rank = (
            reference_ranks[rank_index] if rank_index < len(reference_ranks) else None
        )
        if live_rank is None or reference_rank is None:
            rank_differences.append(
                {
                    "rank_index": rank_index,
                    "live_present": live_rank is not None,
                    "reference_present": reference_rank is not None,
                }
            )
            continue

        live_checksums = live_rank.get("checksums") or {}
        reference_checksums = reference_rank.get("checksums") or {}
        differing_names = sorted(
            name
            for name in live_checksums.keys() | reference_checksums.keys()
            if live_checksums.get(name) != reference_checksums.get(name)
        )
        if not differing_names:
            continue
        rank_differences.append(
            {
                "rank_index": rank_index,
                "parallelism_info": live_rank.get("parallelism_info"),
                "tensor_count": len(differing_names),
                "tensors": [
                    {
                        "name": name,
                        "live": live_checksums.get(name),
                        "reference": reference_checksums.get(name),
                    }
                    for name in differing_names
                ],
            }
        )
    return {
        "live_engine_checksum": live.get("per_engine_checksum"),
        "reference_engine_checksum": reference.get("per_engine_checksum"),
        "rank_count": len(rank_differences),
        "ranks": rank_differences,
    }


def _validate_weight_checksums(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    expected_ranks: int,
    require_draft_weights: bool,
) -> dict[str, Any]:
    before_ranks = before["ranks"]
    after_ranks = after["ranks"]
    if len(before_ranks) != expected_ranks or len(after_ranks) != expected_ranks:
        raise RuntimeError(
            "weight checksum rank count mismatch: "
            f"expected={expected_ranks} before={len(before_ranks)} "
            f"after={len(after_ranks)}"
        )
    if before["per_engine_checksum"] == after["per_engine_checksum"]:
        raise RuntimeError("live engine weight checksum did not change")

    before_draft = {
        (rank_index, name): checksum
        for rank_index, rank in enumerate(before_ranks)
        for name, checksum in rank["checksums"].items()
        if name.startswith("draft.")
    }
    after_draft = {
        (rank_index, name): checksum
        for rank_index, rank in enumerate(after_ranks)
        for name, checksum in rank["checksums"].items()
        if name.startswith("draft.")
    }
    if require_draft_weights and not before_draft:
        raise RuntimeError("speculative serving exposed no draft-weight checksums")
    if before_draft != after_draft:
        raise RuntimeError("speculative draft weights changed with target weights")
    return {
        "ranks": len(after_ranks),
        "per_engine_checksum": after["per_engine_checksum"],
        "draft_tensors": len(after_draft),
        "draft_unchanged": True,
    }


def _compare_fingerprints(
    fingerprints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Record deterministic agreement without treating kernel drift as corruption.

    Exact rank checksums establish weight equality. Greedy generation remains a
    useful behavioral probe, but parallel floating-point kernels can diverge at
    a near-tied token even when the weights are byte-identical.
    """

    first, second = fingerprints
    exact_token_ids = first["output_ids"] == second["output_ids"]
    exact_text = first["text"] == second["text"]
    mismatch = None
    if not exact_token_ids:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(
                        first["output_ids"],
                        second["output_ids"],
                        strict=False,
                    )
                )
                if left != right
            ),
            min(len(first["output_ids"]), len(second["output_ids"])),
        )
    max_logprob_difference = (
        max(
            (
                abs(left - right)
                for left, right in zip(
                    first["output_logprobs"],
                    second["output_logprobs"],
                    strict=True,
                )
            ),
            default=0.0,
        )
        if exact_token_ids
        and first["output_logprobs"]
        and len(first["output_logprobs"]) == len(second["output_logprobs"])
        else None
    )
    result = {
        "exact_text": exact_text,
        **_fingerprint_hashes(first),
    }
    if first["output_ids"]:
        result["exact_token_ids"] = exact_token_ids
        result["tokens"] = len(first["output_ids"])
    if mismatch is not None:
        result["first_token_mismatch"] = mismatch
        result["comparison_token_ids_sha256"] = _fingerprint_hashes(second).get(
            "token_ids_sha256"
        )
    if not exact_text:
        result["comparison_text_sha256"] = _fingerprint_hashes(second)["text_sha256"]
    if max_logprob_difference is not None:
        result["repeat_max_logprob_abs_diff"] = max_logprob_difference
    return result


def _fingerprint_hashes(fingerprint: dict[str, Any]) -> dict[str, str]:
    token_ids = fingerprint["output_ids"]
    output_logprobs = fingerprint["output_logprobs"]
    hashes = {"text_sha256": hashlib.sha256(fingerprint["text"].encode()).hexdigest()}
    if token_ids:
        hashes["token_ids_sha256"] = hashlib.sha256(
            struct.pack(f"<{len(token_ids)}q", *token_ids)
        ).hexdigest()
    if output_logprobs:
        hashes["logprobs_sha256"] = hashlib.sha256(
            struct.pack(f"<{len(output_logprobs)}d", *output_logprobs)
        ).hexdigest()
    return hashes


def _assert_target_changed(
    baseline: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    token_ids_changed = baseline["output_ids"] != target["output_ids"]
    text_changed = baseline["text"] != target["text"]
    if baseline["output_logprobs"] and len(baseline["output_logprobs"]) == len(
        target["output_logprobs"]
    ):
        max_logprob_difference = max(
            (
                abs(left - right)
                for left, right in zip(
                    baseline["output_logprobs"],
                    target["output_logprobs"],
                    strict=True,
                )
            ),
            default=0.0,
        )
    else:
        max_logprob_difference = None
    logprobs_changed = (
        max_logprob_difference is not None and max_logprob_difference > 0.0
    )
    if not (token_ids_changed or text_changed or logprobs_changed):
        raise RuntimeError("post-update fingerprint is identical to the base model")
    return {
        "changed_from_base": True,
        "token_ids_changed_from_base": token_ids_changed,
        "text_changed_from_base": text_changed,
        "max_logprob_abs_diff_from_base": max_logprob_difference,
    }


def _scheduler_processes() -> list[tuple[int, str]]:
    processes = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            title = (entry / "cmdline").read_bytes().split(b"\0", 1)[0].decode()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if title.startswith("sglang::scheduler"):
            processes.append((int(entry.name), title))
    return sorted(processes, key=lambda item: item[1])


def _assert_post_mutation_failure_is_fatal(
    url: str,
    *,
    target_version: int,
    expected_ranks: int,
    injection_delay_s: float,
    settle_s: float,
) -> dict[str, Any]:
    """Kill one scheduler during commit and require the engine to fail closed."""

    import httpx

    schedulers = _scheduler_processes()
    if len(schedulers) != expected_ranks:
        raise RuntimeError(
            "scheduler process count mismatch before fault injection: "
            f"expected={expected_ranks} observed={schedulers}"
        )
    victim_pid, victim_title = schedulers[-1]
    outcome: dict[str, Any] = {}
    request_finished = threading.Event()

    def commit() -> None:
        try:
            with httpx.Client(timeout=300, trust_env=False) as client:
                outcome["response"] = _post(
                    client,
                    url,
                    "/commit_weight_update",
                    {
                        "target_version": target_version,
                        "abort_all_requests": False,
                        "torch_empty_cache": False,
                        "flush_cache": False,
                    },
                )
        except Exception as exc:  # noqa: BLE001 - record the terminal RPC result
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            request_finished.set()

    commit_thread = threading.Thread(
        target=commit,
        name="post-mutation-failure-commit",
        daemon=True,
    )
    commit_thread.start()
    # The collective preflight is millisecond-scale. Wait into the physical
    # disk reload or H2D copy, then freeze one rank during live mutation.
    if request_finished.wait(injection_delay_s):
        raise RuntimeError("weight commit completed before fault injection")

    victim_stopped = False
    try:
        os.kill(victim_pid, signal.SIGSTOP)
        victim_stopped = True
        # Leave the peer ranks inside the mutation/publication window before
        # making the stopped participant terminal.
        if request_finished.wait(settle_s):
            raise RuntimeError("weight commit completed with a scheduler stopped")
        os.kill(victim_pid, signal.SIGKILL)
        victim_stopped = False
    finally:
        if victim_stopped:
            try:
                os.kill(victim_pid, signal.SIGCONT)
            except ProcessLookupError:
                pass

    commit_thread.join(timeout=180)
    if commit_thread.is_alive():
        raise RuntimeError("commit RPC did not terminate after a scheduler died")
    if "response" in outcome:
        raise RuntimeError("partial distributed commit reported success")
    if "error" not in outcome:
        raise RuntimeError("partial distributed commit produced no terminal result")

    health_observation = None
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{url}/health", timeout=3, trust_env=False)
            health_observation = f"HTTP {response.status_code}"
            if response.status_code != 200:
                break
        except Exception as exc:  # noqa: BLE001 - unavailability is expected
            health_observation = f"{type(exc).__name__}: {exc}"
            break
        time.sleep(1)
    else:
        raise RuntimeError("engine remained healthy after a partial live commit")

    try:
        generation = _generate(url)
    except Exception as exc:  # noqa: BLE001 - generation must fail closed
        generation_error = f"{type(exc).__name__}: {exc}"
    else:
        raise RuntimeError(
            f"engine served generation after a partial live commit: {generation}"
        )
    return {
        "target_version": target_version,
        "victim_pid": victim_pid,
        "victim_process": victim_title,
        "injection_delay_s": injection_delay_s,
        "settle_s": settle_s,
        "commit_error": outcome["error"],
        "health_after_failure": health_observation,
        "generation_error": generation_error,
        "failed_closed": True,
    }


def _fingerprint_after_commit(
    url: str,
    *,
    target_version: int,
    previous_fingerprint: dict[str, Any],
    fingerprint_logprobs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation = _generate(url)
    fingerprints = [
        _fingerprint(url, fingerprint_logprobs=fingerprint_logprobs) for _ in range(2)
    ]
    observed_versions = {
        generation["weight_version"],
        *(fingerprint["weight_version"] for fingerprint in fingerprints),
    }
    if observed_versions != {str(target_version)}:
        raise RuntimeError(
            "post-update generation reported unexpected weight versions: "
            f"{sorted(str(value) for value in observed_versions)}"
        )
    correctness = {
        **_compare_fingerprints(fingerprints),
        **_assert_target_changed(previous_fingerprint, fingerprints[0]),
    }
    return fingerprints[0], {
        "generation_after": generation,
        "fingerprint_probes": [
            {
                "wall_s": fingerprint["wall_s"],
                "weight_version": fingerprint["weight_version"],
                **_fingerprint_hashes(fingerprint),
            }
            for fingerprint in fingerprints
        ],
        "correctness": correctness,
    }
