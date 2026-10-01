"""Aggregate rollout metrics for Modal repository-repair agents.

The fully-async scheduler reports only scheduling and staleness. This hook
aggregates environment/tool/verifier fields owned by the Modal adapter.
"""

from __future__ import annotations

import gzip
import heapq
import json
import logging
import os
import random
from array import array
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_KNOWN_EXIT_STATUSES = {
    "Submitted",
    "LimitsExceeded",
    "TimeExceeded",
    "RepeatedFormatError",
    "FormatError",
    "UserInterruption",
    "completed",
    "command_timeout",
    "verifier_timeout",
    "verifier_infra_error",
    "sandbox_not_found",
    "agent_error",
    "sandbox_infra_error",
    "session_record_timeout",
    "session_record_request_error",
    "session_sample_collection_error",
    "session_create_error",
    "agent_function_exception",
    "no_model_calls",
    "prompt_exceeds_max_seq_len",
    "unknown",
}
# The agent never produced a parseable action; a format collapse shows up here first.
_FORMAT_ERROR_STATUSES = {"FormatError", "RepeatedFormatError"}
# Every step keeps a readable slice of the batch: its best and worst trajectories and
# a few whole GRPO groups, every attempt at one task side by side.
_SAMPLE_DUMP_PER_END = 2
_SAMPLE_DUMP_GROUPS = 2
# Whole-batch dumps are converted and written off the drain path, one at a time.
_BATCH_WRITER = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="rollout-batch-dump"
)
_BATCH_LAYOUT = {
    "trajectories.jsonl.gz": "one record per sample, in batch order",
    "tokens.int32": "every sample's full token sequence, concatenated; sample i has num_tokens[i]",
    "rollout_log_probs.float32": "sampler log-probs over each sample's last response_length tokens",
    "loss_mask.uint8": "loss mask over the same response tokens",
    "byte_order": "little-endian",
}

_AGENT_MEAN_ONLY_METRICS = {
    "agent_tool_input_over_64k_count",
    "agent_tool_input_over_64k_ratio",
    "agent_tool_output_hard_limit_count",
    "agent_tool_output_hard_limit_ratio",
    "agent_tool_output_truncated_count",
    "agent_tool_output_truncated_ratio",
    "context_limit_exceeded",
    "generation_bound",
    "infra_error",
    "policy_failure",
    "model_request_count",
    "tool_calls",
    "tool_timeout_count",
    "turns",
    "verifier_reward_missing",
    "verifier_timeout",
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def _summary(metrics: dict[str, Any], prefix: str, values: list[float]) -> None:
    if not values:
        return
    metrics[f"{prefix}_mean"] = sum(values) / len(values)
    metrics[f"{prefix}_p90"] = _percentile(values, 0.90)
    metrics[f"{prefix}_max"] = max(values)


def _numeric_agent_metrics(samples: list[Sample], output: dict[str, Any]) -> None:
    agent_metrics = [sample.metadata.get("agent_metrics") or {} for sample in samples]
    keys = sorted(
        {
            key
            for metrics in agent_metrics
            for key, value in metrics.items()
            if key != "agent_worker_index"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    for key in keys:
        values = [
            float(metrics[key])
            for metrics in agent_metrics
            if isinstance(metrics.get(key), (int, float))
        ]
        if key in _AGENT_MEAN_ONLY_METRICS:
            output[f"rollout_agent/{key}_mean"] = sum(values) / len(values)
        else:
            _summary(output, f"rollout_agent/{key}", values)


def _summarize_sample_metadata(
    samples: list[Sample],
    output: dict[str, Any],
    *,
    metadata_prefix: str,
    metric_prefix: str,
) -> None:
    keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.metadata.items()
            if key.startswith(metadata_prefix) and isinstance(value, (int, float))
        }
    )
    for key in keys:
        values = [
            float(sample.metadata[key])
            for sample in samples
            if isinstance(sample.metadata.get(key), (int, float))
        ]
        _summary(
            output,
            f"{metric_prefix}/{key.removeprefix(metadata_prefix)}",
            values,
        )


def _lifecycle_segments(sample: Sample) -> list[Any]:
    lifecycle = sample.metadata.get("lifecycle", [])
    return lifecycle if isinstance(lifecycle, list) else [lifecycle]


def _backend_seconds(segments: list[Any]) -> list[float]:
    return [
        float(segment["t1"] - segment["t0"])
        for segment in segments
        if isinstance(segment, dict)
        and isinstance(segment.get("t0"), (int, float))
        and isinstance(segment.get("t1"), (int, float))
        and segment["t1"] >= segment["t0"]
    ]


def _request_metrics(samples: list[Sample], output: dict[str, Any]) -> None:
    lifecycle_segments = [
        segment for sample in samples for segment in _lifecycle_segments(sample)
    ]
    backend = _backend_seconds(lifecycle_segments)
    server = [
        float(segment["t1"] - segment["req_ts"])
        for segment in lifecycle_segments
        if isinstance(segment, dict)
        and isinstance(segment.get("req_ts"), (int, float))
        and isinstance(segment.get("t1"), (int, float))
        and segment["t1"] >= segment["req_ts"]
    ]
    pre_backend = [
        float(segment["t0"] - segment["req_ts"])
        for segment in lifecycle_segments
        if isinstance(segment, dict)
        and isinstance(segment.get("req_ts"), (int, float))
        and isinstance(segment.get("t0"), (int, float))
        and segment["t0"] >= segment["req_ts"]
    ]
    client = [
        float(duration)
        for sample in samples
        for duration in (sample.metadata.get("agent_metrics") or {}).get(
            "client_model_request_durations_seconds",
            [],
        )
        if isinstance(duration, (int, float))
    ]
    _summary(output, "rollout_model/request_latency_seconds", backend)
    _summary(output, "rollout_model/server_request_latency_seconds", server)
    _summary(output, "rollout_model/pre_backend_latency_seconds", pre_backend)
    _summary(output, "rollout_model/client_request_latency_seconds", client)

    backend_seconds = sum(backend)
    server_seconds = sum(server)
    pre_backend_seconds = sum(pre_backend)
    client_seconds = sum(client)
    trainable_tokens = sum(sample.effective_response_length for sample in samples)
    unrecorded_requests = max(0, len(client) - len(backend))
    output.update(
        {
            "rollout_model/request_count": len(backend),
            "rollout_model/request_total_seconds": backend_seconds,
            "rollout_model/server_request_count": len(server),
            "rollout_model/server_request_total_seconds": server_seconds,
            "rollout_model/pre_backend_count": len(pre_backend),
            "rollout_model/pre_backend_total_seconds": pre_backend_seconds,
            "rollout_model/client_request_count": len(client),
            "rollout_model/client_request_total_seconds": client_seconds,
            "rollout_model/client_minus_backend_seconds_signed": client_seconds
            - backend_seconds,
            "rollout_model/client_minus_backend_request_count": len(client)
            - len(backend),
            "rollout_model/trainable_completion_tokens": trainable_tokens,
            "rollout_model/trainable_tokens_per_backend_request_second": (
                trainable_tokens / backend_seconds if backend_seconds else 0.0
            ),
            # Session lifecycle segments exist only for successful responses;
            # the mini-SWE client has retries disabled, so this is the number
            # of client calls that did not become a recorded completion.
            "rollout_model/unrecorded_request_count": unrecorded_requests,
        }
    )


def _routing_replay_metrics(samples: list[Sample], output: dict[str, Any]) -> None:
    arrays = [
        routed_experts
        for sample in samples
        if (routed_experts := sample.rollout_routed_experts) is not None
    ]
    if not arrays:
        return

    rows = [len(routed_experts) for routed_experts in arrays]
    raw_bytes = [routed_experts.nbytes for routed_experts in arrays]
    _summary(output, "rollout_r3/rows_per_sample", rows)
    _summary(
        output, "rollout_r3/raw_mib_per_sample", [size / 2**20 for size in raw_bytes]
    )
    output.update(
        {
            "rollout_r3/sample_count": len(arrays),
            "rollout_r3/rows_total": sum(rows),
            "rollout_r3/raw_bytes_total": sum(raw_bytes),
            "rollout_r3/raw_bytes_per_row": (
                sum(raw_bytes) / sum(rows) if sum(rows) else 0.0
            ),
        }
    )


def _exit_status(sample: Sample) -> str:
    status = str(sample.metadata.get("exit_status", "unknown"))
    return status if status in _KNOWN_EXIT_STATUSES else "other"


def _metric_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _raw_reward(sample: Sample, reward_key: str | None) -> float | None:
    """The reward Miles averages into ``episode_raw_reward``."""
    if "raw_reward" in sample.metadata:
        return _number(sample.metadata["raw_reward"])
    reward = getattr(sample, "reward", None)
    if reward_key and isinstance(reward, dict):
        reward = reward.get(reward_key)
    return _number(reward)


def _oldest_weight_version(sample: Sample) -> int | None:
    """Oldest numeric generation version, as Miles' staleness reads it."""
    spans = getattr(sample, "all_weight_version_spans", None) or []
    versions = [int(span.version) for span in spans if str(span.version).isdigit()]
    return min(versions) if versions else None


def _per_source_metrics(
    samples: list[Sample],
    output: dict[str, Any],
    *,
    reward_key: str | None = None,
) -> None:
    """Split the batch by serving pool and by weight view (precision).

    A prompt's samples land on different pools, so the within-prompt delta (a
    sample's reward minus its prompt's mean in this batch) separates a pool's
    effect from prompt difficulty. Staleness is the per-sample form of Miles'
    group staleness: the drain's version minus the sample's oldest generation
    version. Miles filters aborted trajectories before this hook, so
    ``infra_error_ratio`` counts only the failures that still reach training.
    """
    rewards = [_raw_reward(sample, reward_key) for sample in samples]
    prompt_rewards: dict[Any, list[float]] = defaultdict(list)
    for sample, reward in zip(samples, rewards, strict=True):
        group = getattr(sample, "group_index", None)
        if reward is not None and group is not None:
            prompt_rewards[group].append(reward)
    prompt_means = {
        group: sum(values) / len(values)
        for group, values in prompt_rewards.items()
        if len(values) >= 2
    }

    oldest = [_oldest_weight_version(sample) for sample in samples]
    # Miles reports max_staleness = drain version - oldest version in the batch.
    max_staleness = _number(output.get("rollout/fully_async/max_staleness"))
    versioned = [version for version in oldest if version is not None]
    current_version = (
        max_staleness + min(versioned)
        if max_staleness is not None and versioned
        else None
    )

    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample, reward, low in zip(samples, rewards, oldest, strict=True):
        agent_metrics = sample.metadata.get("agent_metrics") or {}
        source = str(sample.metadata.get("rollout_source") or "unknown")
        view = source.rsplit(":", 1)[1] if ":" in source else "unknown"
        group = getattr(sample, "group_index", None)
        completion_tokens = _number(agent_metrics.get("model_completion_tokens_total"))
        values = {
            "raw_reward_mean": reward,
            "within_prompt_reward_delta_mean": (
                reward - prompt_means[group]
                if reward is not None and group in prompt_means
                else None
            ),
            "infra_error_ratio": float(bool(agent_metrics.get("infra_error"))),
            "format_error_ratio": float(_exit_status(sample) in _FORMAT_ERROR_STATUSES),
            "response_length_mean": _number(getattr(sample, "response_length", None)),
            "turns_mean": _number(agent_metrics.get("turns")),
            "staleness_mean": (
                current_version - low
                if current_version is not None and low is not None
                else None
            ),
            # Paired per sample so the speed ratio never mixes in untimed tokens.
            "completion_tokens": completion_tokens,
            "backend_seconds": (
                sum(_backend_seconds(_lifecycle_segments(sample)))
                if completion_tokens is not None
                else None
            ),
        }
        for prefix in (
            f"rollout/by_source/{_metric_name(source)}",
            f"rollout/by_view/{_metric_name(view)}",
        ):
            for name, value in values.items():
                if value is not None:
                    groups[prefix][name].append(value)

    for prefix, metrics in groups.items():
        output[f"{prefix}/sample_count"] = len(metrics["infra_error_ratio"])
        for name, values in metrics.items():
            if name.endswith(("_mean", "_ratio")):
                output[f"{prefix}/{name}"] = sum(values) / len(values)
        # Pooled over the group's requests, so long episodes weigh by their time.
        seconds = sum(metrics["backend_seconds"])
        if metrics["completion_tokens"] and seconds > 0:
            output[f"{prefix}/completion_tokens_per_backend_request_second"] = (
                sum(metrics["completion_tokens"]) / seconds
            )


def add_metrics(
    samples: list[Sample],
    output: dict[str, Any],
    *,
    reward_key: str | None = None,
) -> None:
    if not samples:
        return
    _numeric_agent_metrics(samples, output)
    _summarize_sample_metadata(
        samples,
        output,
        metadata_prefix="session_collect/",
        metric_prefix="rollout_session",
    )
    _request_metrics(samples, output)
    _routing_replay_metrics(samples, output)
    _per_source_metrics(samples, output, reward_key=reward_key)

    statuses = Counter(_exit_status(sample) for sample in samples)
    for status, count in statuses.items():
        safe_status = status.replace("/", "_").replace(" ", "_")
        output[f"rollout_agent/exit_status/{safe_status}_ratio"] = count / len(samples)
    output["rollout_agent/format_error_ratio"] = sum(
        statuses[status] for status in _FORMAT_ERROR_STATUSES
    ) / len(samples)

    agent_metrics = [sample.metadata.get("agent_metrics") or {} for sample in samples]
    verifier_return_codes = [
        int(metrics["verifier_return_code"])
        for metrics in agent_metrics
        if isinstance(metrics.get("verifier_return_code"), (int, float))
    ]
    if verifier_return_codes:
        output["rollout_agent/verifier_nonzero_return_code_ratio"] = sum(
            code != 0 for code in verifier_return_codes
        ) / len(verifier_return_codes)
    context_limits = sum(
        bool(metrics.get("context_limit_exceeded")) for metrics in agent_metrics
    )
    output["rollout_agent/context_limit_exit_ratio"] = context_limits / len(samples)
    output["rollout_agent/step_limit_exit_ratio"] = max(
        0,
        statuses["LimitsExceeded"] - context_limits,
    ) / len(samples)


def _weight_versions(sample: Sample) -> list[int]:
    spans = getattr(sample, "all_weight_version_spans", None) or []
    return sorted({int(span.version) for span in spans if str(span.version).isdigit()})


def _trajectory(
    rollout_id: int, sample: Sample, reward_key: str | None, picks: list[str]
) -> dict[str, Any]:
    agent_metrics = sample.metadata.get("agent_metrics") or {}
    return {
        "rollout_id": rollout_id,
        "picks": picks,
        "index": getattr(sample, "index", None),
        "group_index": getattr(sample, "group_index", None),
        "rollout_source": sample.metadata.get("rollout_source"),
        "weight_versions": _weight_versions(sample),
        "reward": _raw_reward(sample, reward_key),
        "exit_status": sample.metadata.get("exit_status"),
        "turns": agent_metrics.get("turns"),
        "num_tokens": len(getattr(sample, "tokens", None) or ()),
        "response_length": getattr(sample, "response_length", None),
        "prompt": getattr(sample, "prompt", None),
        "response": getattr(sample, "response", None),
    }


def _write_atomically(path: Path, write) -> None:
    """Write through a temporary name, so a volume commit never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as handle:
        write(handle)
    os.replace(partial, path)


def _write_trajectories(path: Path, records: list[dict[str, Any]]) -> None:
    def write(handle) -> None:
        with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=1) as gz:
            for record in records:
                gz.write((json.dumps(record, default=str) + "\n").encode())

    _write_atomically(path, write)


def _step_picks(samples: list[Sample], rollout_id: int, reward_key: str | None):
    """Positions of the batch's best and worst trajectories and of a few whole groups."""
    picks: dict[int, list[str]] = defaultdict(list)
    scored = [
        (reward, position)
        for position, sample in enumerate(samples)
        if (reward := _raw_reward(sample, reward_key)) is not None
    ]
    for _, position in heapq.nlargest(_SAMPLE_DUMP_PER_END, scored):
        picks[position].append("best")
    for _, position in heapq.nsmallest(_SAMPLE_DUMP_PER_END, scored):
        picks[position].append("worst")
    groups: dict[Any, list[int]] = defaultdict(list)
    for position, sample in enumerate(samples):
        if (group := getattr(sample, "group_index", None)) is not None:
            groups[group].append(position)
    # Seeded by the step, so a replayed step keeps the same groups.
    chosen = random.Random(rollout_id).sample(
        sorted(groups), k=min(_SAMPLE_DUMP_GROUPS, len(groups))
    )
    for group in chosen:
        for position in groups[group]:
            picks[position].append("group")
    return picks


def _write_batch(directory: Path, records: list[dict], sequences: list[tuple]) -> None:
    """Write one whole batch: readable trajectories plus its token-level arrays."""
    try:
        _write_trajectories(directory / "trajectories.jsonl.gz", records)
        tokens, log_probs, loss_mask = array("i"), array("f"), array("B")
        for sample_tokens, response_length, sample_log_probs, sample_mask in sequences:
            tokens.extend(sample_tokens)
            log_probs.extend(
                sample_log_probs
                if sample_log_probs is not None
                and len(sample_log_probs) == response_length
                else [float("nan")] * response_length
            )
            loss_mask.extend(
                sample_mask if sample_mask is not None else [1] * response_length
            )
        for name, values in (
            ("tokens.int32", tokens),
            ("rollout_log_probs.float32", log_probs),
            ("loss_mask.uint8", loss_mask),
        ):
            _write_atomically(directory / name, values.tofile)
        _write_atomically(
            directory / "layout.json",
            lambda handle: handle.write(json.dumps(_BATCH_LAYOUT, indent=1).encode()),
        )
    except (OSError, OverflowError, TypeError, ValueError):
        logger.warning(
            "could not write the rollout batch to %s", directory, exc_info=True
        )


def _dump_rollout_history(
    rollout_id: int,
    args: Any,
    samples: list[Sample],
    *,
    reward_key: str | None = None,
) -> Future | None:
    """Save this step's slice of the rollout history next to the run's checkpoints.

    Every step writes ``rollout_samples/rollout_NNNNNN.jsonl.gz``: the best and worst
    trajectories and a few whole GRPO groups, in full. Every ``save_interval`` steps
    also writes the whole batch to ``rollout_batches/rollout_NNNNNN/``, readable and as
    token arrays. That batch trains right after a checkpoint, so the checkpoint
    rescores it exactly offline. The dumps go to the run directory on the run volume,
    which every trainer host commits with each weight publication. The app names it;
    otherwise it is the parent of Miles' ``--save``, which is node-local when saves
    are. A failed write only loses the dump, never the step.
    """
    save = getattr(args, "save", None)
    if not save:
        return None
    run_dir = Path(getattr(args, "stitch_run_dir", None) or Path(save).parent)
    picks = _step_picks(samples, rollout_id, reward_key)
    records = [
        _trajectory(rollout_id, samples[position], reward_key, labels)
        for position, labels in sorted(picks.items())
    ]
    try:
        if records:
            _write_trajectories(
                run_dir / "rollout_samples" / f"rollout_{rollout_id:06d}.jsonl.gz",
                records,
            )
    except OSError:
        logger.warning("could not write the rollout sample dump", exc_info=True)
    interval = getattr(args, "save_interval", None)
    if not interval or rollout_id % interval:
        return None
    # Snapshot references now; conversion and compression run off the drain path.
    batch = [
        _trajectory(rollout_id, sample, reward_key, ["batch"]) for sample in samples
    ]
    sequences = [
        (
            list(getattr(sample, "tokens", None) or ()),
            int(getattr(sample, "response_length", 0) or 0),
            getattr(sample, "rollout_log_probs", None),
            getattr(sample, "loss_mask", None),
        )
        for sample in samples
    ]
    directory = run_dir / "rollout_batches" / f"rollout_{rollout_id:06d}"
    return _BATCH_WRITER.submit(_write_batch, directory, batch, sequences)


def log_rollout_data(
    rollout_id: int,
    args,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Extend the standard Miles rollout log; returning False preserves it."""
    del rollout_time
    if rollout_extra_metrics is not None and samples:
        reward_key = getattr(args, "reward_key", None)
        add_metrics(samples, rollout_extra_metrics, reward_key=reward_key)
        _dump_rollout_history(rollout_id, args, samples, reward_key=reward_key)
        # Exact vectors are needed only for the aggregate above. Do not carry
        # hundreds of per-turn floats per trajectory into trainer object-store
        # payloads after their p50/p90/max/count totals have been recorded.
        for sample in samples:
            agent_metrics = sample.metadata.get("agent_metrics")
            if isinstance(agent_metrics, dict):
                agent_metrics.pop(
                    "client_model_request_durations_seconds",
                    None,
                )
    return False
