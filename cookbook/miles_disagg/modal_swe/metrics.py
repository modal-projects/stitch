"""Aggregate rollout metrics for Modal repository-repair agents.

The fully-async scheduler reports only scheduling and staleness. This hook
aggregates environment/tool/verifier fields owned by the Modal adapter.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miles.utils.types import Sample

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


def _training_batch_composition_metrics(
    samples: list[Sample], output: dict[str, Any]
) -> None:
    output["rollout/training_batch/sample_count"] = len(samples)
    sources = Counter(
        str(sample.metadata.get("rollout_source") or "unknown") for sample in samples
    )
    for source, count in sources.items():
        safe_source = source.replace("/", "_").replace(" ", "_")
        prefix = f"rollout/training_batch/{safe_source}"
        output[f"{prefix}/sample_count"] = count
        output[f"{prefix}/sample_percentage"] = 100 * count / len(samples)


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


def _weight_version_range(sample: Sample) -> tuple[int | None, int | None]:
    """Oldest and newest numeric generation versions, as Miles' staleness reads them."""
    spans = getattr(sample, "all_weight_version_spans", None) or []
    versions = [int(span.version) for span in spans if str(span.version).isdigit()]
    return (min(versions), max(versions)) if versions else (None, None)


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
    group staleness: the drain's version minus the sample's oldest (or, for
    post-generation, newest) generation version. Miles filters aborted
    trajectories before this hook, so ``infra_error_ratio`` counts only the
    failures that still reach training.
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

    versions = [_weight_version_range(sample) for sample in samples]
    # Miles reports max_staleness = drain version - oldest version in the batch.
    max_staleness = _number(output.get("rollout/fully_async/max_staleness"))
    oldest = [low for low, _ in versions if low is not None]
    current_version = (
        max_staleness + min(oldest) if max_staleness is not None and oldest else None
    )

    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sample, reward, (low, high) in zip(samples, rewards, versions, strict=True):
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
            "response_length_mean": _number(getattr(sample, "response_length", None)),
            "turns_mean": _number(agent_metrics.get("turns")),
            "total_time_mean": _number(agent_metrics.get("total_time")),
            "staleness_mean": (
                current_version - low
                if current_version is not None and low is not None
                else None
            ),
            "post_generation_staleness_mean": (
                current_version - high
                if current_version is not None and high is not None
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
    _training_batch_composition_metrics(samples, output)
    _per_source_metrics(samples, output, reward_key=reward_key)

    known_statuses = {
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
    statuses = Counter(
        status if status in known_statuses else "other"
        for sample in samples
        for status in [str(sample.metadata.get("exit_status", "unknown"))]
    )
    for status, count in statuses.items():
        safe_status = status.replace("/", "_").replace(" ", "_")
        output[f"rollout_agent/exit_status/{safe_status}_ratio"] = count / len(samples)

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


def log_rollout_data(
    rollout_id: int,
    args,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Extend the standard Miles rollout log; returning False preserves it."""
    del rollout_id, rollout_time
    if rollout_extra_metrics is not None and samples:
        add_metrics(
            samples,
            rollout_extra_metrics,
            reward_key=getattr(args, "reward_key", None),
        )
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
