import json
from types import SimpleNamespace

import pytest

from cookbook.miles_disagg.modal_swe.metrics import (
    _per_source_metrics,
    _request_metrics,
    _routing_replay_metrics,
    _training_batch_composition_metrics,
    log_rollout_data,
)


def _sample(source=None, *, reward=None, group=None, versions=(), **fields):
    metadata = fields.pop("metadata", {})
    if source is not None:
        metadata["rollout_source"] = source
    return SimpleNamespace(
        metadata=metadata,
        reward=reward,
        group_index=group,
        all_weight_version_spans=[SimpleNamespace(version=v) for v in versions],
        **fields,
    )


def test_request_metrics_split_session_and_backend_latency():
    sample = SimpleNamespace(
        effective_response_length=50,
        metadata={
            "lifecycle": [
                {"req_ts": 10.0, "t0": 10.25, "t1": 11.25},
                {"req_ts": 20.0, "t0": 20.75, "t1": 22.75},
            ],
            "agent_metrics": {"client_model_request_durations_seconds": [1.5, 3.0]},
        },
    )
    output = {}

    _request_metrics([sample], output)

    assert output["rollout_model/request_total_seconds"] == 3.0
    assert output["rollout_model/server_request_total_seconds"] == 4.0
    assert output["rollout_model/pre_backend_total_seconds"] == 1.0
    assert output["rollout_model/client_request_total_seconds"] == 4.5
    assert output["rollout_model/server_request_latency_seconds_mean"] == 2.0
    assert output["rollout_model/pre_backend_latency_seconds_mean"] == 0.5
    assert output[
        "rollout_model/trainable_tokens_per_backend_request_second"
    ] == pytest.approx(50 / 3)


def test_request_metrics_accepts_old_lifecycle_without_server_timestamp():
    sample = SimpleNamespace(
        effective_response_length=4,
        metadata={
            "lifecycle": {"t0": 2.0, "t1": 3.0},
            "agent_metrics": {"client_model_request_durations_seconds": [1.25]},
        },
    )
    output = {}

    _request_metrics([sample], output)

    assert output["rollout_model/request_total_seconds"] == 1.0
    assert output["rollout_model/server_request_count"] == 0
    assert output["rollout_model/pre_backend_count"] == 0


def test_routing_replay_metrics_report_incremental_raw_volume():
    class Array:
        def __init__(self, rows: int, bytes_per_row: int) -> None:
            self.rows = rows
            self.nbytes = rows * bytes_per_row

        def __len__(self) -> int:
            return self.rows

    samples = [
        SimpleNamespace(rollout_routed_experts=Array(3, 2 * 4 * 4)),
        SimpleNamespace(rollout_routed_experts=Array(5, 2 * 4 * 4)),
        SimpleNamespace(rollout_routed_experts=None),
    ]
    output = {}

    _routing_replay_metrics(samples, output)

    assert output["rollout_r3/sample_count"] == 2
    assert output["rollout_r3/rows_total"] == 8
    assert output["rollout_r3/raw_bytes_total"] == 8 * 2 * 4 * 4
    assert output["rollout_r3/raw_bytes_per_row"] == 2 * 4 * 4


def test_training_batch_composition_metrics_count_final_samples():
    samples = [
        SimpleNamespace(metadata={"rollout_source": "ServerH100FP8:fp8"}),
        SimpleNamespace(metadata={"rollout_source": "ServerH100FP8:fp8"}),
        SimpleNamespace(metadata={"rollout_source": "ServerB300NVFP4W4A16:nvfp4"}),
        SimpleNamespace(metadata={}),
    ]
    output = {}

    _training_batch_composition_metrics(samples, output)

    assert output["rollout/training_batch/sample_count"] == 4
    assert output["rollout/training_batch/ServerH100FP8:fp8/sample_count"] == 2
    assert output["rollout/training_batch/ServerH100FP8:fp8/sample_percentage"] == 50.0
    assert output["rollout/training_batch/ServerB300NVFP4W4A16:nvfp4/sample_count"] == 1
    assert (
        output["rollout/training_batch/ServerB300NVFP4W4A16:nvfp4/sample_percentage"]
        == 25.0
    )
    assert output["rollout/training_batch/unknown/sample_count"] == 1
    assert output["rollout/training_batch/unknown/sample_percentage"] == 25.0


def test_per_source_metrics_group_by_pool_and_view():
    samples = [
        _sample("ServerH100FP8:fp8", reward=1.0, response_length=10),
        _sample("ServerH200FP8:fp8", reward=0.0, response_length=30),
        _sample("ServerB300NVFP4W4A16:nvfp4", reward=1.0, response_length=20),
        _sample(reward=0.0),
    ]
    output = {}

    _per_source_metrics(samples, output)

    assert output["rollout/by_source/ServerH100FP8:fp8/sample_count"] == 1
    assert output["rollout/by_source/ServerH200FP8:fp8/raw_reward_mean"] == 0.0
    assert output["rollout/by_view/fp8/sample_count"] == 2
    assert output["rollout/by_view/fp8/raw_reward_mean"] == 0.5
    assert output["rollout/by_view/fp8/response_length_mean"] == 20.0
    assert output["rollout/by_view/nvfp4/raw_reward_mean"] == 1.0
    assert output["rollout/by_source/unknown/sample_count"] == 1
    assert output["rollout/by_view/unknown/sample_count"] == 1


def test_per_source_metrics_within_prompt_reward_delta():
    samples = [
        # Prompt 0: mean 1/3.
        _sample("A:fp8", reward=1.0, group=0),
        _sample("B:nvfp4", reward=0.0, group=0),
        _sample("C:bf16", reward=0.0, group=0),
        # Prompt 1: mean 1/2.
        _sample("A:fp8", reward=0.0, group=1),
        _sample("B:nvfp4", reward=1.0, group=1),
        # A prompt with one member in the batch has no delta.
        _sample("A:fp8", reward=1.0, group=2),
    ]
    output = {}

    _per_source_metrics(samples, output)

    assert output["rollout/by_view/fp8/raw_reward_mean"] == pytest.approx(2 / 3)
    assert output["rollout/by_view/fp8/within_prompt_reward_delta_mean"] == (
        pytest.approx((2 / 3 - 1 / 2) / 2)
    )
    assert output["rollout/by_view/nvfp4/within_prompt_reward_delta_mean"] == (
        pytest.approx((-1 / 3 + 1 / 2) / 2)
    )
    assert output["rollout/by_view/bf16/within_prompt_reward_delta_mean"] == (
        pytest.approx(-1 / 3)
    )


def test_per_source_metrics_pool_request_speed_over_timed_samples():
    samples = [
        _sample(
            "A:fp8",
            metadata={
                "agent_metrics": {"model_completion_tokens_total": 100},
                "lifecycle": [{"t0": 0.0, "t1": 1.0}, {"t0": 5.0, "t1": 6.0}],
            },
        ),
        _sample(
            "A:fp8",
            metadata={
                "agent_metrics": {"model_completion_tokens_total": 300},
                "lifecycle": {"t0": 0.0, "t1": 4.0},
            },
        ),
        # Timed but without token usage: excluded from both sides of the ratio.
        _sample("A:fp8", metadata={"lifecycle": {"t0": 0.0, "t1": 10.0}}),
    ]
    output = {}

    _per_source_metrics(samples, output)

    assert output[
        "rollout/by_view/fp8/completion_tokens_per_backend_request_second"
    ] == pytest.approx(400 / 6)


def test_per_source_metrics_staleness_uses_the_drain_version():
    samples = [
        _sample("A:fp8", versions=("3", "5", "default")),
        _sample("B:nvfp4", versions=("6", "7")),
        _sample("C:bf16"),
    ]
    # Miles: max_staleness = drain version - oldest version in the batch (3).
    output = {"rollout/fully_async/max_staleness": 4}

    _per_source_metrics(samples, output)

    assert output["rollout/by_view/fp8/staleness_mean"] == 4
    assert output["rollout/by_view/fp8/post_generation_staleness_mean"] == 2
    assert output["rollout/by_view/nvfp4/staleness_mean"] == 1
    assert output["rollout/by_view/nvfp4/post_generation_staleness_mean"] == 0
    assert "rollout/by_view/bf16/staleness_mean" not in output

    unversioned = {}
    _per_source_metrics(samples, unversioned)
    assert not any("staleness" in key for key in unversioned)


def test_per_source_metrics_tolerate_a_bare_sample():
    output = {}

    _per_source_metrics([SimpleNamespace(metadata={})], output)

    assert output == {
        "rollout/by_source/unknown/sample_count": 1,
        "rollout/by_source/unknown/exit_status/unknown_ratio": 1.0,
        "rollout/by_source/unknown/infra_error_ratio": 0.0,
        "rollout/by_source/unknown/format_error_ratio": 0.0,
        "rollout/by_view/unknown/sample_count": 1,
        "rollout/by_view/unknown/exit_status/unknown_ratio": 1.0,
        "rollout/by_view/unknown/infra_error_ratio": 0.0,
        "rollout/by_view/unknown/format_error_ratio": 0.0,
    }


def test_rollout_log_hook_reports_per_view_metrics():
    samples = [
        _sample(
            "ServerH100FP8:fp8",
            reward=1.0,
            group=0,
            effective_response_length=4,
            rollout_routed_experts=None,
            metadata={"agent_metrics": {"infra_error": 1, "turns": 3}},
        ),
        _sample(
            "ServerB200NVFP4W4A16:nvfp4",
            reward={"score": 0.0},
            group=0,
            effective_response_length=6,
            rollout_routed_experts=None,
        ),
    ]
    metrics = {}

    assert (
        log_rollout_data(0, SimpleNamespace(reward_key="score"), samples, metrics, 0.0)
        is False
    )

    assert metrics["rollout/by_view/fp8/infra_error_ratio"] == 1.0
    assert metrics["rollout/by_view/fp8/turns_mean"] == 3.0
    assert metrics["rollout/by_view/nvfp4/raw_reward_mean"] == 0.0
    assert metrics["rollout/by_view/fp8/within_prompt_reward_delta_mean"] == 0.5


def test_per_source_metrics_split_exit_statuses_by_view():
    samples = [
        _sample("ServerH100FP8:fp8", metadata={"exit_status": "RepeatedFormatError"}),
        _sample("ServerH200FP8:fp8", metadata={"exit_status": "Submitted"}),
        _sample("ServerB300NVFP4W4A16:nvfp4", metadata={"exit_status": "FormatError"}),
        _sample("ServerH200BF16:bf16", metadata={"exit_status": "made-up status"}),
    ]
    output = {}

    _per_source_metrics(samples, output)

    assert output["rollout/by_view/fp8/exit_status/RepeatedFormatError_ratio"] == 0.5
    assert output["rollout/by_view/fp8/exit_status/Submitted_ratio"] == 0.5
    assert output["rollout/by_view/fp8/format_error_ratio"] == 0.5
    assert output["rollout/by_view/nvfp4/format_error_ratio"] == 1.0
    assert output["rollout/by_source/ServerH100FP8:fp8/format_error_ratio"] == 1.0
    # Unknown statuses fold into "other", as in the global ratios.
    assert output["rollout/by_view/bf16/exit_status/other_ratio"] == 1.0
    assert output["rollout/by_view/bf16/format_error_ratio"] == 0.0


def _dump_sample(reward, index, *, source="ServerH100FP8:fp8", response=""):
    return _sample(
        source,
        reward=reward,
        group=0,
        index=index,
        response=response,
        response_length=len(response),
        effective_response_length=1,
        rollout_routed_experts=None,
        metadata={"exit_status": "Submitted", "agent_metrics": {"turns": 2}},
    )


def test_rollout_log_hook_reports_a_global_format_error_ratio(tmp_path):
    samples = [_dump_sample(0.0, 0), _dump_sample(1.0, 1)]
    samples[0].metadata["exit_status"] = "RepeatedFormatError"
    metrics = {}

    log_rollout_data(
        3, SimpleNamespace(save=str(tmp_path / "checkpoints")), samples, metrics, 0.0
    )

    assert metrics["rollout_agent/format_error_ratio"] == 0.5


def test_rollout_log_hook_dumps_the_best_and_worst_trajectories(tmp_path):
    long_turn = "x" * 3_000
    samples = [
        _dump_sample(
            reward,
            index,
            response=f"<|im_start|>assistant\nfirst<|im_end|><|im_start|>assistant\n{long_turn}",
        )
        for index, reward in enumerate([0.5, 1.0, 0.0, 0.9, 0.1, 0.4])
    ]

    log_rollout_data(
        7, SimpleNamespace(save=str(tmp_path / "run" / "checkpoints")), samples, {}, 0.0
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "rollout_samples" / "rollout_000007.jsonl")
        .read_text()
        .splitlines()
    ]
    # Two highest (1.0, 0.9) and two lowest (0.0, 0.1) rewards, in batch order.
    assert [row["index"] for row in rows] == [1, 2, 3, 4]
    assert {row["rollout_id"] for row in rows} == {7}
    assert rows[0]["exit_status"] == "Submitted"
    assert rows[0]["rollout_source"] == "ServerH100FP8:fp8"
    # Only the last assistant turn is kept, capped at 2,000 characters.
    assert rows[0]["last_assistant"] == ("\n" + long_turn)[:2_000]


def test_rollout_log_hook_skips_the_dump_without_a_run_directory(tmp_path):
    samples = [_dump_sample(1.0, 0)]

    assert log_rollout_data(0, SimpleNamespace(save=None), samples, {}, 0.0) is False
    assert not any(tmp_path.iterdir())


def test_a_failed_dump_never_fails_the_step(tmp_path):
    blocker = tmp_path / "run"
    blocker.write_text("not a directory")
    samples = [_dump_sample(1.0, 0)]
    metrics = {}

    assert (
        log_rollout_data(
            0, SimpleNamespace(save=str(blocker / "checkpoints")), samples, metrics, 0.0
        )
        is False
    )
    assert metrics["rollout_agent/format_error_ratio"] == 0.0
