from types import SimpleNamespace

import pytest

from cookbook.miles_disagg.modal_swe.metrics import (
    _request_metrics,
    _routing_replay_metrics,
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
