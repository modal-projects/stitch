import pytest

from tools.fleet.precision import _pair_metrics, summarize


def test_pair_metrics_report_difference_and_normalized_ess() -> None:
    result = _pair_metrics([-1.0, -2.0], [-1.0, -1.0])

    assert result["mean_abs_diff"] == 0.5
    assert result["max_abs_diff"] == 1.0
    assert 0.0 < result["token_ess"] <= 1.0


def test_pair_metrics_reject_misaligned_vectors() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        _pair_metrics([-1.0], [-1.0, -2.0])


def test_summary_aggregates_each_comparison() -> None:
    metric = {
        "mean_diff": 0.0,
        "mean_abs_diff": 0.25,
        "p50_abs_diff": 0.2,
        "p95_abs_diff": 0.5,
        "p99_abs_diff": 0.5,
        "max_abs_diff": 0.5,
        "token_ess": 0.9,
    }
    result = summarize(
        [
            {
                "tokens": 8,
                "candidate_decode_vs_reference": metric,
                "candidate_prefill_vs_reference": metric,
                "candidate_decode_vs_prefill": metric,
            }
        ]
    )

    assert result["samples"] == 1
    assert result["tokens"] == 8
    assert result["candidate_prefill_vs_reference"]["mean_abs_diff"] == 0.25
