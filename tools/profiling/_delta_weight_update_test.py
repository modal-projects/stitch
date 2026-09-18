from __future__ import annotations

import json
import math

import httpx
import pytest

from tools.profiling._delta_weight_update import (
    WeightUpdateSpec,
    _assert_repeat_consistency,
    _assert_target_changed,
    _generate,
)


def _fingerprint(*logprobs: float) -> dict:
    return {
        "text": "The Moon reflects sunlight.",
        "output_ids": [1, 2],
        "output_logprobs": list(logprobs),
    }


def test_repeat_rejects_different_logprobs_by_default() -> None:
    with pytest.raises(RuntimeError, match="logprob.*tolerance"):
        _assert_repeat_consistency(
            [_fingerprint(-1.0, -2.0), _fingerprint(-30.0, -60.0)]
        )


def test_repeat_accepts_exact_logprobs_by_default() -> None:
    fingerprint = _fingerprint(-1.0, -2.0)

    result = _assert_repeat_consistency([fingerprint, fingerprint])

    assert result["repeat_max_logprob_abs_diff"] == 0.0
    assert result["logprob_abs_tolerance"] == 0.0


@pytest.mark.parametrize("difference", [0.0, 0.125])
def test_repeat_reports_the_declared_tolerance(difference: float) -> None:
    result = _assert_repeat_consistency(
        [_fingerprint(-1.0, -2.0), _fingerprint(-1.0 + difference, -2.0)],
        logprob_abs_tolerance=0.125,
    )

    assert result["repeat_max_logprob_abs_diff"] == difference
    assert result["logprob_abs_tolerance"] == 0.125
    assert result["exact_token_ids"]


def test_repeat_rejects_logprobs_beyond_the_declared_tolerance() -> None:
    with pytest.raises(RuntimeError, match="logprob.*tolerance"):
        _assert_repeat_consistency(
            [_fingerprint(-1.0, -2.0), _fingerprint(-0.75, -2.0)],
            logprob_abs_tolerance=0.125,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("invalid_index", [0, 1])
def test_repeat_rejects_nonfinite_logprobs(value: float, invalid_index: int) -> None:
    fingerprints = [_fingerprint(-1.0, -2.0), _fingerprint(-1.0, -2.0)]
    fingerprints[invalid_index]["output_logprobs"][0] = value

    with pytest.raises(RuntimeError, match="finite"):
        _assert_repeat_consistency(fingerprints)


@pytest.mark.parametrize("logprobs", [[], [-1.0], [-1.0, -2.0, -3.0]])
def test_repeat_requires_one_logprob_per_token(logprobs: list[float]) -> None:
    fingerprint = _fingerprint(*logprobs)

    with pytest.raises(RuntimeError, match="logprobs.*incomplete"):
        _assert_repeat_consistency([fingerprint, fingerprint])


def test_repeat_requires_token_observations() -> None:
    fingerprint = {"text": "", "output_ids": [], "output_logprobs": []}

    with pytest.raises(RuntimeError, match="token IDs.*incomplete"):
        _assert_repeat_consistency([fingerprint, fingerprint])


def test_dspark_can_explicitly_use_text_only_fingerprints() -> None:
    baseline = {"text": "before update", "output_ids": [], "output_logprobs": []}
    target = {**baseline, "text": "after update"}

    repeat = _assert_repeat_consistency([target, target], require_logprobs=False)
    changed = _assert_target_changed(baseline, target, require_logprobs=False)

    assert repeat["exact_text"]
    assert "exact_token_ids" not in repeat
    assert "repeat_max_logprob_abs_diff" not in repeat
    assert changed["text_changed_from_base"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_target_change_rejects_nonfinite_logprobs(value: float) -> None:
    target = {**_fingerprint(value, -2.0), "text": "a different completion"}

    with pytest.raises(RuntimeError, match="finite"):
        _assert_target_changed(_fingerprint(-1.0, -2.0), target)


@pytest.mark.parametrize(
    ("logprobs", "message"),
    [
        ([math.nan, -2.0], "finite"),
        ([math.inf, -2.0], "finite"),
        ([-math.inf, -2.0], "finite"),
        ([-1.0], "incomplete"),
    ],
)
def test_generation_rejects_invalid_fingerprint_observations(
    monkeypatch, logprobs: list[float], message: str
) -> None:
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={"data": [{"id": "test"}]},
        ),
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", url),
            text=json.dumps(
                {
                    "text": "The Moon reflects sunlight.",
                    "output_ids": [1, 2],
                    "meta_info": {"output_token_logprobs": logprobs},
                }
            ),
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        _generate("http://engine", fingerprint=True)


@pytest.mark.parametrize("tolerance", [-1.0, math.nan, math.inf])
def test_profile_rejects_an_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        WeightUpdateSpec(
            model_name="test",
            base_checkpoint_dir="/base",
            local_target_checkpoint_dir="/target",
            local_canonical_checkpoint_dir="/canonical",
            server_args={},
            logprob_abs_tolerance=tolerance,
        )
