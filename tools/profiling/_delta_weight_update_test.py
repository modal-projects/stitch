import math

import pytest

from tools.profiling._delta_weight_update import (
    _assert_repeat_consistency,
    _assert_target_changed,
)


def _fingerprint(*logprobs):
    return {"text": "moonlight", "output_ids": [1, 2], "output_logprobs": logprobs}


def test_repeated_logprobs_use_zero_or_explicit_tolerance():
    base = _fingerprint(-1.0, -2.0)
    assert _assert_repeat_consistency([base, base])["repeat_max_logprob_abs_diff"] == 0
    drifted = _fingerprint(-0.875, -2.0)
    with pytest.raises(RuntimeError, match="tolerance"):
        _assert_repeat_consistency([base, drifted])
    result = _assert_repeat_consistency([base, drifted], logprob_abs_tolerance=0.125)
    assert (
        result["repeat_max_logprob_abs_diff"]
        == result["logprob_abs_tolerance"]
        == 0.125
    )


@pytest.mark.parametrize("logprobs", [(math.nan, -2.0), (-1.0,)])
def test_numerically_invalid_output_cannot_pass_repeat_or_change(logprobs):
    invalid = _fingerprint(*logprobs)
    with pytest.raises(RuntimeError, match="finite|incomplete"):
        _assert_repeat_consistency([invalid, invalid])
    with pytest.raises(RuntimeError, match="finite|incomplete"):
        _assert_target_changed(_fingerprint(-1.0, -2.0), invalid)


def test_dspark_retains_explicit_text_only_fingerprints():
    base = {"text": "before", "output_ids": [], "output_logprobs": []}
    target = {**base, "text": "after"}
    assert _assert_repeat_consistency([target, target], require_logprobs=False)[
        "exact_text"
    ]
    assert _assert_target_changed(base, target, require_logprobs=False)[
        "text_changed_from_base"
    ]
