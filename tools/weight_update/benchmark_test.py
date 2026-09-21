from tools.weight_update.sglang import (
    _assert_target_changed,
    _compare_fingerprints,
)


def _fingerprint(*logprobs, text="moonlight", token_ids=(1, 2)):
    return {
        "text": text,
        "output_ids": list(token_ids),
        "output_logprobs": list(logprobs),
    }


def test_equal_fingerprints_record_exact_behavior() -> None:
    fingerprint = _fingerprint(-1.0, -2.0)

    result = _compare_fingerprints([fingerprint, fingerprint])

    assert result["exact_text"]
    assert result["exact_token_ids"]
    assert result["repeat_max_logprob_abs_diff"] == 0.0


def test_kernel_drift_is_recorded_without_substituting_for_weight_checksums() -> None:
    first = _fingerprint(-1.0, -2.0)
    second = _fingerprint(-0.5, text="different", token_ids=(1, 3))

    result = _compare_fingerprints([first, second])

    assert not result["exact_text"]
    assert not result["exact_token_ids"]
    assert result["first_token_mismatch"] == 1
    assert "comparison_text_sha256" in result
    assert "repeat_max_logprob_abs_diff" not in result


def test_target_fingerprint_must_change() -> None:
    base = _fingerprint(-1.0, -2.0)
    changed = _fingerprint(-0.75, -2.0)

    result = _assert_target_changed(base, changed)

    assert result["changed_from_base"]
    assert result["max_logprob_abs_diff_from_base"] == 0.25
