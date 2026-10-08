import json

import pytest

from cookbook.miles_disagg.figures import history
from cookbook.miles_disagg.figures.__main__ import _steps_supplied
from cookbook.miles_disagg.figures.runs import STUDY_RUNS, Run, load_runs


def _attempt(first: int, last: int, *, tag: str) -> list[dict]:
    """One W&B run as Miles logs it: a rollout row and a train row per step."""
    rows = []
    for step in range(first, last + 1):
        rows.append(
            {"rollout/step": step, "rollout/raw_reward": float(step), "tag": tag}
        )
        rows.append({"train/step": step, "train/loss": -float(step), "tag": tag})
    return rows


def test_a_resumed_attempt_replaces_the_steps_it_relogged():
    """SC r03: attempt 1 logged steps 0-111, attempt 2 resumed from its save at 110."""
    rows = history.stitch([_attempt(0, 111, tag="a"), _attempt(110, 143, tag="b")])

    steps, rewards = history.series(rows, "rollout/raw_reward")
    assert steps == list(range(0, 144))
    owner = {row["rollout/step"]: row["tag"] for row in rows if "rollout/step" in row}
    assert owner[109] == "a" and owner[110] == "b" and owner[111] == "b"
    assert {row[history.ATTEMPT_KEY] for row in rows if row["tag"] == "b"} == {1}


def test_each_step_axis_is_cut_at_the_next_attempts_own_first_step():
    later = _attempt(110, 120, tag="b")
    later = [row for row in later if "rollout/step" in row] + [
        {"train/step": step, "train/loss": 0.0, "tag": "b"} for step in range(112, 121)
    ]

    rows = history.stitch([_attempt(0, 111, tag="a"), later])

    train = {row["train/step"]: row["tag"] for row in rows if "train/step" in row}
    assert train[111] == "a" and train[112] == "b"


def test_a_single_attempt_is_kept_whole_and_rows_without_a_step_are_dropped():
    rows = history.stitch([_attempt(0, 3, tag="a") + [{"_runtime": 5.0}]])

    assert len(rows) == 8
    assert all(history.step_axis(row) for row in rows)


def test_drop_refill_leaves_out_each_resumes_first_steps_on_each_axis():
    later = [row for row in _attempt(110, 120, tag="b") if "rollout/step" in row] + [
        {"train/step": step, "train/loss": 0.0, "tag": "b"} for step in range(112, 121)
    ]
    rows = history.stitch(
        [_attempt(0, 111, tag="a"), _attempt(105, 115, tag="b"), later]
    )

    kept = history.drop_refill(rows, 5)

    def steps(axis, tag):
        return sorted(row[axis] for row in kept if axis in row and row["tag"] == tag)

    assert steps("rollout/step", "a") == list(range(0, 105))
    assert steps("rollout/step", "b") == list(range(115, 121))
    # The second attempt's train axis starts at 105, the third's at 112.
    assert steps("train/step", "b") == [110, 111, *range(117, 121)]
    assert history.drop_refill(rows, 0) == rows


def test_up_to_keeps_steps_through_the_last_one_on_both_axes():
    rows = history.stitch([_attempt(0, 20, tag="a")])

    capped = history.up_to(rows, 10)

    assert history.series(capped, "rollout/raw_reward")[0] == list(range(0, 11))
    assert history.series(capped, "train/loss")[0] == list(range(0, 11))
    assert history.up_to(rows, None) == rows


def test_series_keeps_the_last_value_of_a_step_and_skips_non_numbers():
    rows = [
        {"rollout/step": 1, "m": 1.0},
        {"rollout/step": 1, "m": 2.0},
        {"rollout/step": 2, "m": float("nan")},
        {"rollout/step": 3, "m": True},
        {"rollout/step": 4, "m": "x"},
        {"rollout/step": 5, "m": 3},
    ]

    assert history.series(rows, "m") == ([1, 5], [2.0, 3.0])


def test_metrics_lists_numeric_keys_but_not_steps_or_wandb_internals():
    rows = [{"rollout/step": 1, "a": 1.0, "_step": 3, "flag": True, "s": "x"}]

    assert history.metrics(rows) == ["a"]


def test_ema_matches_wandb_smoothing():
    assert history.ema([1.0, 3.0], 0.0) == [1.0, 3.0]
    assert history.ema([1.0, 3.0], 0.5) == [1.0, 2.0]


def test_steps_supplied_reports_each_attempts_span_per_axis():
    rows = history.stitch([_attempt(0, 111, tag="a"), _attempt(110, 143, tag="b")])

    assert _steps_supplied(rows) == {
        "0": {"rollout/step": [0, 109], "train/step": [0, 109]},
        "1": {"rollout/step": [110, 143], "train/step": [110, 143]},
    }


def test_runs_validate_fleet_arm_and_distinct_attempts():
    with pytest.raises(ValueError, match="fleet"):
        Run("x", "L3", "GRPO", ("a",))
    with pytest.raises(ValueError, match="arm"):
        Run("x", "L0", "PPO", ("a",))
    with pytest.raises(ValueError, match="distinct"):
        Run("x", "L0", "GRPO", ("a", "a"))


def test_study_runs_have_unique_labels_and_wandb_ids():
    labels = [run.label for run in STUDY_RUNS]
    ids = [run_id for run in STUDY_RUNS for run_id in run.attempts]
    assert len(set(labels)) == len(labels)
    assert len(set(ids)) == len(ids)


def test_a_run_list_file_replaces_the_study_runs(tmp_path):
    path = tmp_path / "runs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "label": "L1 IcePop",
                    "fleet": "L1",
                    "arm": "IcePop",
                    "attempts": ["a", "b"],
                }
            ]
        )
    )

    assert load_runs(path) == (Run("L1 IcePop", "L1", "IcePop", ("a", "b")),)
