"""Join a record run's W&B attempts into one history, and read series out of it.

Miles logs two kinds of rows: rollout rows carry ``rollout/step`` and train rows carry
``train/step``. A resumed attempt starts again at its save and re-logs the steps the
previous attempt reached after it; those later rows are the record, because the run
continued from them. So for each step axis, an attempt keeps only the rows before the
first step the next attempt logged on that axis.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

STEP_KEYS = ("rollout/step", "train/step")
ATTEMPT_KEY = "_attempt"


def step_axis(row: dict[str, Any]) -> str | None:
    """The step key a row is logged against, if any."""
    for key in STEP_KEYS:
        if row.get(key) is not None:
            return key
    return None


def stitch(attempts: Sequence[Iterable[dict[str, Any]]]) -> list[dict[str, Any]]:
    """One history from the attempts in order: each row tagged with its attempt index,
    and on each step axis a later attempt's steps replace the earlier attempts'."""
    rows_by_attempt = [
        [row for row in rows if step_axis(row) is not None] for rows in attempts
    ]
    first_steps = [
        {
            key: min((row[key] for row in rows if step_axis(row) == key), default=None)
            for key in STEP_KEYS
        }
        for rows in rows_by_attempt
    ]
    stitched = []
    for index, rows in enumerate(rows_by_attempt):
        for row in rows:
            key = step_axis(row)
            cutoff = next(
                (
                    later[key]
                    for later in first_steps[index + 1 :]
                    if later[key] is not None
                ),
                None,
            )
            if cutoff is None or row[key] < cutoff:
                stitched.append({**row, ATTEMPT_KEY: index})
    return stitched


def drop_refill(rows: Sequence[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    """``rows`` without the first ``window`` steps of each resumed attempt, per step axis.

    A resume empties the rollout buffer, so its first steps train on the episodes that
    finish first, sampled from the restored weights: staleness falls to about one
    version and step time doubles until the buffer refills."""
    starts: dict[tuple[int, str], int] = {}
    for row in rows:
        key = step_axis(row)
        if key is not None and row[ATTEMPT_KEY] > 0:
            slot = (row[ATTEMPT_KEY], key)
            starts[slot] = min(starts.get(slot, row[key]), row[key])
    kept = []
    for row in rows:
        key = step_axis(row)
        start = None if key is None else starts.get((row[ATTEMPT_KEY], key))
        if key is None or start is None or not start <= row[key] < start + window:
            kept.append(row)
    return kept


def up_to(
    rows: Iterable[dict[str, Any]], last_step: int | None
) -> list[dict[str, Any]]:
    """The rows logged at or before ``last_step`` on their step axis (all when None)."""
    return [
        row
        for row in rows
        if (key := step_axis(row)) is not None
        and (last_step is None or row[key] <= last_step)
    ]


def series(
    rows: Iterable[dict[str, Any]], metric: str
) -> tuple[list[int], list[float]]:
    """``metric`` against its row's step, sorted by step; a step logged twice keeps
    its last value. Non-numeric and non-finite values are skipped."""
    by_step: dict[int, float] = {}
    for row in rows:
        value, key = row.get(metric), step_axis(row)
        if (
            key is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            continue
        if math.isfinite(value):
            by_step[int(row[key])] = float(value)
    steps = sorted(by_step)
    return steps, [by_step[step] for step in steps]


def metrics(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Every numeric metric in the history, excluding W&B's own and the step keys."""
    names = set()
    for row in rows:
        for key, value in row.items():
            if (
                not key.startswith("_")
                and key not in STEP_KEYS
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                names.add(key)
    return sorted(names)


def ema(values: Sequence[float], weight: float) -> list[float]:
    """W&B-style exponential smoothing; ``weight`` 0 returns the values unchanged."""
    smoothed, last = [], None
    for value in values:
        last = value if last is None else weight * last + (1 - weight) * value
        smoothed.append(last)
    return smoothed
