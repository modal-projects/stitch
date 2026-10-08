"""The runs a figure compares: each is one record run, with every W&B run it logged to.

``STUDY_RUNS`` comes from ``experiments.EXPERIMENTS``, which follows the record-runs table
in ``QWEN36_RUNS.md``; a JSON file of the same shape (a list of objects with these fields)
replaces it for any other comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cookbook.miles_disagg.figures.experiments import ARMS, EXPERIMENTS, FLEETS


@dataclass(frozen=True)
class Run:
    """``attempts`` are the record run's W&B run ids in the order they ran; a later
    attempt resumed from a save of the one before it."""

    label: str
    fleet: str
    arm: str
    attempts: tuple[str, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if self.fleet not in FLEETS:
            raise ValueError(f"{self.label}: fleet must be one of {FLEETS}")
        if self.arm not in ARMS:
            raise ValueError(f"{self.label}: arm must be one of {ARMS}")
        if not self.attempts or len(set(self.attempts)) != len(self.attempts):
            raise ValueError(f"{self.label}: attempts must be distinct W&B run ids")


# The experiments with a training history, from the study's one experiment manifest.
STUDY_RUNS = tuple(
    Run(
        experiment.label,
        experiment.fleet,
        experiment.arm,
        experiment.attempts,
        experiment.note,
    )
    for experiment in EXPERIMENTS
    if experiment.attempts
)


def load_runs(path: Path) -> tuple[Run, ...]:
    rows = json.loads(Path(path).read_text())
    return tuple(
        Run(
            label=row["label"],
            fleet=row["fleet"],
            arm=row["arm"],
            attempts=tuple(row["attempts"]),
            note=row.get("note", ""),
        )
        for row in rows
    )
