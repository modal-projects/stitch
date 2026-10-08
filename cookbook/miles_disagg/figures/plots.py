"""Plots shared by every figure: one line per run, colored by arm and dashed by fleet,
so the same run looks the same in every figure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cookbook.miles_disagg.figures import history
from cookbook.miles_disagg.figures.runs import Run

ARM_COLORS = {
    "GRPO": "#6b7280",
    "IcePop": "#2563eb",
    "SC": "#ea580c",
    "SC+MIS": "#16a34a",
}
FLEET_STYLES = {"L0": ":", "L1": "--", "L2": "-"}


def style(run: Run) -> dict[str, Any]:
    return {"color": ARM_COLORS[run.arm], "linestyle": FLEET_STYLES[run.fleet]}


def metric_figure(
    metric: str,
    runs: Sequence[Run],
    histories: Mapping[str, list[dict[str, Any]]],
    *,
    smoothing: float,
    out: Path,
    formats: Sequence[str] = ("png",),
) -> bool:
    """One metric across runs: the raw series faint, the smoothed one on top. Returns
    False, writing nothing, when no run logged the metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    drawn = False
    for run in runs:
        steps, values = history.series(histories[run.label], metric)
        if not steps:
            continue
        drawn = True
        ax.plot(steps, values, alpha=0.2, linewidth=0.8, **style(run))
        ax.plot(
            steps,
            history.ema(values, smoothing),
            linewidth=1.6,
            label=run.label,
            **style(run),
        )
    if drawn:
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        out.parent.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            fig.savefig(out.with_suffix(f".{fmt}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return drawn
