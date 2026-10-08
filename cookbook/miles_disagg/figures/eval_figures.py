"""The main eval figures: pass@k against training step, on SWE-bench Pro V2 and its hard
subset, for every experiment in ``experiments.MAIN_EVAL``.

Each build reads every finished eval point from the eval Volume, so a newly evaluated
checkpoint appears the next time it runs. The blog's Figure 4 is pass@4 on the hard subset
and the full set, side by side; beside it, one figure per task subset has a panel per
pass@k. Every figure shares one legend: color is the run, mixed-pool (L2) runs are solid lines and
the homogeneous-hardware (L0) run is dashed. Every run starts from the base model, so each
curve starts at the base model's step-0 point, drawn in black. Every experiment is in the
legend whether or not it has points yet. An experiment's ``eval_proxy`` (an earlier run that
stood in before it had points) stays beside it as its own entry: the same color, hollow
markers on a dotted line, labelled as the stand-in. Points carry no error bars;
``points.csv`` keeps each one's standard error.

    uv run --extra modal --with matplotlib python -m cookbook.miles_disagg.figures.eval_figures \\
        --out ~/blog-figures
Writes OUT/eval/{pass4,full,hard51}.{png,svg}, points.csv and SOURCES.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cookbook.miles_disagg.figures import eval_source
from cookbook.miles_disagg.figures.experiments import BY_KEY, MAIN_EVAL, Experiment

KS = (1, 2, 4)
SUBSET_TITLES = {
    "full": "Evaluation on SWE-bench Pro V2",
    "hard51": "Evaluation on SWE-bench Pro V2, HARD-51 subset",
}
# Figure 4: one pass@k, a panel per subset.
FIGURE_K = 4
FIGURE_PANELS = (("hard51", "HARD-51 subset"), ("full", "Full set"))
# Color is what a method adds to plain GRPO: gray for nothing, one hue per building block
# (orange for SC, blue for MIS, magenta for top-p mask replay), and greens for SC + MIS,
# darker with top-p mask replay. A method keeps its color on both pools; the line style
# tells the pools apart.
COLORS = {
    "l2_grpo": "#6b7280",
    "l0_grpo": "#6b7280",
    "l2_sc": "#ea580c",
    "l2_icepop": "#2563eb",
    "l2_grpo_top_p": "#c026d3",
    "l2_sc_mis": "#22c55e",
    "l2_sc_mis_top_p": "#15803d",
    "l0_sc_mis_top_p": "#15803d",
}
BASE_COLOR = "#4b5563"
# Legend columns: the base model alone, then methods by how many building blocks they add
# to plain GRPO: none, one, and SC + MIS with its variants.
LEGEND_COLUMNS = (
    ("base-model",),
    ("l2_grpo", "l0_grpo"),
    ("l2_sc", "l2_icepop", "l2_grpo_top_p"),
    ("l2_sc_mis", "l2_sc_mis_top_p", "l0_sc_mis_top_p"),
)


@dataclass(frozen=True)
class Series:
    """What one legend entry draws: an experiment's points, its proxy's (when ``proxy`` is
    set), or nothing yet."""

    experiment: Experiment
    points: tuple[eval_source.EvalPoint, ...]
    proxy: Experiment | None = None

    @property
    def key(self) -> str:
        """The experiment whose points these are: the proxy's, for a proxy entry."""
        return self.proxy.key if self.proxy is not None else self.experiment.key

    @property
    def label(self) -> str:
        if self.proxy is not None:
            return f"{self.proxy.label}, earlier run (stand-in for {self.experiment.label})"
        if not self.points:
            return f"{self.experiment.label} (no eval yet)"
        return self.experiment.label

    @property
    def status(self) -> str:
        return (
            "proxy"
            if self.proxy is not None
            else ("evaluated" if self.points else "pending")
        )


# Finished points left out of the figures by an editorial choice, not because they are
# faulty (those are archived off the eval Volume instead): (experiment key, step) ->
# why. The points stay on the Volume and in the registry.
HIDDEN_POINTS = {
    (
        "l2_grpo",
        100,
    ): "user, 2026-10-06: a one-point rebound between collapsed steps 80 and 120; "
    "GRPO's training reward also recovers briefly near step 100",
    ("l2_grpo_top_p", 20): "user, 2026-10-06: left out of the figures",
    ("l2_grpo", 120): "user, 2026-10-07: left out of the figures",
    ("l2_icepop", 150): "user, 2026-10-07: left out of the figures",
    ("l2_icepop", 170): "user, 2026-10-07: left out of the figures",
    ("l2_grpo_top_p", 80): "user, 2026-10-07: left out of the figures",
    ("l2_grpo", 60): "user, 2026-10-07: left out of the figures",
    ("l2_grpo", 80): "user, 2026-10-07: left out of the figures",
    ("l2_icepop", 180): "user, 2026-10-07: left out of the figures",
}
# Steps left out for every method: the figures keep the 20-step grid.
HIDDEN_STEPS = {70: "user, 2026-10-07", 90: "user, 2026-10-07", 110: "user, 2026-10-07"}


def collect(
    source: eval_source.EvalSource, keys: Sequence[str] = MAIN_EVAL
) -> tuple[list[Series], eval_source.EvalPoint]:
    """Each experiment's finished points, less ``HIDDEN_POINTS``, followed by its
    proxy's when it has one, and the base point."""
    series = []
    for key in keys:
        experiment = BY_KEY[key]
        points = (
            [
                point
                for point in source.run_points(experiment.recipe, experiment.run_id)
                if (key, point.version) not in HIDDEN_POINTS
                and point.version not in HIDDEN_STEPS
            ]
            if experiment.run_id
            else []
        )
        series.append(Series(experiment, tuple(points)))
        if experiment.eval_proxy is not None:
            proxy = BY_KEY[experiment.eval_proxy]
            series.append(
                Series(
                    experiment,
                    tuple(source.run_points(proxy.recipe, proxy.run_id)),
                    proxy,
                )
            )
    return series, source.base()


def style(series: Series) -> dict[str, Any]:
    """Color is the correction, marker the sampling and line style the pool: solid for the
    mixed pool, dashed for all-B200. A proxy is dotted, with hollow markers."""
    color = COLORS[series.experiment.key]
    if series.proxy is not None:
        linestyle = ":"
    else:
        linestyle = "--" if series.experiment.fleet == "L0" else "-"
    return {
        "color": color,
        "linestyle": linestyle,
        # Triangles for top-p sampling with mask replay, circles for full vocabulary.
        "marker": "^" if series.experiment.sampling == "top-p" else "o",
        # Hollow markers for all-B200 runs (and any stand-in), filled for the mixed pool.
        "markerfacecolor": "white"
        if series.proxy is not None or series.experiment.fleet == "L0"
        else color,
    }


def draw(
    subset: str,
    series: Sequence[Series],
    base: eval_source.EvalPoint,
    hard: frozenset[str],
    ks: Sequence[int] = KS,
) -> Any:
    """One figure per task subset: a panel per pass@k, sharing one legend."""
    panels = [(subset, k, f"pass@{k}") for k in ks]
    return draw_panels(panels, series, base, hard, SUBSET_TITLES[subset])


def draw_figure(
    series: Sequence[Series],
    base: eval_source.EvalPoint,
    hard: frozenset[str],
    k: int = FIGURE_K,
) -> Any:
    """Figure 4: pass@``k`` with a panel per subset, sharing one legend."""
    panels = [(subset, k, title) for subset, title in FIGURE_PANELS]
    return draw_panels(
        panels, series, base, hard, f"Evaluation on SWE-bench Pro V2, pass@{k}"
    )


def draw_panels(
    panels: Sequence[tuple[str, int, str]],
    series: Sequence[Series],
    base: eval_source.EvalPoint,
    hard: frozenset[str],
    title: str,
) -> Any:
    """A panel per (subset, k, panel title), sharing one legend. Every run starts from
    the base model, so each curve starts at its score at step 0, and a dotted line marks
    that score across the panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * max(len(panels), 2), 4.8))
    last_step = 0
    for ax, (subset, k, panel_title) in zip(axes, panels, strict=True):
        base_scores = base.scores(subset, hard)
        metric = f"pass@{k}"
        for entry in series:
            points = [point for point in entry.points if point.n_samples >= k]
            if not points:
                continue
            steps = [0] + [point.version for point in points]
            values = [base_scores[metric]] + [
                point.scores(subset, hard)[metric] for point in points
            ]
            ax.plot(steps, values, linewidth=1.8, markersize=5, **style(entry))
            last_step = max(last_step, *steps)
        ax.axhline(
            base_scores[metric],
            color=BASE_COLOR,
            linestyle=":",
            linewidth=1.2,
            zorder=1,
            gid="base-model",
        )
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    right = max(140, last_step + 20)
    for ax in axes:
        ax.set_xlim(-4, right)
        ax.set_xticks(range(0, right + 1, 20))
    entries = {
        entry.key: Line2D(
            [], [], linewidth=1.8, markersize=5, label=entry.label, **style(entry)
        )
        for entry in series
    }
    entries["base-model"] = Line2D(
        [], [], color=BASE_COLOR, linestyle=":", linewidth=1.2, label="Base model"
    )
    handles, ncol = legend_order(entries)
    fig.suptitle(title, fontsize=12)
    # One legend for all panels, below them; a tight save keeps figure-level legends.
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.30, wspace=0.25)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        fontsize=8,
        frameon=False,
    )
    return fig


def legend_order(
    handles: Mapping[str, Any], *, one_column: bool = False
) -> tuple[list[Any], int]:
    """Legend handles laid out in LEGEND_COLUMNS, filled column by column, with any handle
    outside them (such as the base model) in one more column; blank entries pad the
    short columns. With ``one_column``, the same groups stack in one column, one blank
    entry between groups. Returns the handles and the number of columns."""
    from matplotlib.lines import Line2D

    columns = [
        [handles[key] for key in column if key in handles] for column in LEGEND_COLUMNS
    ]
    columns = [column for column in columns if column]
    placed = {key for column in LEGEND_COLUMNS for key in column}
    rest = [handle for key, handle in handles.items() if key not in placed]
    if rest:
        columns.append(rest)
    if not columns:
        return [], 1
    if one_column:
        stacked = []
        for index, column in enumerate(columns):
            if index:
                stacked.append(Line2D([], [], alpha=0, label=" "))
            stacked += column
        return stacked, 1
    depth = max(len(column) for column in columns)
    ordered = []
    for column in columns:
        blanks = depth - len(column)
        ordered += column + [Line2D([], [], alpha=0, label=" ") for _ in range(blanks)]
    return ordered, len(columns)


def sources(series: Sequence[Series], base: eval_source.EvalPoint) -> dict[str, Any]:
    """Exactly which eval points each figure entry was drawn from."""

    def describe(point: eval_source.EvalPoint) -> dict[str, Any]:
        return {
            "step": point.version,
            "path": point.path,
            "commit": point.commit[:7],
            "dirty": point.dirty,
            "n_samples": point.n_samples,
        }

    return {
        "base": describe(base),
        "experiments": [
            {
                "key": entry.experiment.key,
                "label": entry.label,
                "recipe": entry.experiment.recipe,
                "run_id": entry.experiment.run_id,
                "status": entry.status,
                "proxy": entry.proxy.key if entry.proxy else None,
                "points": [describe(point) for point in entry.points],
            }
            for entry in series
        ],
    }


def write(
    out: Path,
    series: Sequence[Series],
    base: eval_source.EvalPoint,
    hard: frozenset[str],
    *,
    formats: Sequence[str] = ("png", "svg"),
) -> list[Path]:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*_pass@*"):  # the earlier one-figure-per-k layout
        stale.unlink()
    written = []
    figures = {"pass4": lambda: draw_figure(series, base, hard)} | {
        subset: (lambda subset=subset: draw(subset, series, base, hard))
        for subset in eval_source.SUBSETS
    }
    for name, make in figures.items():
        fig = make()
        for fmt in formats:
            path = out / f"{name}.{fmt}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    rows = eval_source.rows_for_table(
        {
            "base": [base],
            **{entry.key: list(entry.points) for entry in series},
        },
        lambda key: (
            "Base model"
            if key == "base"
            else next(entry.label for entry in series if entry.key == key)
        ),
        hard,
    )
    with (out / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written.append(out / "points.csv")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache" / "stitch-figures"
    )
    parser.add_argument("--refresh", action="store_true", help="refetch cached points")
    args = parser.parse_args()

    from cookbook.miles_disagg.eval_configs import swebench_pro_hetero as spec

    source = eval_source.EvalSource(spec, args.cache, refresh=args.refresh)
    series, base = collect(source)
    hard = eval_source.hard_tasks()
    out = args.out / "eval"
    written = write(out, series, base, hard)
    summary: Mapping[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stitch_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "task_set": source.task_set,
        **sources(series, base),
    }
    (out / "SOURCES.json").write_text(json.dumps(summary, indent=2) + "\n")
    for entry in series:
        steps = ", ".join(str(point.version) for point in entry.points) or "-"
        print(f"{entry.label}: {entry.status}, steps {steps}", flush=True)
    print(f"wrote {len(written)} files to {out}", flush=True)


if __name__ == "__main__":
    main()
