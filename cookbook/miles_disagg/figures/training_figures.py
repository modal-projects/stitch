"""Training-dynamics figures and the per-pool table, from W&B.

Each figure answers one of section 5's questions and shares the eval figures' style:
color is the experiment, mixed-pool (L2) runs are solid and homogeneous-hardware (L0) runs are
dashed, and every experiment is in the one shared legend whether or not it has logged data
yet. Each run's W&B attempts are joined into one history (``history.stitch``); line panels
draw the raw series faintly and its exponential smoothing on top, against training step.

    p1_failure          what failure looks like (reward, length, repetition, format errors)
    p2_mechanism        why methods fail (gradient, entropy, mismatch, SC's tail model)
    p3_mismatch         where the mismatch comes from (by sampler, by staleness, over time)
    p4_uniform_hardware the final recipe on the mixed pool and on homogeneous hardware
    p5_pools.{md,csv}   what each pool contributes, for the final recipe
    p5_pool_shares      each pool's share of the final recipe's samples over training
    p5_pool_problems    each pool's share of the samples, tokens and extreme ratios
    p5_view_updates     what keeping each weight view fresh costs per update

    uv run --with wandb --with matplotlib python -m cookbook.miles_disagg.figures.training_figures \\
        --out ~/blog-figures
Writes OUT/training/<figure>.{png,svg}, p5_pools.{md,csv} and SOURCES.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cookbook.miles_disagg.figures import fetch, history
from cookbook.miles_disagg.figures.eval_figures import COLORS, legend_order
from cookbook.miles_disagg.figures.experiments import BY_KEY, MAIN_EVAL, Experiment

# The same experiments as the main eval figures. A proxy stands in only for eval points.
MAIN_TRAINING = MAIN_EVAL
FINAL_RECIPE = ("l2_sc_mis_top_p", "l0_sc_mis_top_p")
RELATIVE_WINDOW = 5

# The mixed pool's samplers, as the runs tag them: a reader-facing name, the weight
# precision, and the GPU family, grouped by family. A pool's mismatch and its broken
# episodes follow the family (and so the attention kernel), not the weight precision.
POOLS = {
    "ServerH100FP8:fp8": ("H100, FP8", "fp8", "hopper"),
    "ServerH200FP8:fp8": ("H200, FP8", "fp8", "hopper"),
    "ServerH100BF16TP2:bf16": ("H100, BF16", "bf16", "hopper"),
    "ServerH200BF16:bf16": ("H200, BF16", "bf16", "hopper"),
    "ServerB200NVFP4W4A16:nvfp4": ("B200, NVFP4", "nvfp4", "b200_b300"),
    "ServerB300NVFP4W4A16:nvfp4": ("B300, NVFP4", "nvfp4", "b200_b300"),
    "ServerA100BF16TP2:bf16": ("A100, BF16", "bf16", "a100_rtx"),
    "ServerRTXPRO6000BF16TP2:bf16": ("RTX PRO 6000, BF16", "bf16", "a100_rtx"),
}
# Each family's attention backend, as the hetero config sets it.
FAMILIES = {
    "hopper": ("H100 and H200", "FA3", "#c2410c"),
    "b200_b300": ("B200 and B300", "TRT-LLM", "#6d28d9"),
    "a100_rtx": ("A100 and RTX PRO 6000", "FlashInfer", "#0f766e"),
}
# The weight views, for the per-view weight updates.
PRECISION_COLORS = {"bf16": "#0f766e", "fp8": "#b45309", "nvfp4": "#7c3aed"}
# One shade per pool, grouped by GPU family, for the stacked shares.
POOL_SHADES = {
    "ServerH100FP8:fp8": "#9a3412",
    "ServerH200FP8:fp8": "#ea580c",
    "ServerH100BF16TP2:bf16": "#f59e0b",
    "ServerH200BF16:bf16": "#fde68a",
    "ServerB200NVFP4W4A16:nvfp4": "#6d28d9",
    "ServerB300NVFP4W4A16:nvfp4": "#c4b5fd",
    "ServerA100BF16TP2:bf16": "#0f766e",
    "ServerRTXPRO6000BF16TP2:bf16": "#5eead4",
}
DARK_SHADES = frozenset({"#9a3412", "#ea580c", "#6d28d9", "#0f766e"})
LAGS = (
    ("lag_0_1", "0-1"),
    ("lag_2_3", "2-3"),
    ("lag_4_5", "4-5"),
    ("lag_6_plus", "6+"),
)
FORMAT_ERRORS = "rollout_agent/exit_status/RepeatedFormatError_ratio"
# The same share for one pool, under rollout/by_source/<pool>/.
FORMAT_ERROR_FIELD = "exit_status/RepeatedFormatError_ratio"

# P3 compares the final recipe on both pools. It is stable on both, so the window runs
# from step 1 to the last step both have trained, and grows as they train.
MISMATCH_FLEETS = ((FINAL_RECIPE[1], "all-B200 BF16"),)
MISMATCH_POOLED = FINAL_RECIPE[0]


@dataclass(frozen=True)
class Panel:
    """One metric. ``relative`` divides each run's series by the mean of its first
    ``RELATIVE_WINDOW`` steps, so runs whose scales differ (raw or normalized
    advantages) compare by how far they moved."""

    metric: str
    label: str
    relative: bool = False
    log: bool = False


@dataclass(frozen=True)
class Figure:
    """``colors`` and ``labels`` override a run's method color and legend label, for a
    figure whose lines differ by sampler pool rather than by method. ``last_step`` cuts
    this figure's lines at that training step (None: each run's latest)."""

    name: str
    title: str
    panels: tuple[Panel, ...]
    keys: tuple[str, ...] = MAIN_TRAINING
    colors: tuple[tuple[str, str], ...] = ()
    labels: tuple[tuple[str, str], ...] = ()
    last_step: int | None = None


# Steps left out after each resume while the rollout buffer refills (``history.drop_refill``).
RESUME_REFILL_STEPS = 5
# The last training step drawn. None draws every run to its latest step (user, 2026-10-07;
# 110 from 2026-10-06).
LAST_STEP = None

FIGURES = (
    Figure(
        "p1_failure",
        "What failure looks like",
        (
            Panel("rollout/raw_reward", "training reward"),
            Panel("rollout/response_lengths", "response length (tokens)"),
            Panel("rollout/repetition_frac", "share of repetitive responses"),
            Panel(FORMAT_ERRORS, "share of episodes ended by repeated format errors"),
        ),
    ),
    Figure(
        "p2_mechanism",
        "Why methods fail",
        (
            Panel(
                "train/grad_norm",
                "gradient norm, relative to its first steps",
                relative=True,
                log=True,
            ),
            Panel("train/entropy_loss", "trainer entropy"),
            Panel("train/train_rollout_kl/all", "train/rollout KL", log=True),
            Panel(
                "train/train_rollout_ratio_tail_frac/all",
                "share of tokens with ratio outside [0.2, 5]",
                log=True,
            ),
            Panel(
                "train/sc_tail_ratio",
                "score centering's tail ratio (1 = consistent)",
                log=True,
            ),
        ),
    ),
    Figure(
        "p4_uniform_hardware",
        f"{BY_KEY[FINAL_RECIPE[0]].label} on the mixed pool and on homogeneous hardware",
        (
            Panel("rollout/raw_reward", "training reward"),
            Panel("train/train_rollout_kl/all", "train/rollout KL", log=True),
            Panel("train/entropy_loss", "trainer entropy"),
            Panel("train/grad_norm", "gradient norm"),
        ),
        keys=FINAL_RECIPE,
        # Both lines are the same method, so color tells the pools apart (user, 2026-10-07).
        colors=((FINAL_RECIPE[0], "#15803d"), (FINAL_RECIPE[1], "#334155")),
        labels=((FINAL_RECIPE[0], "mixed pool"), (FINAL_RECIPE[1], "all-B200 BF16")),
        # Both pools over the same steps (user, 2026-10-07).
        last_step=160,
    ),
)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def series_for(
    rows: Sequence[dict[str, Any]], panel: Panel, smoothing: float
) -> tuple[list[int], list[float], list[float]] | None:
    """Steps, raw values and smoothed values of one panel's metric, or None if absent."""
    steps, values = history.series(rows, panel.metric)
    if not steps:
        return None
    if panel.relative:
        head = values[:RELATIVE_WINDOW]
        scale = sum(head) / len(head)
        if scale <= 0:
            return None
        values = [value / scale for value in values]
    return steps, values, history.ema(values, smoothing)


def window_mean(
    rows: Sequence[dict[str, Any]], metric: str, window: tuple[int, int]
) -> float | None:
    """The metric's mean over steps ``window[0]..window[1]``, or None if it has none there."""
    steps, values = history.series(rows, metric)
    inside = [
        v for s, v in zip(steps, values, strict=True) if window[0] <= s <= window[1]
    ]
    return sum(inside) / len(inside) if inside else None


def style(experiment: Experiment) -> dict[str, Any]:
    return {
        "color": COLORS[experiment.key],
        "linestyle": "--" if experiment.fleet == "L0" else "-",
    }


def _color_override(figure: Figure, key: str) -> dict[str, str]:
    color = dict(figure.colors).get(key)
    return {"color": color} if color else {}


def label(experiment: Experiment, rows: Sequence[dict[str, Any]] | None) -> str:
    return experiment.label if rows else f"{experiment.label} (no data yet)"


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def draw(
    figure: Figure,
    histories: Mapping[str, Sequence[dict[str, Any]]],
    *,
    smoothing: float = 0.6,
) -> Any:
    """One figure: a panel per metric, sharing one legend of every experiment."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    panels = figure.panels
    spare = []
    if len(panels) > 4:
        cols = math.ceil((len(panels) + 1) / 2)  # one cell left over for the legend
        fig, grid = plt.subplots(2, cols, figsize=(4.8 * cols, 8.4))
        cells = list(grid.flat)
        axes, spare = cells[: len(panels)], cells[len(panels) :]
    else:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.4))
        axes = list(axes) if len(panels) > 1 else [axes]
    for ax, panel in zip(axes, panels, strict=True):
        for key in figure.keys:
            rows = history.up_to(histories.get(key) or [], figure.last_step)
            if not rows:
                continue
            drawn = series_for(rows, panel, smoothing)
            if drawn is None:
                continue
            steps, raw, smooth = drawn
            kwargs = {**style(BY_KEY[key]), **_color_override(figure, key)}
            ax.plot(steps, raw, alpha=0.2, linewidth=0.8, **kwargs)
            ax.plot(steps, smooth, linewidth=1.8, **kwargs)
        ax.set_title(panel.label, fontsize=10)
        ax.set_xlabel("training step")
        if panel.log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    by_key = {
        key: Line2D(
            [],
            [],
            linewidth=1.8,
            label=dict(figure.labels).get(key)
            or label(BY_KEY[key], histories.get(key)),
            **{**style(BY_KEY[key]), **_color_override(figure, key)},
        )
        for key in figure.keys
    }
    handles, ncol = legend_order(by_key)
    fig.suptitle(figure.title, fontsize=12)
    if spare:
        for ax in spare:
            ax.axis("off")
        stacked, _ = legend_order(
            {key: handle for key, handle in by_key.items()}, one_column=True
        )
        spare[0].legend(handles=stacked, loc="center", fontsize=9, frameon=False)
        fig.subplots_adjust(
            left=0.05, right=0.99, top=0.91, bottom=0.07, wspace=0.28, hspace=0.38
        )
        return fig
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.30, wspace=0.28)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol,
        fontsize=8,
        frameon=False,
    )
    return fig


def window_spread(
    rows: Sequence[dict[str, Any]], metric: str, window: tuple[int, int]
) -> float | None:
    """The metric's standard deviation across steps ``window[0]..window[1]``."""
    steps, values = history.series(rows, metric)
    inside = [
        v for s, v in zip(steps, values, strict=True) if window[0] <= s <= window[1]
    ]
    if len(inside) < 2:
        return None
    mean = sum(inside) / len(inside)
    return math.sqrt(sum((v - mean) ** 2 for v in inside) / (len(inside) - 1))


def mismatch_window(
    histories: Mapping[str, Sequence[dict[str, Any]]],
) -> tuple[int, int]:
    """Steps 1 to the last step that the pooled run and every homogeneous pool have logged."""
    lasts = []
    for key in (MISMATCH_POOLED, *(key for key, _ in MISMATCH_FLEETS)):
        steps, _ = history.series(histories.get(key, ()), "train/train_rollout_kl/all")
        if steps:
            lasts.append(max(steps))
    return (1, max(1, min(lasts))) if lasts else (1, 1)


def mismatch_values(
    histories: Mapping[str, Sequence[dict[str, Any]]],
    window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """P3's numbers: mean train/rollout KL over ``window`` (by default
    ``mismatch_window``) for each homogeneous pool and each mixed-pool sampler, and the mixed
    pool's by staleness bucket."""
    window = window or mismatch_window(histories)
    fleets = [
        (
            name,
            window_mean(histories.get(key, ()), "train/train_rollout_kl/all", window),
        )
        for key, name in MISMATCH_FLEETS
    ]
    pooled = histories.get(MISMATCH_POOLED, ())
    pools = [
        (
            name,
            family,
            window_mean(pooled, f"train/train_rollout_kl/by_source/{source}", window),
            window_spread(pooled, f"train/train_rollout_kl/by_source/{source}", window),
        )
        for source, (name, _, family) in POOLS.items()
    ]
    lags = [
        (name, window_mean(pooled, f"train/train_rollout_kl/{key}", window))
        for key, name in LAGS
    ]
    return {"window": list(window), "fleets": fleets, "pools": pools, "lags": lags}


def family_series(
    rows: Sequence[dict[str, Any]], family: str
) -> tuple[list[int], list[float]]:
    """A GPU family's train/rollout KL at each step: the mean over its pools that logged."""
    by_step: dict[int, list[float]] = {}
    for source, (_, _, member) in POOLS.items():
        if member != family:
            continue
        steps, values = history.series(
            rows, f"train/train_rollout_kl/by_source/{source}"
        )
        for step, value in zip(steps, values, strict=True):
            by_step.setdefault(step, []).append(value)
    steps = sorted(by_step)
    return steps, [sum(by_step[step]) / len(by_step[step]) for step in steps]


# P3 by source: each panel holds one source of the mismatch fixed in turn, against the all-B200
# BF16 samplers (engine and staleness only) as a reference line.
PRECISION_PAIRS = (
    ("H100", ("ServerH100FP8:fp8", "ServerH100BF16TP2:bf16")),
    ("H200", ("ServerH200FP8:fp8", "ServerH200BF16:bf16")),
)
BF16_ACCELERATORS = (
    ("A100", "ServerA100BF16TP2:bf16"),
    ("RTX PRO 6000", "ServerRTXPRO6000BF16TP2:bf16"),
    ("H100", "ServerH100BF16TP2:bf16"),
    ("H200", "ServerH200BF16:bf16"),
)
PRECISION_LABELS = {"bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4"}


def mismatch_by_source(
    histories: Mapping[str, Sequence[dict[str, Any]]],
    window: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """The numbers behind ``draw_mismatch_sources``: mean train/rollout KL (and its spread
    across steps) per mixed-pool sampler, for the all-B200 BF16 samplers, and per staleness
    bucket, over ``window`` (by default ``mismatch_window``)."""
    window = window or mismatch_window(histories)
    pooled = histories.get(MISMATCH_POOLED, ())
    uniform = histories.get(MISMATCH_FLEETS[0][0], ())
    metric = "train/train_rollout_kl"
    samplers = {
        source: (
            window_mean(pooled, f"{metric}/by_source/{source}", window),
            window_spread(pooled, f"{metric}/by_source/{source}", window),
        )
        for source in POOLS
    }
    return {
        "window": list(window),
        "samplers": samplers,
        "uniform": (
            window_mean(uniform, f"{metric}/all", window),
            window_spread(uniform, f"{metric}/all", window),
        ),
        "lags": [
            (name, window_mean(pooled, f"{metric}/{key}", window)) for key, name in LAGS
        ],
    }


def draw_mismatch_sources(
    histories: Mapping[str, Sequence[dict[str, Any]]],
    *,
    window: tuple[int, int] | None = None,
) -> Any:
    """P3, one source at a time: the same accelerator at different weight precisions, the
    same BF16 weights on different accelerators, and the mixed pool by staleness. A dashed
    line marks the all-B200 BF16 samplers, which differ from the trainer only in engine and
    staleness. The all-B200 bar (hatched) also differs in KV cache precision (BF16, against the
    mixed pool's FP8)."""
    plt = _pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    numbers = mismatch_by_source(histories, window)
    window = tuple(numbers["window"])
    samplers, (uniform, uniform_spread) = numbers["samplers"], numbers["uniform"]
    fig, (precision_ax, accelerator_ax, lag_ax) = plt.subplots(
        1,
        3,
        figsize=(16, 4.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1.3, 1.4, 1.0]},
    )
    error_kw = {"elinewidth": 0.8, "capsize": 2, "ecolor": "#374151"}

    def bar(ax, x, value, spread, color, hatch=""):
        if value is None:
            return
        drawn = ax.bar(
            x,
            value,
            width=0.38,
            yerr=spread or 0.0,
            color=color,
            alpha=0.85,
            error_kw=error_kw,
        )
        drawn[0].set_hatch(hatch)

    # Precision: the same GPU at two weight precisions, and B200 NVFP4 (mixed pool) beside
    # the all-B200 BF16 samplers.
    groups = [*PRECISION_PAIRS, ("B200", ("ServerB200NVFP4W4A16:nvfp4", None))]
    for index, (_, (low, high)) in enumerate(groups):
        low_value, low_spread = samplers[low]
        bar(
            precision_ax,
            index - 0.2,
            low_value,
            low_spread,
            PRECISION_COLORS[POOLS[low][1]],
        )
        if high is not None:
            high_value, high_spread = samplers[high]
            bar(
                precision_ax,
                index + 0.2,
                high_value,
                high_spread,
                PRECISION_COLORS["bf16"],
            )
        else:
            bar(
                precision_ax,
                index + 0.2,
                uniform,
                uniform_spread,
                PRECISION_COLORS["bf16"],
                "//",
            )
    precision_ax.set_xticks(range(len(groups)))
    precision_ax.set_xticklabels([name for name, _ in groups])
    precision_ax.set_title("precision: the same accelerator", fontsize=10)
    precision_ax.legend(
        handles=[
            Patch(color=PRECISION_COLORS[key], alpha=0.85, label=label)
            for key, label in PRECISION_LABELS.items()
        ]
        + [
            Patch(
                facecolor="white",
                edgecolor="#374151",
                hatch="//",
                label="all-B200 BF16 pool",
            )
        ]
        + [
            Line2D(
                [],
                [],
                color="#334155",
                linestyle="--",
                linewidth=1.0,
                label="all-B200 BF16 samplers",
            )
        ],
        fontsize=8,
        loc="upper right",
    )
    # Accelerator: the same BF16 weights on each GPU, all in the BF16 color.
    entries = [
        (name, *samplers[source], PRECISION_COLORS["bf16"], "")
        for name, source in BF16_ACCELERATORS
    ] + [("B200", uniform, uniform_spread, PRECISION_COLORS["bf16"], "//")]
    for index, (_, value, spread, color, hatch) in enumerate(entries):
        bar(accelerator_ax, index, value, spread, color, hatch)
    accelerator_ax.set_xticks(range(len(entries)))
    accelerator_ax.set_xticklabels([entry[0] for entry in entries], fontsize=9)
    accelerator_ax.set_title("accelerator: the same BF16 weights", fontsize=10)
    # Staleness: the mixed pool by how many versions a sample lags behind the trainer.
    lag_names = [name for name, _ in numbers["lags"]]
    lag_ax.bar(
        range(len(lag_names)),
        [value or 0.0 for _, value in numbers["lags"]],
        width=0.6,
        color="#4b5563",
        alpha=0.85,
    )
    lag_ax.set_xticks(range(len(lag_names)))
    lag_ax.set_xticklabels(lag_names)
    lag_ax.set_xlabel("weight versions behind the trainer")
    lag_ax.set_title("weight version: the mixed pool by staleness", fontsize=10)
    for ax in (precision_ax, accelerator_ax, lag_ax):
        if uniform is not None:
            ax.axhline(uniform, color="#334155", linestyle="--", linewidth=1.0)
        ax.grid(axis="y", alpha=0.3)
    precision_ax.set_ylabel("train/rollout KL (whiskers: spread across steps)")
    fig.suptitle(
        f"Which sources move the mismatch, steps {window[0]}-{window[1]}", fontsize=12
    )
    fig.tight_layout()
    return fig


def draw_mismatch(
    histories: Mapping[str, Sequence[dict[str, Any]]],
    *,
    window: tuple[int, int] | None = None,
    smoothing: float = 0.6,
) -> Any:
    """P3: where the mismatch comes from. The final recipe on both pools over the steps
    both have trained: the homogeneous pool beside each mixed-pool sampler, the mixed pool by
    staleness, and the mixed pool's mismatch by accelerator family over the whole run."""
    plt = _pyplot()
    from matplotlib.patches import Patch

    window = window or mismatch_window(histories)
    values = mismatch_values(histories, window)
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5.0), gridspec_kw={"width_ratios": [1.8, 1.0, 1.6]}
    )
    bar_ax, lag_ax, time_ax = axes
    # The homogeneous pool first (hatched; it is B200), then the mixed pool's samplers by
    # mismatch, each in its accelerator family's color.
    entries = [
        (name, value or 0.0, None, FAMILIES["b200_b300"][2], "//")
        for name, value in values["fleets"]
    ]
    entries += sorted(
        (
            (name, value or 0.0, spread, FAMILIES[family][2], "")
            for name, family, value, spread in values["pools"]
        ),
        key=lambda entry: entry[1],
    )
    positions = list(range(len(entries)))[::-1]
    bars = bar_ax.barh(
        positions,
        [entry[1] for entry in entries],
        xerr=[entry[2] or 0.0 for entry in entries],
        color=[entry[3] for entry in entries],
        alpha=0.85,
        error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#374151"},
    )
    for bar, entry in zip(bars, entries, strict=True):
        bar.set_hatch(entry[4])
    bar_ax.set_yticks(positions)
    bar_ax.set_yticklabels([entry[0] for entry in entries], fontsize=8)
    bar_ax.axhline(
        positions[len(values["fleets"]) - 1] - 0.5, color="#9ca3af", linewidth=0.8
    )
    bar_ax.set_xlabel("train/rollout KL (whiskers: spread across steps)")
    bar_ax.set_title(
        f"by sampler, steps {window[0]}-{window[1]} (hatched: the all-B200 pool)",
        fontsize=10,
    )
    lag_names = [name for name, _ in values["lags"]]
    lag_ax.bar(
        range(len(lag_names)),
        [value or 0.0 for _, value in values["lags"]],
        color="#4b5563",
        alpha=0.85,
    )
    lag_ax.set_xticks(range(len(lag_names)))
    lag_ax.set_xticklabels(lag_names, fontsize=9)
    lag_ax.set_xlabel("staleness (weight versions behind the trainer)")
    lag_ax.set_title(
        f"mixed pool by staleness, steps {window[0]}-{window[1]}", fontsize=10
    )
    pooled = histories.get(MISMATCH_POOLED, ())
    for family, (_, _, color) in FAMILIES.items():
        steps, raw = family_series(pooled, family)
        if not steps:
            continue
        time_ax.plot(steps, raw, color=color, alpha=0.2, linewidth=0.8)
        time_ax.plot(steps, history.ema(raw, smoothing), color=color, linewidth=1.8)
    time_ax.axvspan(window[0], window[1], color="#e5e7eb", alpha=0.6, zorder=0)
    time_ax.set_yscale("log")
    time_ax.set_xlabel("training step")
    time_ax.set_title("mixed pool by accelerator family, over the run", fontsize=10)
    bar_ax.grid(alpha=0.3, axis="x")
    for ax in (lag_ax, time_ax):
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        f"Where the mismatch comes from: {BY_KEY[MISMATCH_POOLED].label} "
        "(train/rollout KL)",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.24, wspace=0.30)
    fig.legend(
        handles=[
            Patch(color=color, label=f"{gpus}: {kernel} attention")
            for gpus, kernel, color in FAMILIES.values()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    return fig


# The blog's table. The CSV also keeps the within-prompt reward delta and the per-request
# token rate; the rate counts each turn's queueing and prefill as well as its decode, and
# depends on how many sessions routing gives each engine, so it does not rank hardware.
POOL_COLUMNS = (
    ("pool", "Pool"),
    ("attention", "Attention kernel"),
    ("sample_share", "Share of samples"),
    ("reward", "Training reward"),
    ("staleness", "Staleness (versions)"),
    ("format_errors", "Repeated format errors"),
    ("kl", "Train/rollout KL"),
    ("ratio_outside", "Tokens with ratio outside [0.2, 5]"),
)


def pool_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """P5: each mixed-pool sampler over every logged step. Per-sample quantities are
    weighted by the pool's samples at each step; per-token ones are step means."""
    out = []
    totals = {source: 0.0 for source in POOLS}
    sums: dict[str, dict[str, float]] = {source: {} for source in POOLS}
    weighted = (
        "raw_reward_mean",
        "within_prompt_reward_delta_mean",
        "staleness_mean",
        FORMAT_ERROR_FIELD,
        "completion_tokens_per_backend_request_second",
    )
    for row in rows:
        for source in POOLS:
            count = row.get(f"rollout/by_source/{source}/sample_count")
            if not (_number(count) and count > 0):
                continue
            totals[source] += count
            for name in weighted:
                value = row.get(f"rollout/by_source/{source}/{name}")
                if value is None and name == FORMAT_ERROR_FIELD:
                    value = 0.0  # logged only when some episode ended that way
                if _number(value):
                    sums[source][name] = sums[source].get(name, 0.0) + count * value
                    sums[source][name + "#n"] = (
                        sums[source].get(name + "#n", 0.0) + count
                    )
    grand = sum(totals.values())
    for source, (name, _, family) in POOLS.items():

        def mean(field: str, source: str = source) -> float | None:
            n = sums[source].get(field + "#n")
            return sums[source][field] / n if n else None

        steps_kl = history.series(rows, f"train/train_rollout_kl/by_source/{source}")[1]
        steps_tail = history.series(
            rows, f"train/train_rollout_ratio_tail_frac/by_source/{source}"
        )[1]
        out.append(
            {
                "pool": name,
                "attention": FAMILIES[family][1],
                "sample_share": totals[source] / grand if grand else None,
                "reward": mean("raw_reward_mean"),
                "reward_vs_prompt": mean("within_prompt_reward_delta_mean"),
                "request_tokens_per_s": mean(
                    "completion_tokens_per_backend_request_second"
                ),
                "staleness": mean("staleness_mean"),
                "format_errors": mean(FORMAT_ERROR_FIELD),
                "kl": sum(steps_kl) / len(steps_kl) if steps_kl else None,
                "ratio_outside": sum(steps_tail) / len(steps_tail)
                if steps_tail
                else None,
            }
        )
    return out


def _format(field: str, value: Any) -> str:
    if value is None:
        return "-"
    if field in ("pool", "attention"):
        return str(value)
    if field in ("sample_share", "format_errors", "ratio_outside"):
        return f"{100 * value:.2f}%"
    if field == "reward_vs_prompt":
        return f"{value:+.3f}"
    if field == "request_tokens_per_s":
        return f"{value:.0f}"
    if field in ("staleness",):
        return f"{value:.2f}"
    if field == "kl":
        return f"{value:.4f}"
    return f"{value:.3f}"


def write_pool_table(out: Path, rows: Sequence[dict[str, Any]], steps: int) -> None:
    table = pool_table(rows)
    with (out / "p5_pools.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = [
        f"{BY_KEY[FINAL_RECIPE[0]].label} on the mixed pool, over {steps} training steps.",
        "",
        "| " + " | ".join(title for _, title in POOL_COLUMNS) + " |",
        "|" + "---|" * len(POOL_COLUMNS),
    ]
    for entry in table:
        lines.append(
            "| "
            + " | ".join(_format(field, entry[field]) for field, _ in POOL_COLUMNS)
            + " |"
        )
    (out / "p5_pools.md").write_text("\n".join(lines) + "\n")


SHARE_PANELS = (
    ("sample_count", "share of training samples"),
    ("token_count", "share of training tokens"),
    ("gradient", "share of samples that carry gradient"),
)


def pool_shares(
    rows: Sequence[dict[str, Any]], quantity: str, smoothing: float
) -> tuple[list[int], dict[str, list[float]]]:
    """Each pool's share of the training batch at each step, after smoothing each pool's
    count: samples, tokens, or samples that carry gradient (nonzero advantage)."""
    counts: dict[str, dict[int, float]] = {}
    for source in POOLS:
        prefix = f"rollout/training_batch/{source}"
        steps, samples = history.series(rows, f"{prefix}/sample_count")
        if quantity == "gradient":
            share = _by_step(rows, f"{prefix}/gradient_sample_percentage")
            values = [
                n * share.get(step, 0.0) / 100
                for step, n in zip(steps, samples, strict=True)
            ]
        elif quantity == "token_count":
            steps, values = history.series(rows, f"{prefix}/token_count")
        else:
            values = samples
        counts[source] = dict(zip(steps, values, strict=True))
    steps = sorted({step for by_step in counts.values() for step in by_step})
    smoothed = {
        source: history.ema(
            [counts[source].get(step, 0.0) for step in steps], smoothing
        )
        for source in POOLS
    }
    totals = [sum(smoothed[source][i] for source in POOLS) for i in range(len(steps))]
    return steps, {
        source: [
            v / t if t else 0.0 for v, t in zip(smoothed[source], totals, strict=True)
        ]
        for source in POOLS
    }


def draw_pool_shares(rows: Sequence[dict[str, Any]], *, smoothing: float = 0.6) -> Any:
    """Each pool's share of the final recipe's training samples over training steps, as
    a 100% stacked area. (``pool_shares`` also gives token and gradient-carrying shares;
    they track the sample shares closely, so the figure shows samples only.)"""
    plt = _pyplot()
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 4.8))
    steps, shares = pool_shares(rows, "sample_count", smoothing)
    if steps:
        ax.stackplot(
            steps,
            *[shares[source] for source in POOLS],
            colors=[POOL_SHADES[source] for source in POOLS],
            linewidth=0.3,
            edgecolor="white",
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("training step")
    ax.set_ylabel("share of training samples")
    ax.margins(x=0)
    ax.set_title(
        "Share of training samples by sampler pool, in one run on the mixed pool",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.08, right=0.74, top=0.90, bottom=0.12)
    fig.legend(
        handles=[
            Patch(color=POOL_SHADES[source], label=name)
            for source, (name, _, _) in reversed(list(POOLS.items()))
        ],
        loc="center left",
        bbox_to_anchor=(0.75, 0.5),
        fontsize=9,
        frameon=False,
        title="pool (top to bottom)",
        title_fontsize=9,
    )
    return fig


PROBLEM_ROWS = (
    ("samples", "training samples"),
    ("tokens", "training tokens"),
    ("format_errors", "episodes ended by\nrepeated format errors"),
    ("ratio_outside", "tokens with ratio\noutside [0.2, 5]"),
)
# The rows Figure 10 draws. The user (2026-10-07) dropped the format-error row and the
# step-0 format-error panel: both point at one accelerator family. problem_shares and
# base_format_errors still compute them.
DRAWN_PROBLEM_ROWS = ("samples", "tokens", "ratio_outside")


def _by_step(rows: Sequence[dict[str, Any]], metric: str) -> dict[int, float]:
    steps, values = history.series(rows, metric)
    return dict(zip(steps, values, strict=True))


def problem_shares(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Each pool's share, over every logged step, of the training samples, the training
    tokens, the episodes that ended on repeated format errors, and the training tokens
    whose train/rollout ratio left [0.2, 5]. A step's counts pair the rollout batch with
    the train step of the same number, which trains on it."""
    totals: dict[str, dict[str, float]] = {
        quantity: {source: 0.0 for source in POOLS} for quantity, _ in PROBLEM_ROWS
    }
    for source in POOLS:
        samples = _by_step(rows, f"rollout/by_source/{source}/sample_count")
        errors = _by_step(rows, f"rollout/by_source/{source}/{FORMAT_ERROR_FIELD}")
        tokens = _by_step(rows, f"rollout/training_batch/{source}/token_count")
        outside = _by_step(
            rows, f"train/train_rollout_ratio_tail_frac/by_source/{source}"
        )
        for step, count in samples.items():
            totals["samples"][source] += count
            # A status that no episode of the pool ended on is not logged that step.
            totals["format_errors"][source] += count * errors.get(step, 0.0)
        for step, count in tokens.items():
            totals["tokens"][source] += count
            if step in outside:
                totals["ratio_outside"][source] += count * outside[step]
    shares = {}
    for quantity, by_source in totals.items():
        grand = sum(by_source.values())
        shares[quantity] = {
            source: value / grand if grand else 0.0
            for source, value in by_source.items()
        }
    return shares


def base_format_errors(
    histories: Mapping[str, Sequence[dict[str, Any]]], keys: Sequence[str]
) -> dict[str, list[tuple[str, float]]]:
    """Each pool's share of episodes that ended on repeated format errors at step 0, when
    every run still samples from the base model: one value per mixed-pool run."""
    out: dict[str, list[tuple[str, float]]] = {source: [] for source in POOLS}
    for key in keys:
        rows = histories.get(key)
        if not rows or BY_KEY[key].fleet != "L2":
            continue
        for source in POOLS:
            prefix = f"rollout/by_source/{source}"
            count = _by_step(rows, f"{prefix}/sample_count").get(0)
            if not count:
                continue
            errors = _by_step(rows, f"{prefix}/{FORMAT_ERROR_FIELD}")
            out[source].append((key, errors.get(0, 0.0)))
    return out


def draw_pool_problems(
    rows: Sequence[dict[str, Any]],
    histories: Mapping[str, Sequence[dict[str, Any]]],
) -> Any:
    """Each pool's share of the final recipe's training samples, training tokens and
    tokens whose train/rollout ratio left [0.2, 5] (100% stacked), so a pool's share of
    the extreme ratios can be read against its share of the data. ``histories`` is kept
    for the caller's signature; the step-0 panel it fed is no longer drawn."""
    plt = _pyplot()
    from matplotlib.patches import Patch

    del histories
    experiment = BY_KEY[FINAL_RECIPE[0]]
    drawn = [
        (quantity, title)
        for quantity, title in PROBLEM_ROWS
        if quantity in DRAWN_PROBLEM_ROWS
    ]
    fig, stack_ax = plt.subplots(figsize=(10, 3.9))
    shares = problem_shares(rows)
    positions = list(range(len(drawn)))[::-1]
    for position, (quantity, _) in zip(positions, drawn, strict=True):
        left = 0.0
        for source in POOLS:
            width = shares[quantity][source]
            stack_ax.barh(
                position,
                width,
                left=left,
                color=POOL_SHADES[source],
                edgecolor="white",
                linewidth=0.6,
            )
            if width >= 0.06:
                stack_ax.text(
                    left + width / 2,
                    position,
                    f"{100 * width:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if POOL_SHADES[source] in DARK_SHADES else "#111827",
                )
            left += width
    stack_ax.set_yticks(positions)
    stack_ax.set_yticklabels([title for _, title in drawn], fontsize=9)
    stack_ax.set_xlim(0, 1)
    stack_ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    steps = history.series(rows, "rollout/raw_reward")[0]
    span = f"steps {steps[0]}-{steps[-1]}" if steps else "no data yet"
    stack_ax.set_title(
        f"each pool's share: {experiment.label} on the mixed pool, {span}", fontsize=10
    )
    fig.suptitle("Where the extreme ratios come from, pool by pool", fontsize=12)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.80, bottom=0.30)
    fig.legend(
        handles=[
            Patch(
                color=POOL_SHADES[source],
                label=f"{name} ({FAMILIES[family][1]} attention)",
            )
            for source, (name, _, family) in POOLS.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    return fig


VIEW_PANELS = (
    ("perf/update_weights_wire_bytes", "bytes shipped per weight update (GB)", 1e-9),
    ("perf/update_weights_density", "share of the view's bytes that change (%)", 100.0),
)


def draw_view_updates(rows: Sequence[dict[str, Any]]) -> Any:
    """Per weight view: what one weight update ships to the samplers (its delta's size on
    the wire) and the share of the view's bytes it changes, over the final recipe's run."""
    plt = _pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    experiment = BY_KEY[FINAL_RECIPE[0]]
    fig, axes = plt.subplots(1, len(VIEW_PANELS), figsize=(11, 4.4))
    for ax, (metric, title, scale) in zip(axes, VIEW_PANELS, strict=True):
        for view, color in PRECISION_COLORS.items():
            steps, values = history.series(rows, f"{metric}/{view}")
            if steps:
                ax.plot(
                    steps,
                    [value * scale for value in values],
                    color=color,
                    marker="o",
                    markersize=3,
                    linewidth=1.6,
                )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("training step")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Keeping three weight views fresh: {experiment.label} on the mixed pool",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.25, wspace=0.25)
    fig.legend(
        handles=[
            Line2D([], [], color=color, linewidth=1.6, label=f"{view.upper()} weights")
            for view, color in PRECISION_COLORS.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    return fig


def load(
    keys: Sequence[str], api: Any, cache: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Each experiment's stitched W&B history, and which attempts supplied which steps."""
    histories, sources = {}, {}
    for key in keys:
        experiment = BY_KEY[key]
        if not experiment.attempts:
            sources[key] = {"label": experiment.label, "attempts": []}
            continue
        attempts, missing = [], []
        for run_id in experiment.attempts:
            try:
                attempts.append(fetch.fetch_run(api, run_id, cache))
            except Exception as error:  # noqa: BLE001 — a deleted W&B run
                # Keep the cached copy if there is one; otherwise draw without it.
                hit = fetch.cached(cache, run_id)
                if hit is not None:
                    attempts.append(hit)
                else:
                    missing.append({"run_id": run_id, "error": str(error)[:200]})
        if not attempts:
            sources[key] = {
                "label": experiment.label,
                "attempts": [],
                "missing": missing,
            }
            continue
        rows = history.up_to(
            history.drop_refill(
                history.stitch([rows for rows, _ in attempts]), RESUME_REFILL_STEPS
            ),
            LAST_STEP,
        )
        histories[key] = rows
        spans: dict[str, dict[str, list[int]]] = {}
        for row in rows:
            axis = history.step_axis(row)
            span = spans.setdefault(str(row[history.ATTEMPT_KEY]), {}).setdefault(
                axis, [row[axis], row[axis]]
            )
            span[0], span[1] = min(span[0], row[axis]), max(span[1], row[axis])
        sources[key] = {
            "label": experiment.label,
            "recipe": experiment.recipe,
            "run_id": experiment.run_id,
            "attempts": [
                {**meta, "supplied": spans.get(str(index), {})}
                for index, (_, meta) in enumerate(attempts)
            ],
            "missing": missing,
        }
    return histories, sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, default=Path.home() / ".cache" / "stitch-figures"
    )
    parser.add_argument("--smoothing", type=float, default=0.6)
    args = parser.parse_args()

    plt = _pyplot()
    import wandb

    keys = sorted(
        {key for figure in FIGURES for key in figure.keys}
        | {key for key, _ in MISMATCH_FLEETS}
        | {MISMATCH_POOLED}
    )
    histories, sources = load(keys, wandb.Api(timeout=120), args.cache)
    out = args.out / "training"
    out.mkdir(parents=True, exist_ok=True)
    # p2_score_centering folded into p2_mechanism; p5_pool_quality became p5_pool_problems.
    for stale in (*out.glob("p2_score_centering.*"), *out.glob("p5_pool_quality.*")):
        stale.unlink()
    rendered = [
        (figure.name, draw(figure, histories, smoothing=args.smoothing))
        for figure in FIGURES
    ]
    rendered.append(("p3_mismatch", draw_mismatch_sources(histories)))
    rendered.append(
        (
            "p5_pool_shares",
            draw_pool_shares(
                histories.get(FINAL_RECIPE[0], []), smoothing=args.smoothing
            ),
        )
    )
    final = histories.get(FINAL_RECIPE[0], [])
    rendered.append(("p5_pool_problems", draw_pool_problems(final, histories)))
    rendered.append(("p5_view_updates", draw_view_updates(final)))
    for name, fig in rendered:
        for fmt in ("png", "svg"):
            fig.savefig(out / f"{name}.{fmt}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    final_steps = len(history.series(final, "rollout/raw_reward")[0])
    write_pool_table(out, final, final_steps)
    (out / "SOURCES.json").write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stitch_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True
                ).stdout.strip(),
                "smoothing": args.smoothing,
                "figures": {
                    **{
                        figure.name: [panel.metric for panel in figure.panels]
                        for figure in FIGURES
                    },
                    "p3_mismatch": mismatch_values(histories),
                    "p5_pools": {"experiment": FINAL_RECIPE[0], "steps": final_steps},
                },
                "experiments": sources,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    for key in keys:
        rows = histories.get(key)
        print(f"{BY_KEY[key].label}: {len(rows) if rows else 0} rows", flush=True)
    print(f"wrote {len(rendered)} figures and the pool table to {out}", flush=True)


if __name__ == "__main__":
    main()
