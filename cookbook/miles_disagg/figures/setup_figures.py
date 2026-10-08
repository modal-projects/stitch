"""The setup figure: the mixed pool's samplers feeding one trainer.

Each sampler pool is one box inside a dashed frame for the whole pool, labeled with its
weight precision, attention kernel and share of the samples, and filled with the same
color it has in the per-pool figures. Boxes run in the per-pool figures' order: the four
Hopper (FA3) pools, then B200 and B300 (TRT-LLM), then A100 and RTX PRO 6000 (FlashInfer). The shares come from the fleet config, not from a
run: the router gives each pool sessions in proportion to its engines times its sessions
per engine, and the trained batch follows (Table 2 measures it).

    uv run --with matplotlib python -m cookbook.miles_disagg.figures.setup_figures \\
        --out ~/blog-figures
Writes OUT/setup/fig0_setup.{png,svg}.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cookbook.miles_disagg.figures.training_figures import (
    DARK_SHADES,
    FAMILIES,
    POOL_SHADES,
    POOLS,
)

# Modal GPU name -> the accelerator as the blog names it.
ACCELERATORS = {
    "H100!": "H100",
    "H200": "H200",
    "B200": "B200",
    "B300": "B300",
    "A100-80GB": "A100",
    "RTX-PRO-6000": "RTX PRO 6000",
}
COLUMNS = 4
PRECISIONS = {"bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4"}
KERNELS = {"fa3": "FA3", "trtllm_mha": "TRT-LLM", "flashinfer": "FlashInfer"}


@dataclass(frozen=True)
class Sampler:
    pool: str  # the per-pool figures' key, "<pool name>:<weight view>"
    accelerator: str
    family: str  # the per-pool figures' attention-kernel family
    precision: str
    kernel: str
    share: float


def samplers(pools: Sequence[Any]) -> list[Sampler]:
    """One entry per pool, with its share of the router's sessions, in display order:
    by attention-kernel family as the per-pool figures order them, then accelerator,
    then precision from highest to lowest."""
    weights = [pool.min_containers * pool.target_inputs for pool in pools]
    total = sum(weights)
    entries = []
    for pool, weight in zip(pools, weights, strict=True):
        key = f"{pool.name}:{pool.weight_view}"
        entries.append(
            Sampler(
                pool=key,
                accelerator=ACCELERATORS[pool.gpu],
                family=POOLS[key][2],
                precision=PRECISIONS[pool.weight_view],
                kernel=KERNELS[pool.sglang_args["--attention-backend"]],
                share=weight / total,
            )
        )
    precision_order = list(PRECISIONS.values())
    accelerator_order = list(ACCELERATORS.values())
    family_order = list(FAMILIES)
    return sorted(
        entries,
        key=lambda s: (
            family_order.index(s.family),
            accelerator_order.index(s.accelerator),
            precision_order.index(s.precision),
        ),
    )


def draw_setup(entries: Sequence[Sampler], trainer_gpu: str) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    box_w, box_h, gap = 2.15, 1.55, 0.2
    row_h = box_h + 0.4
    left = 0.45
    rows = [entries[i : i + COLUMNS] for i in range(0, len(entries), COLUMNS)]
    pool_right = left + min(len(entries), COLUMNS) * (box_w + gap) - gap
    height = len(rows) * row_h + 0.2
    trainer_x, trainer_w = pool_right + 3.0, 2.4
    width = trainer_x + trainer_w + 0.3

    fig, ax = plt.subplots(figsize=(width * 0.92, (height + 1.5) * 0.92))
    ax.set_xlim(0, width)
    ax.set_ylim(-1.35, height + 0.75)
    ax.axis("off")
    biggest = max(s.share for s in entries)

    for row, row_entries in enumerate(rows):
        y = height - (row + 1) * row_h + 0.2
        for col, s in enumerate(row_entries):
            x = left + col * (box_w + gap)
            face = POOL_SHADES[s.pool]
            ink = "white" if face in DARK_SHADES else "#111827"
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    box_w,
                    box_h,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=face,
                    edgecolor="#374151",
                    linewidth=0.8,
                )
            )
            ax.text(
                x + 0.12,
                y + box_h - 0.24,
                s.accelerator,
                ha="left",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=ink,
            )
            ax.text(
                x + 0.12,
                y + box_h - 0.56,
                f"{s.precision} weights",
                ha="left",
                va="center",
                fontsize=9,
                color=ink,
            )
            ax.text(
                x + 0.12,
                y + box_h - 0.84,
                f"{s.kernel} attention",
                ha="left",
                va="center",
                fontsize=9,
                color=ink,
            )
            # The share as a bar on a common scale, so boxes compare at a glance.
            bar_w = (box_w - 0.24) * s.share / biggest
            ax.add_patch(
                Rectangle(
                    (x + 0.12, y + 0.13),
                    box_w - 0.24,
                    0.14,
                    facecolor="white",
                    alpha=0.35,
                    linewidth=0,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x + 0.12, y + 0.13),
                    bar_w,
                    0.14,
                    facecolor=ink,
                    alpha=0.85,
                    linewidth=0,
                )
            )
            ax.text(
                x + 0.12,
                y + 0.42,
                f"{s.share:.0%} of samples",
                ha="left",
                va="center",
                fontsize=8.5,
                color=ink,
            )

    # One dashed frame around every sampler: together they are the pool.
    top = height - row_h + 0.2 + box_h + 0.25
    bottom = height - len(rows) * row_h + 0.2 - 0.25
    frame_right = pool_right + 0.25
    ax.add_patch(
        FancyBboxPatch(
            (left - 0.25, bottom),
            frame_right - (left - 0.25),
            top - bottom,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor="none",
            edgecolor="#6b7280",
            linewidth=1.3,
            linestyle=(0, (5, 4)),
        )
    )
    ax.text(
        left - 0.15,
        top + 0.15,
        "Samplers",
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#374151",
    )

    mid = height / 2 + 0.1
    ax.add_patch(
        FancyBboxPatch(
            (trainer_x, mid - 0.85),
            trainer_w,
            1.7,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#f3f4f6",
            edgecolor="#111827",
            linewidth=1.2,
        )
    )
    ax.text(
        trainer_x + trainer_w / 2,
        mid + 0.45,
        "Trainer",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        trainer_x + trainer_w / 2,
        mid + 0.05,
        "BF16 weights",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.text(
        trainer_x + trainer_w / 2,
        mid - 0.3,
        f"on {trainer_gpu}",
        ha="center",
        va="center",
        fontsize=9,
    )

    arrow = {
        "arrowstyle": "-|>",
        "mutation_scale": 18,
        "linewidth": 1.6,
        "color": "#111827",
    }
    ax.add_patch(
        FancyArrowPatch(
            (frame_right + 0.1, mid + 0.35), (trainer_x - 0.1, mid + 0.35), **arrow
        )
    )
    ax.text(
        (frame_right + trainer_x) / 2,
        mid + 0.55,
        "trajectories, with each\nsampler's log-probabilities",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.add_patch(
        FancyArrowPatch(
            (trainer_x - 0.1, mid - 0.35), (frame_right + 0.1, mid - 0.35), **arrow
        )
    )
    ax.text(
        (frame_right + trainer_x) / 2,
        mid - 0.55,
        "new weights, in each pool's\nprecision: BF16, FP8, NVFP4",
        ha="center",
        va="top",
        fontsize=9,
    )

    notes = (
        "Fully asynchronous: the samplers keep generating while the trainer updates, "
        "so a trajectory can span several weight versions.",
        "Every sampler keeps its KV cache in FP8. A router assigns each agent session to "
        "one pool, in proportion to its serving capacity, and the session stays there.",
    )
    for i, note in enumerate(notes):
        ax.text(
            0.05,
            -0.55 - 0.38 * i,
            note,
            ha="left",
            va="center",
            fontsize=8.5,
            color="#4b5563",
        )
    return fig


def main() -> None:
    from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_hetero as hetero

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out / "setup"
    out.mkdir(parents=True, exist_ok=True)
    fig = draw_setup(samplers(hetero.modal.rollout_pools), hetero.modal.gpu)
    for fmt in ("png", "svg"):
        fig.savefig(out / f"fig0_setup.{fmt}", dpi=200, bbox_inches="tight")
    print(out / "fig0_setup.png")


if __name__ == "__main__":
    main()
