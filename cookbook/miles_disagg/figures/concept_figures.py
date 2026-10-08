"""Concept figures: illustrations from the blog's equations, not from run data.

``wrong_measure``: at one prefix, the expected push on each candidate token's logit, on-policy
and under a stale sampler, with drift removed (blog section 5):

    on-policy:      dz_v ∝ p(v) (Q(v) - E_p[Q])
    under sampler:  dz_v ∝ q(v) (Q(v) - E_q[Q])

    uv run --with matplotlib python -m cookbook.miles_disagg.figures.concept_figures \\
        --out ~/blog-figures
Writes OUT/concept/wrong_measure.{png,svg}.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# A toy prefix: the trainer has sharpened onto token A and nearly abandoned token D, while a
# stale sampler still draws D often. Values Q are the expected reward after each token.
TOKENS = ("A", "B", "C", "D")
TRAINER = (0.60, 0.25, 0.13, 0.02)
SAMPLER = (0.45, 0.25, 0.12, 0.18)
VALUES = (1.0, 0.6, 0.2, 0.0)
TRAINER_COLOR = "#15803d"
SAMPLER_COLOR = "#94a3b8"


def push(probs: Sequence[float], values: Sequence[float]) -> list[float]:
    """The expected push on each logit when tokens are drawn from ``probs`` with drift
    removed: probs(v) * (Q(v) - E_probs[Q])."""
    mean = sum(p * q for p, q in zip(probs, values, strict=True))
    return [p * (q - mean) for p, q in zip(probs, values, strict=True)]


def draw_wrong_measure() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (prob_ax, push_ax) = plt.subplots(1, 2, figsize=(10, 3.6))
    x = list(range(len(TOKENS)))
    width = 0.38
    prob_ax.bar(
        [i - width / 2 for i in x],
        TRAINER,
        width,
        color=TRAINER_COLOR,
        label="trainer, $p_\\theta$",
    )
    prob_ax.bar(
        [i + width / 2 for i in x],
        SAMPLER,
        width,
        color=SAMPLER_COLOR,
        label="stale sampler, $q$",
    )
    prob_ax.set_title("probability of each candidate token", fontsize=10)
    on_policy, under_sampler = push(TRAINER, VALUES), push(SAMPLER, VALUES)
    push_ax.bar(
        [i - width / 2 for i in x],
        on_policy,
        width,
        color=TRAINER_COLOR,
        label="on-policy",
    )
    push_ax.bar(
        [i + width / 2 for i in x],
        under_sampler,
        width,
        color=SAMPLER_COLOR,
        label="under the stale sampler",
    )
    push_ax.axhline(0.0, color="#374151", linewidth=0.8)
    push_ax.set_title("expected push on each token's logit", fontsize=10)
    for ax in (prob_ax, push_ax):
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{t}\nQ = {q:g}" for t, q in zip(TOKENS, VALUES, strict=True)]
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("The wrong measure, at one prefix", fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out / "concept"
    out.mkdir(parents=True, exist_ok=True)
    fig = draw_wrong_measure()
    for fmt in ("png", "svg"):
        fig.savefig(out / f"wrong_measure.{fmt}", dpi=200, bbox_inches="tight")
    print(f"wrote {out}/wrong_measure.png")


if __name__ == "__main__":
    main()
