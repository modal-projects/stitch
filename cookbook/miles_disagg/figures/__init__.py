"""Training-curve figures for the heterogeneous study, from W&B.

A record run can span several trainer attempts (a resume after Modal's 24 h limit is a
new W&B run), so a run here is an ordered list of W&B runs; ``history.stitch`` joins
them into one curve per step axis. Run with the plotting extras:

    uv run --with wandb --with matplotlib python -m cookbook.miles_disagg.figures \\
        --out ~/blog-figures [--runs runs.json]
"""
