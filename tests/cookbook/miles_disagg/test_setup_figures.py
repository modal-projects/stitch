import math

import pytest

from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_hetero as hetero
from cookbook.miles_disagg.figures import setup_figures
from cookbook.miles_disagg.figures.training_figures import POOL_SHADES


def test_every_pool_is_one_sampler_with_its_router_share():
    entries = setup_figures.samplers(hetero.modal.rollout_pools)

    assert len(entries) == len(hetero.modal.rollout_pools)
    assert math.isclose(sum(s.share for s in entries), 1.0)
    # Each pool keeps the color it has in the per-pool figures.
    assert {s.pool for s in entries} == set(POOL_SHADES)
    shares = {(s.accelerator, s.precision): s.share for s in entries}
    # Sessions per pool are engines times sessions per engine.
    assert shares["B300", "NVFP4"] == pytest.approx(384 / 1376)
    assert shares["A100", "BF16"] == pytest.approx(48 / 1376)


def test_samplers_run_in_the_per_pool_figures_order_with_hopper_in_two_precisions():
    entries = setup_figures.samplers(hetero.modal.rollout_pools)

    assert [(s.accelerator, s.precision) for s in entries] == [
        ("H100", "BF16"),
        ("H100", "FP8"),
        ("H200", "BF16"),
        ("H200", "FP8"),
        ("B200", "NVFP4"),
        ("B300", "NVFP4"),
        ("A100", "BF16"),
        ("RTX PRO 6000", "BF16"),
    ]
    # One row per family once the grid wraps at four boxes.
    assert {s.kernel for s in entries[: setup_figures.COLUMNS]} == {"FA3"}


def test_the_setup_figure_draws(tmp_path):
    pytest.importorskip("matplotlib")
    entries = setup_figures.samplers(hetero.modal.rollout_pools)

    fig = setup_figures.draw_setup(entries, "B200")
    fig.savefig(tmp_path / "setup.png")

    assert (tmp_path / "setup.png").stat().st_size > 0
