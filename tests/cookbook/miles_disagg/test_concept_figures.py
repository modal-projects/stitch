import pytest

from cookbook.miles_disagg.figures import concept_figures as cf


def test_push_follows_the_blog_formula_and_sums_to_zero():
    pushes = cf.push((0.5, 0.5), (1.0, 0.0))

    assert pushes == pytest.approx([0.25, -0.25])
    assert sum(cf.push(cf.SAMPLER, cf.VALUES)) == pytest.approx(0.0)


def test_the_abandoned_token_is_pushed_harder_under_the_stale_sampler():
    on_policy = cf.push(cf.TRAINER, cf.VALUES)
    under_sampler = cf.push(cf.SAMPLER, cf.VALUES)
    abandoned = cf.TOKENS.index("D")

    assert cf.TRAINER[abandoned] < cf.SAMPLER[abandoned]
    assert abs(under_sampler[abandoned]) > 5 * abs(on_policy[abandoned])
    assert sum(cf.TRAINER) == pytest.approx(1.0)
    assert sum(cf.SAMPLER) == pytest.approx(1.0)


def test_the_figure_draws_two_panels():
    pytest.importorskip("matplotlib")

    assert len(cf.draw_wrong_measure().axes) == 2
