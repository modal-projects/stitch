import pytest

from cookbook.miles_disagg.bench import configs
from cookbook.miles_disagg.figures import cost_figures as cf

TRAINING_POOLS = (
    "ServerB200NVFP4W4A16",
    "ServerB300NVFP4W4A16",
    "ServerH200FP8",
    "ServerH200BF16",
    "ServerH100FP8",
    "ServerH100BF16TP2",
    "ServerA100BF16TP2",
    "ServerRTXPRO6000BF16TP2",
    "ServerB200BF16",
)


def _point(config, variant, sessions, latency, requests_per_gpu, **extra):
    return {
        "run_id": "r",
        "config": config,
        "variant": variant,
        "sessions": sessions,
        "latency_s_p90": latency,
        "output_tok_s_per_gpu": requests_per_gpu * 200,
        "requests_s_per_gpu": requests_per_gpu,
        **extra,
    }


def test_every_training_pool_maps_to_the_benchmark_row_that_serves_it_as_run():
    pools = cf.pool_configs()

    assert set(TRAINING_POOLS) <= set(pools)
    assert pools["ServerB200BF16"] == "b200-bf16"
    assert pools["ServerA100BF16TP2"] == "a100-bf16"


def test_an_engine_costs_its_gpus_at_list_price():
    assert cf.container_price(configs.CONFIGS["b200-bf16"]) == pytest.approx(6.25)
    assert cf.container_price(configs.CONFIGS["a100-bf16"]) == pytest.approx(2 * 2.50)


def test_as_run_prices_the_base_variant_and_projected_the_cheapest_measured_one():
    rows = [
        _point("b200-bf16", "base", 8, 4.0, 3.0),
        _point("b200-bf16", "base", 16, 6.0, 5.0),
        _point("b200-bf16", "base", 24, 12.0, 4.0),
        _point("b200-bf16", "cg-64", 16, 6.0, 6.0),
        # The client, not the engine, limited this point: it never counts.
        _point("b200-bf16", "cg-64", 24, 8.0, 9.0, stop_reason="client_bound"),
    ]
    price = cf.container_price(configs.CONFIGS["b200-bf16"])

    as_run = cf.turn_prices(rows, projected=False)["b200-bf16"]
    projected = cf.turn_prices(rows, projected=True)["b200-bf16"]

    # 10 s falls between 6 s (5 turns/s) and 12 s (4 turns/s); the best point is 5.
    assert as_run.variant == "base"
    assert as_run.turns_per_container_hour == pytest.approx(5.0 * 3600)
    assert as_run.usd == pytest.approx(price / (5.0 * 3600))
    assert projected.variant == "cg-64"
    assert projected.usd == pytest.approx(price / (6.0 * 3600))


def test_turns_per_step_are_samples_times_turns_per_sample_by_pool():
    rows = [
        {
            "rollout/step": 3,
            "rollout/by_source/ServerH200FP8:fp8/sample_count": 100,
            "rollout/by_source/ServerH200FP8:fp8/turns_mean": 40.0,
            "rollout/by_source/ServerA100BF16TP2:bf16/sample_count": 0,
            "rollout/by_source/ServerA100BF16TP2:bf16/turns_mean": 30.0,
        },
        {"train/step": 3, "train/loss": 0.1},
    ]

    assert cf.turns_by_step(rows) == {3: {"ServerH200FP8": 4000.0}}


def test_version_v_costs_the_rollout_steps_it_trained_on():
    prices = {
        "h200-fp8": cf.TurnPrice("h200-fp8", "base", 1000.0, 5.0),
        "a100-bf16": cf.TurnPrice("a100-bf16", "base", 500.0, 5.0),
    }
    pools = {"ServerH200FP8": "h200-fp8", "ServerA100BF16TP2": "a100-bf16"}
    turns = {0: {"ServerH200FP8": 1000.0}, 1: {"ServerA100BF16TP2": 500.0}}

    cost = cf.cumulative_cost(turns, [(0, prices)], pools)

    assert cost == {0: 0.0, 1: pytest.approx(5.0), 2: pytest.approx(10.0)}
    with pytest.raises(KeyError, match="ServerB300"):
        cf.cumulative_cost({0: {"ServerB300NVFP4W4A16": 1.0}}, [(0, prices)], pools)


def test_a_step_between_trace_anchors_interpolates_its_cost_per_turn():
    early = {"h200-fp8": cf.TurnPrice("h200-fp8", "base", 1000.0, 5.0)}  # $0.005
    late = {"h200-fp8": cf.TurnPrice("h200-fp8", "base", 500.0, 5.0)}  # $0.010
    anchors = [(140, late), (0, early)]

    assert cf.usd_per_turn("h200-fp8", 0, anchors) == pytest.approx(0.005)
    assert cf.usd_per_turn("h200-fp8", 70, anchors) == pytest.approx(0.0075)
    assert cf.usd_per_turn("h200-fp8", 200, anchors) == pytest.approx(0.010)
    with pytest.raises(KeyError, match="a100"):
        cf.usd_per_turn("a100-bf16", 70, anchors)


def test_both_fleets_draw_as_run_in_each_panel():
    pytest.importorskip("matplotlib")
    from types import SimpleNamespace

    from cookbook.miles_disagg.figures import eval_figures
    from cookbook.miles_disagg.figures.experiments import BY_KEY

    class Point(SimpleNamespace):
        def scores(self, subset, hard):
            return {"pass@4": self.value}

    base = Point(version=0, n_samples=4, value=0.5)
    series = [
        eval_figures.Series(BY_KEY[key], (Point(version=20, n_samples=4, value=0.6),))
        for key in cf.RUNS
    ]
    costs = {
        (key, scenario): {0: 0.0, 20: 1000.0}
        for key in cf.RUNS
        for scenario in ("as run", "projected")
    }

    fig = cf.draw(series, base, frozenset(), costs)

    assert [ax.get_title() for ax in fig.axes] == ["HARD-51 subset", "Full set"]
    (legend,) = fig.legends
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["Mixed pool", "All-B200 BF16", "Base model"]
    curves = [
        line for line in fig.axes[0].get_lines() if line.get_gid() != "base-model"
    ]
    assert [list(line.get_xdata()) for line in curves] == [[0.0, 1.0]] * 2
