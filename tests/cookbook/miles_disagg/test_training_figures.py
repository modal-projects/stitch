import pytest

from cookbook.miles_disagg.figures import training_figures as tf
from cookbook.miles_disagg.figures.experiments import BY_KEY

H100_FP8 = "ServerH100FP8:fp8"
B300_NVFP4 = "ServerB300NVFP4W4A16:nvfp4"


def _history(metric, values, *, axis="train/step"):
    return [{axis: step, metric: value} for step, value in enumerate(values)]


def _pool_row(step, source, **fields):
    return {
        "rollout/step": step,
        **{f"rollout/by_source/{source}/{key}": value for key, value in fields.items()},
    }


def test_relative_panels_divide_by_the_first_steps_mean():
    panel = tf.Panel("train/grad_norm", "g", relative=True)
    rows = _history("train/grad_norm", [2.0, 2.0, 2.0, 2.0, 2.0, 8.0])

    steps, raw, smooth = tf.series_for(rows, panel, smoothing=0.0)

    assert steps == [0, 1, 2, 3, 4, 5]
    assert raw == [1.0, 1.0, 1.0, 1.0, 1.0, 4.0]
    assert smooth == raw


def test_failure_reads_the_logged_share_of_repeated_format_errors():
    (figure,) = [f for f in tf.FIGURES if f.name == "p1_failure"]
    assert tf.FORMAT_ERRORS in [panel.metric for panel in figure.panels]
    assert tf.FORMAT_ERRORS == "rollout_agent/exit_status/RepeatedFormatError_ratio"


def test_window_mean_uses_only_steps_inside_the_window():
    rows = _history("m", [10.0, 1.0, 2.0, 3.0, 100.0])

    assert tf.window_mean(rows, "m", (1, 3)) == pytest.approx(2.0)
    assert tf.window_mean(rows, "absent", (1, 3)) is None


def test_pool_table_weights_per_sample_fields_and_shares_samples():
    rows = [
        _pool_row(
            1,
            H100_FP8,
            sample_count=10,
            raw_reward_mean=1.0,
            staleness_mean=2.0,
            **{"exit_status/RepeatedFormatError_ratio": 0.5},
        ),
        _pool_row(
            2, H100_FP8, sample_count=30, raw_reward_mean=0.0, staleness_mean=4.0
        ),
        _pool_row(
            1, B300_NVFP4, sample_count=60, raw_reward_mean=0.5, staleness_mean=1.0
        ),
        {"train/step": 1, f"train/train_rollout_kl/by_source/{H100_FP8}": 0.01},
        {"train/step": 2, f"train/train_rollout_kl/by_source/{H100_FP8}": 0.03},
    ]

    table = {entry["pool"]: entry for entry in tf.pool_table(rows)}

    h100 = table[tf.POOLS[H100_FP8][0]]
    assert h100["sample_share"] == pytest.approx(0.4)
    assert h100["reward"] == pytest.approx(10 / 40)
    assert h100["staleness"] == pytest.approx((20 + 120) / 40)
    assert h100["kl"] == pytest.approx(0.02)
    # Step 2 logged no format-error status for H100: none of its 30 episodes.
    assert h100["format_errors"] == pytest.approx(5 / 40)
    assert table[tf.POOLS[B300_NVFP4][0]]["format_errors"] == 0.0
    assert table[tf.POOLS[B300_NVFP4][0]]["sample_share"] == pytest.approx(0.6)
    # A pool with no samples is listed, empty.
    assert (
        table["A100, BF16"]["sample_share"] == 0.0
        and table["A100, BF16"]["reward"] is None
    )


def test_mismatch_compares_the_uniform_pool_with_each_mixed_pool_sampler():
    histories = {
        "l0_sc_mis_top_p": _history("train/train_rollout_kl/all", [0.0, 0.002, 0.002]),
        "l2_sc_mis_top_p": _history(
            f"train/train_rollout_kl/by_source/{H100_FP8}", [0.0, 0.008, 0.008]
        )
        + _history("train/train_rollout_kl/lag_0_1", [0.0, 0.005, 0.005]),
    }

    values = tf.mismatch_values(histories, window=(1, 2))

    # The final recipe on both pools.
    assert tf.MISMATCH_POOLED == tf.FINAL_RECIPE[0]
    assert [key for key, _ in tf.MISMATCH_FLEETS] == [tf.FINAL_RECIPE[1]]
    assert [round(v, 4) for _, v in values["fleets"]] == [0.002]
    pools = {name: value for name, _, value, _ in values["pools"]}
    assert pools[tf.POOLS[H100_FP8][0]] == pytest.approx(0.008)
    assert pools["A100, BF16"] is None
    assert dict(values["lags"])["0-1"] == pytest.approx(0.005)


def test_a_run_without_the_metric_is_skipped_but_still_in_the_legend():
    pytest.importorskip("matplotlib")
    from matplotlib.legend import Legend

    figure = tf.FIGURES[0]
    histories = {
        "l2_grpo": _history("rollout/raw_reward", [0.4, 0.3, 0.2], axis="rollout/step"),
        "l2_icepop": _history("train/grad_norm", [1.0, 1.0]),  # no panel metric
    }

    fig = tf.draw(figure, histories, smoothing=0.5)

    (legend,) = [artist for artist in fig.legends if isinstance(artist, Legend)]
    # Blank entries only pad the legend's family columns.
    labels = [text.get_text() for text in legend.get_texts() if text.get_text().strip()]
    assert len(labels) == len(figure.keys)
    assert BY_KEY["l2_grpo"].label in labels
    assert f"{BY_KEY['l0_sc_mis_top_p'].label} (no data yet)" in labels
    # GRPO's raw and smoothed series only; IcePop has no reward rows.
    assert len(fig.axes[0].get_lines()) == 2


def test_the_mismatch_window_ends_where_the_shorter_run_does():
    histories = {
        "l0_sc_mis_top_p": _history("train/train_rollout_kl/all", [0.001] * 5),
        "l2_sc_mis_top_p": _history("train/train_rollout_kl/all", [0.003] * 8),
    }

    assert tf.mismatch_window(histories) == (1, 4)
    assert tf.mismatch_values(histories)["window"] == [1, 4]


def test_the_mismatch_figure_draws_with_missing_samplers():
    pytest.importorskip("matplotlib")
    histories = {
        "l0_sc_mis_top_p": _history("train/train_rollout_kl/all", [0.0, 0.002]),
        "l2_sc_mis_top_p": _history(
            "train/train_rollout_kl/bf16", [0.003, 0.004, 0.005]
        ),
    }

    fig = tf.draw_mismatch(histories)

    assert len(fig.axes) == 3


def test_the_uniform_hardware_run_is_dashed_and_mixed_pool_runs_solid():
    for key in tf.MAIN_TRAINING:
        expected = "--" if BY_KEY[key].fleet == "L0" else "-"
        assert tf.style(BY_KEY[key])["linestyle"] == expected


def test_pool_shares_sum_to_one_at_every_step_and_weight_gradient_samples():
    rows = []
    for step, (h100, b300) in enumerate([(10, 30), (20, 20), (30, 10)]):
        row = {"rollout/step": step}
        for source, count, grad in ((H100_FP8, h100, 50.0), (B300_NVFP4, b300, 100.0)):
            row[f"rollout/training_batch/{source}/sample_count"] = count
            row[f"rollout/training_batch/{source}/gradient_sample_percentage"] = grad
        rows.append(row)

    steps, shares = tf.pool_shares(rows, "sample_count", smoothing=0.0)
    _, gradient = tf.pool_shares(rows, "gradient", smoothing=0.0)

    assert steps == [0, 1, 2]
    for i in range(3):
        assert sum(shares[source][i] for source in tf.POOLS) == pytest.approx(1.0)
    assert shares[H100_FP8] == pytest.approx([0.25, 0.5, 0.75])
    # Half of the H100 samples carry gradient, all of the B300 ones.
    assert gradient[H100_FP8][0] == pytest.approx(5 / (5 + 30))


def test_the_pool_figures_draw_from_partial_data():
    pytest.importorskip("matplotlib")
    rows = [
        {
            "rollout/step": 1,
            f"rollout/training_batch/{H100_FP8}/sample_count": 10,
            f"rollout/by_source/{H100_FP8}/sample_count": 10,
            f"rollout/by_source/{H100_FP8}/exit_status/RepeatedFormatError_ratio": 0.2,
            f"rollout/by_source/{H100_FP8}/within_prompt_reward_delta_mean": -0.02,
        },
        {"train/step": 1, f"train/train_rollout_kl/by_source/{H100_FP8}": 0.017},
    ]

    shares = tf.draw_pool_shares(rows)
    problems = tf.draw_pool_problems(rows, {"l2_sc_mis": rows})
    views = tf.draw_view_updates(rows)

    assert len(shares.axes) == 1
    assert len(problems.axes) == 1
    labels = [tick.get_text() for tick in problems.axes[0].get_yticklabels()]
    assert not any("format errors" in label for label in labels)
    assert len(views.axes) == 2


def test_problem_shares_weight_each_pool_by_its_samples_and_tokens():
    rows = [
        _pool_row(1, H100_FP8, sample_count=10, **{tf.FORMAT_ERROR_FIELD: 0.3}),
        # No episode of B300 ended on format errors, so the status is not logged.
        _pool_row(1, B300_NVFP4, sample_count=30),
        {
            "rollout/step": 1,
            f"rollout/training_batch/{H100_FP8}/token_count": 100,
            f"rollout/training_batch/{B300_NVFP4}/token_count": 300,
        },
        {
            "train/step": 1,
            f"train/train_rollout_ratio_tail_frac/by_source/{H100_FP8}": 0.03,
            f"train/train_rollout_ratio_tail_frac/by_source/{B300_NVFP4}": 0.001,
        },
    ]

    shares = tf.problem_shares(rows)

    assert shares["samples"][H100_FP8] == pytest.approx(0.25)
    assert shares["tokens"][B300_NVFP4] == pytest.approx(0.75)
    assert shares["format_errors"][H100_FP8] == pytest.approx(1.0)
    assert shares["ratio_outside"][H100_FP8] == pytest.approx(3 / (3 + 0.3))
    for by_source in shares.values():
        assert sum(by_source.values()) == pytest.approx(1.0)


def test_base_format_errors_read_step_zero_of_mixed_pool_runs_only():
    mixed = [
        _pool_row(0, H100_FP8, sample_count=10, **{tf.FORMAT_ERROR_FIELD: 0.2}),
        _pool_row(0, B300_NVFP4, sample_count=30),
        _pool_row(1, H100_FP8, sample_count=10, **{tf.FORMAT_ERROR_FIELD: 0.9}),
    ]
    uniform = [_pool_row(0, H100_FP8, sample_count=5, **{tf.FORMAT_ERROR_FIELD: 0.5})]

    base = tf.base_format_errors(
        {"l2_grpo": mixed, "l0_sc_mis_top_p": uniform},
        ("l2_grpo", "l0_sc_mis_top_p"),
    )

    assert base[H100_FP8] == [("l2_grpo", pytest.approx(0.2))]
    assert base[B300_NVFP4] == [("l2_grpo", 0.0)]
    assert base["ServerA100BF16TP2:bf16"] == []


def test_every_pool_has_a_family_with_its_attention_kernel():
    for name, precision, family in tf.POOLS.values():
        assert family in tf.FAMILIES, name
        assert precision in tf.PRECISION_COLORS, name
    hopper = {name for name, _, family in tf.POOLS.values() if family == "hopper"}
    assert hopper == {"H100, FP8", "H200, FP8", "H100, BF16", "H200, BF16"}


def test_family_kernels_match_the_hetero_config():
    from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_hetero as hetero

    backends = {"FA3": "fa3", "TRT-LLM": "trtllm_mha", "FlashInfer": "flashinfer"}
    configured = {
        f"{pool.name}:{pool.weight_view}": pool.sglang_args["--attention-backend"]
        for pool in hetero.modal.rollout_pools
    }
    assert set(configured) == set(tf.POOLS)
    for source, (_, _, family) in tf.POOLS.items():
        assert configured[source] == backends[tf.FAMILIES[family][1]], source


def test_the_two_pool_figure_tells_the_pools_apart_by_color():
    (figure,) = [f for f in tf.FIGURES if f.name == "p4_uniform_hardware"]
    colors = dict(figure.colors)

    assert set(colors) == set(tf.FINAL_RECIPE)
    assert colors[tf.FINAL_RECIPE[0]] != colors[tf.FINAL_RECIPE[1]]
    assert dict(figure.labels) == {
        tf.FINAL_RECIPE[0]: "mixed pool",
        tf.FINAL_RECIPE[1]: "all-B200 BF16",
    }


def test_a_figure_can_stop_its_lines_at_a_step():
    pytest.importorskip("matplotlib")
    (figure,) = [f for f in tf.FIGURES if f.name == "p4_uniform_hardware"]
    assert figure.last_step == 160
    rows = [
        {
            "rollout/step": step,
            "rollout/raw_reward": 0.4,
            "train/step": step,
            "train/grad_norm": 0.01,
        }
        for step in (150, 160, 170, 180)
    ]

    fig = tf.draw(figure, {key: rows for key in figure.keys})

    reward = fig.axes[0]
    assert max(max(line.get_xdata()) for line in reward.get_lines()) == 160


def test_the_source_figure_holds_one_source_fixed_per_panel():
    pytest.importorskip("matplotlib")
    pooled = [
        {
            "train/step": 1,
            f"train/train_rollout_kl/by_source/{H100_FP8}": 0.014,
            "train/train_rollout_kl/lag_0_1": 0.005,
        },
        {
            "train/step": 2,
            f"train/train_rollout_kl/by_source/{H100_FP8}": 0.012,
            "train/train_rollout_kl/lag_0_1": 0.006,
        },
    ]
    uniform = [
        {"train/step": 1, "train/train_rollout_kl/all": 0.0013},
        {"train/step": 2, "train/train_rollout_kl/all": 0.0013},
    ]
    histories = {tf.MISMATCH_POOLED: pooled, tf.MISMATCH_FLEETS[0][0]: uniform}

    numbers = tf.mismatch_by_source(histories)
    fig = tf.draw_mismatch_sources(histories)

    assert numbers["samplers"][H100_FP8][0] == pytest.approx(0.013)
    assert numbers["uniform"][0] == pytest.approx(0.0013)
    # Samplers with no data (e.g. a pool that never logged) leave a gap, not an error.
    assert numbers["samplers"][B300_NVFP4][0] is None
    assert len(fig.axes) == 3
