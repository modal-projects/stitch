"""The sweep's ramp, stop rules, and throughput-at-speed summary."""

import asyncio
import json

import pytest

from cookbook.miles_disagg.bench import configs, replay, sweep
from cookbook.miles_disagg.bench.fake_server import FakeEngine, synthetic_trajectory


def _point(sessions, speed, throughput, **extra):
    return {
        "sessions": sessions,
        "decode_tok_s_p50": speed,
        "output_tok_s_per_gpu": throughput,
        **extra,
    }


def _turns(sessions, latency_p90, throughput, **extra):
    return {
        "sessions": sessions,
        "latency_s_p90": latency_p90,
        "output_tok_s_per_gpu": throughput,
        **extra,
    }


# The rate metric these tests were written against (higher is faster).
RATE = {"speed_key": "decode_tok_s_p50"}
RATE_PLAN = {"speed_metric": "decode_tok_s_p50", "targets": (60.0, 30.0)}


def test_throughput_interpolates_between_the_points_straddling_the_target():
    curve = [
        _point(8, 120, 800),
        _point(16, 100, 1500),
        _point(32, 70, 2200),
        _point(48, 50, 2600),
    ]

    assert sweep.throughput_at_speed(curve, 60, **RATE) == (2400.0, "crossed")
    # A target met exactly is a measured point.
    assert sweep.throughput_at_speed(curve, 70, **RATE) == (2200.0, "crossed")
    # Every point is faster than 30 tok/s: the best is only a lower bound.
    assert sweep.throughput_at_speed(curve, 30, **RATE) == (2600.0, "lower_bound")
    assert sweep.throughput_at_speed(curve, 150, **RATE) == (None, "unreached")


def test_throughput_at_a_turn_latency_bound_keeps_the_load_before_the_kv_cliff():
    # Shaped like B200 BF16's validation: output peaks at 16 sessions per engine, then
    # the KV cache fills and p90 turn latency jumps.
    curve = [
        _turns(8, 3.8, 865),
        _turns(16, 5.5, 1092),
        _turns(24, 9.0, 1000),
        _turns(32, 38.0, 277),
    ]

    # Interpolating to 10 s between 24 and 32 gives ~975; 16 sessions measured more.
    assert sweep.throughput_at_speed(curve, 10) == (1092.0, "crossed")
    # Past the peak, a looser bound buys nothing.
    assert sweep.throughput_at_speed(curve, 20) == (1092.0, "crossed")
    assert sweep.throughput_at_speed(curve, 60) == (1092.0, "lower_bound")
    assert sweep.throughput_at_speed(curve, 3) == (None, "unreached")


def test_throughput_interpolates_in_latency_while_output_still_grows():
    curve = [_turns(8, 4.0, 800), _turns(16, 8.0, 1200), _turns(24, 16.0, 1400)]

    # 10 s lies a quarter of the way from 8 s to 16 s: 1200 + 0.25 * 200.
    assert sweep.throughput_at_speed(curve, 10) == (pytest.approx(1250.0), "crossed")


def test_throughput_takes_the_best_load_when_the_curve_is_not_monotonic():
    curve = [
        _point(8, 90, 700),
        _point(16, 55, 1100),
        _point(24, 65, 1300),
        _point(32, 40, 1400),
    ]

    value, kind = sweep.throughput_at_speed(curve, 60, **RATE)

    # Between 24 (65 tok/s) and 32 (40 tok/s): 1300 + 5/25 * 100.
    assert (value, kind) == (pytest.approx(1320.0), "crossed")


def test_throughput_reads_csv_strings_and_skips_empty_points():
    curve = [
        {"sessions": "8", "decode_tok_s_p50": "80", "output_tok_s_per_gpu": "600"},
        {"sessions": "16", "decode_tok_s_p50": "", "output_tok_s_per_gpu": "900"},
        {"sessions": "32", "decode_tok_s_p50": "40", "output_tok_s_per_gpu": "1000"},
    ]

    assert sweep.throughput_at_speed(curve, 60, **RATE) == (
        pytest.approx(800.0),
        "crossed",
    )
    assert sweep.throughput_at_speed(curve, 50, speed_key="e2e_tok_s_p50") == (
        None,
        "unreached",
    )


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ([_point(8, 29, 100)], "below_slowest_target"),
        ([_point(8, None, 100)], "below_slowest_target"),
        # One point that adds under 3% is not yet saturation; two in a row are.
        ([_point(8, 90, 500), _point(16, 70, 510)], None),
        ([_point(8, 90, 500), _point(16, 70, 510), _point(20, 65, 505)], "saturated"),
        ([_point(8, 90, 500), _point(16, 70, 510), _point(20, 65, 600)], None),
        ([_point(8, 90, 500, requests=90, errors=10)], "errors"),
        ([_point(8, 90, 500), _point(16, 70, 900, requests=100, errors=2)], None),
    ],
)
def test_ramp_stops_below_the_slowest_target_on_saturation_or_errors(rows, reason):
    assert sweep.stop_reason(rows, sweep.SweepConfig(**RATE_PLAN)) == reason


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ([_turns(8, 3.8, 865), _turns(16, 5.5, 1092)], None),
        # Over the loosest bound (20 s at p90) the ramp ends.
        ([_turns(8, 3.8, 865), _turns(32, 38.0, 277)], "below_slowest_target"),
        ([_turns(8, 3.8, 865), _turns(16, 15.0, 1092)], None),
    ],
)
def test_ramp_stops_once_p90_turn_latency_passes_the_loosest_bound(rows, reason):
    plan = sweep.SweepConfig()

    assert (plan.speed_metric, plan.targets) == ("latency_s_p90", (10.0, 20.0))
    assert sweep.stop_reason(rows, plan) == reason


def test_ramp_stops_when_the_client_wakes_late_and_the_summary_skips_that_point():
    healthy = _turns(8, 3.8, 865, loop_lag_s_p90=0.02)
    late = _turns(16, 5.5, 700, loop_lag_s_p90=0.46)

    assert sweep.stop_reason([healthy], sweep.SweepConfig()) is None
    assert sweep.stop_reason([healthy, late], sweep.SweepConfig()) == "client_bound"
    late["stop_reason"] = "client_bound"
    assert sweep.throughput_at_speed([healthy, late], 10.0) == (865, "lower_bound")


def test_rate_metrics_need_streamed_responses():
    parser = sweep.argparse.ArgumentParser()
    sweep.add_sweep_arguments(parser)

    with pytest.raises(SystemExit, match="needs --stream"):
        sweep.sweep_config_from_args(
            parser.parse_args(["--speed-metric", "decode_tok_s_p50"])
        )
    plan = sweep.sweep_config_from_args(
        parser.parse_args(["--speed-metric", "decode_tok_s_p50", "--stream"])
    )
    assert plan.speed_metric == "decode_tok_s_p50"
    assert sweep.sweep_config_from_args(parser.parse_args([])).speed_metric == (
        "latency_s_p90"
    )


def test_sweep_config_validates_and_caps_the_ramp_at_the_batch_ceiling():
    plan = sweep.SweepConfig()

    assert plan.sessions_up_to(64) == (8, 16, 20, 24, 28, 32, 48, 64)
    assert plan.sessions_up_to(4) == ()
    with pytest.raises(ValueError):
        sweep.SweepConfig(sessions_per_engine=(16, 8))
    with pytest.raises(ValueError):
        sweep.SweepConfig(speed_metric="ttft")
    with pytest.raises(ValueError):
        sweep.SweepConfig(targets=())
    with pytest.raises(ValueError):
        sweep.SweepConfig(saturation_patience=0)


def test_point_rows_report_output_per_gpu():
    summary = {
        "sessions": 64,
        "output_tok_s": 8000.0,
        "decode_tok_s_p50": 55.0,
        "error_kinds": {"timeout": 1},
        "server": {"rate:sglang:generation_tokens_total": 7900.0, "mean:x": 1.0},
    }
    labels = sweep.variant_labels(
        configs.CONFIGS["h100-bf16"], configs.CONFIGS["h100-bf16"].variants[0]
    )

    row = sweep.point_row(
        labels, summary, sessions_per_engine=32, engines=2, warmup_s=130
    )

    assert set(row) == set(sweep.POINT_COLUMNS)
    assert (row["gpus"], row["output_tok_s_per_gpu"]) == (4, 2000.0)
    assert (row["config"], row["variant"], row["price_per_gpu_hour"]) == (
        "h100-bf16",
        "base",
        3.95,
    )
    assert row["server_gen_tok_s"] == 7900.0
    assert json.loads(row["server_metrics"])["mean:x"] == 1.0
    assert json.loads(row["error_kinds"]) == {"timeout": 1}


def _rows(config, variant, run_id, curve):
    bench = configs.CONFIGS[config]
    labels = {**sweep.variant_labels(bench, bench.variant(variant)), "run_id": run_id}
    return [{**labels, **point} for point in curve]


def test_summary_is_relative_to_b200_bf16_in_throughput_and_cost(tmp_path):
    rows = [
        *_rows("b200-bf16", "base", "r1", [_point(8, 90, 1000), _point(16, 50, 1500)]),
        # A later run of the same variant replaces the earlier one.
        *_rows("h200-fp8", "base", "r0", [_point(8, 90, 9999), _point(16, 10, 9999)]),
        *_rows("h200-fp8", "base", "r2", [_point(8, 80, 700), _point(16, 40, 1000)]),
        *_rows("h200-fp8", "kv-bf16", "r2", [_point(8, 70, 600), _point(16, 50, 1200)]),
    ]
    for row in rows:
        sweep.append_csv(tmp_path / f"{row['config']}.csv", row)

    summary = sweep.summarize(
        sweep.read_csv(sorted(tmp_path.glob("*.csv"))), targets=(60.0, 30.0), **RATE
    )

    baseline, h200 = summary
    assert baseline["config"] == "b200-bf16"
    # B200 BF16 at 60 tok/s: 1000 + (90-60)/(90-50) * 500.
    assert baseline["tok_s_per_gpu@60"] == pytest.approx(1375.0)
    assert baseline["rel_throughput@60"] == pytest.approx(1.0)
    assert baseline["rel_cost_per_token@60"] == pytest.approx(1.0)
    # H200 FP8 at 60: base gives 700 + 20/40 * 300 = 850, kv-bf16 gives
    # 600 + 10/20 * 600 = 900, so kv-bf16 wins.
    assert h200["variant@60"] == "kv-bf16"
    assert h200["tok_s_per_gpu@60"] == pytest.approx(900.0)
    assert h200["kind@60"] == "crossed"
    assert h200["rel_throughput@60"] == pytest.approx(900 / 1375)
    assert h200["rel_cost_per_token@60"] == pytest.approx((4.54 / 900) / (6.25 / 1375))
    assert h200["usd_per_mtok@60"] == pytest.approx(4.54 / (900 * 3600) * 1e6)
    # At 30 tok/s every H200 point was faster: the best is a lower bound.
    assert (h200["tok_s_per_gpu@30"], h200["kind@30"]) == (
        pytest.approx(1200.0),
        "lower_bound",
    )


def test_summary_prices_a_thousand_turns_for_the_best_and_the_base_variant():
    base = [
        {**_point(8, 90, 1000), "requests_s": 2.0, "gpus": 1},
        {**_point(16, 50, 1500), "requests_s": 3.0, "gpus": 1},
    ]
    better = [
        {**_point(8, 90, 1200), "requests_s": 2.4, "gpus": 1},
        {**_point(16, 50, 1800), "requests_s": 3.6, "gpus": 1},
    ]
    rows = _rows("b200-bf16", "base", "r1", base) + _rows(
        "b200-bf16", "cg-64", "r1", better
    )

    (row,) = sweep.summarize(rows, targets=(30.0,), **RATE)

    # Every point meets 30 tok/s, so each variant's best (16 sessions) is used.
    assert row["variant@30"] == "cg-64"
    assert row["turns_per_gpu_hour@30"] == pytest.approx(3.6 * 3600)
    assert row["usd_per_kturn@30"] == pytest.approx(6.25 / (3.6 * 3600) * 1000)
    assert row["turns_per_gpu_hour_base@30"] == pytest.approx(3.0 * 3600)


def test_summary_without_the_baseline_leaves_relative_columns_empty():
    rows = _rows("a100-bf16", "base", "r1", [_point(8, 40, 300)])

    (row,) = sweep.summarize(rows, targets=(60.0, 30.0), **RATE)

    assert row["tok_s_per_gpu@30"] == 300.0
    assert row["rel_throughput@30"] is None and row["rel_cost_per_token@30"] is None
    assert (row["tok_s_per_gpu@60"], row["kind@60"]) == (None, "unreached")


def test_ramp_against_a_saturating_engine_writes_points_and_stops(tmp_path):
    async def scenario() -> None:
        trajectories = [
            replay.parse_trajectory(
                synthetic_trajectory(f"s{i}", calls=40, completion_tokens=40), f"s{i}"
            )
            for i in range(6)
        ]
        plan = sweep.SweepConfig(
            sessions_per_engine=(2, 4, 8, 16),
            warmup_s=0.3,
            max_warmup_s=3.0,
            window_s=2.0,
            # Off, so the ramp stops on speed alone, deterministically.
            saturation_gain=0.0,
            **RATE_PLAN,
        )
        bench = configs.CONFIGS["b200-bf16"]
        out = tmp_path / "b200-bf16" / "base.csv"
        seen = []
        # Four requests decode at 100 tok/s; past four, they share 400 tok/s.
        async with FakeEngine(token_rate=100, capacity=4) as engine:
            rows = await sweep.run_sweep(
                trajectories,
                [replay.Target(engine.url)],
                labels=sweep.variant_labels(bench, bench.variants[0]),
                sweep=plan,
                replay_config=replay.ReplayConfig(seed=3, metrics_interval_s=0.5),
                csv_path=out,
                on_point=seen.append,
            )

        brief = [
            {k: r[k] for k in ("sessions", "decode_tok_s_p50", "stop_reason")}
            for r in rows
        ]
        assert [row["sessions"] for row in rows] == [2, 4, 8, 16], brief
        assert seen == rows
        assert rows[-1]["stop_reason"] == "below_slowest_target"
        assert [row["stop_reason"] for row in rows[:-1]] == [None, None, None]
        fast, full, shared, crowded = (row["decode_tok_s_p50"] for row in rows)
        assert fast > 75 and full > 75 and 35 < shared < 65 and crowded < 30
        assert rows[1]["output_tok_s_per_gpu"] > 1.6 * rows[0]["output_tok_s_per_gpu"]
        # Past capacity, more sessions add no output.
        assert rows[3]["output_tok_s_per_gpu"] < 1.2 * rows[1]["output_tok_s_per_gpu"]
        written = sweep.read_csv([out])
        assert [int(row["sessions"]) for row in written] == [2, 4, 8, 16]
        assert {row["config"] for row in written} == {"b200-bf16"}
        value, kind = sweep.throughput_at_speed(written, 60, **RATE)
        assert kind == "crossed" and value == pytest.approx(
            rows[1]["output_tok_s_per_gpu"], rel=0.25
        )

    asyncio.run(scenario())


def test_summarize_cli_prints_and_writes(tmp_path, capsys):
    path = tmp_path / "points.csv"
    for row in _rows(
        "b200-bf16", "base", "r1", [_point(8, 90, 1000), _point(16, 50, 1500)]
    ):
        sweep.append_csv(path, row)

    sweep.main(
        [
            "summarize",
            str(path),
            "--speed-metric",
            "decode_tok_s_p50",
            "--targets",
            "60,30",
            "--out",
            str(tmp_path / "summary.csv"),
        ]
    )

    printed = json.loads(capsys.readouterr().out.splitlines()[0])
    assert printed["config"] == "b200-bf16" and printed["rel_throughput@60"] == 1.0
    assert (tmp_path / "summary.csv").read_text().startswith("config,")


def test_a_point_reports_the_engines_tokenizer_cpu_per_finished_request():
    bench = configs.CONFIGS["b200-bf16"]
    summary = {
        "sessions": 16,
        "output_tok_s": 4000.0,
        "requests_s": 20.0,
        "server": {
            "rate:sglang:process_cpu_seconds_total": 0.5,
            "rate:sglang:e2e_request_latency_seconds_count": 20.0,
        },
    }

    row = sweep.point_row(
        sweep.variant_labels(bench, bench.variants[0]),
        summary,
        sessions_per_engine=4,
        engines=4,
        warmup_s=1.0,
    )

    assert row["server_cpu_s_per_request"] == pytest.approx(0.025)


def test_a_point_that_loses_its_engine_is_remeasured_on_the_replacement(tmp_path):
    async def scenario() -> None:
        trajectories = [
            replay.parse_trajectory(
                synthetic_trajectory(f"p{i}", calls=20, completion_tokens=20), f"p{i}"
            )
            for i in range(4)
        ]
        plan = sweep.SweepConfig(
            sessions_per_engine=(2,), warmup_s=0.2, max_warmup_s=1.0, window_s=1.0
        )
        bench = configs.CONFIGS["b200-bf16"]
        refreshed = []
        async with (
            FakeEngine(token_rate=400, reject=lambda body: 503) as preempted,
            FakeEngine(token_rate=400) as replacement,
        ):

            async def refresh():
                refreshed.append(True)
                return [replay.Target(replacement.url)]

            rows = await sweep.run_sweep(
                trajectories,
                [replay.Target(preempted.url)],
                labels=sweep.variant_labels(bench, bench.variants[0]),
                sweep=plan,
                replay_config=replay.ReplayConfig(
                    seed=4, metrics_path=None, error_backoff_s=0.01, stream=False
                ),
                refresh_targets=refresh,
            )

        (row,) = rows
        assert refreshed == [True]
        assert row["engine_retries"] == 1
        assert row["errors"] == 0 and row["requests"] > 0
        assert replacement.requests

    asyncio.run(scenario())


def test_a_point_that_lost_requests_never_prices_a_variant():
    good = _turns(8, 4.0, 900)
    lost = _turns(16, 6.0, 1200, stop_reason="errors")

    assert sweep.throughput_at_speed([good, lost], 10.0) == (900, "lower_bound")


def test_stray_fragments_in_a_points_csv_are_skipped(tmp_path):
    path = tmp_path / "base.csv"
    sweep.append_csv(
        path, {"config": "b200-bf16", "sessions_per_engine": 8, "variant": "base"}
    )
    sweep.append_csv(
        path, {"config": "b200-bf16", "sessions_per_engine": 16, "variant": "base"}
    )
    # What the volume left behind: the tail of an earlier row's metrics on its own line,
    # and a row whose fields shifted, so a metric fragment lands in stop_reason.
    width = len(sweep.POINT_COLUMNS)
    with path.open("a", newline="") as handle:
        handle.write('_sum"": 7283.4}",\r\n')
        handle.write(
            "r1,,b200-bf16,,base,"
            + "," * (width - 6)
            + '" ""mean:sglang:x"": 0.0}"\r\n'
        )

    rows = sweep.read_csv([path])

    assert [row["sessions_per_engine"] for row in rows] == ["8", "16"]
    assert not list(tmp_path.glob(".*.tmp"))
