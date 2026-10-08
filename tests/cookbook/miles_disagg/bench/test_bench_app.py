"""The bench Modal app's definition and the launcher's plan; nothing here talks to Modal."""

import importlib
import sys
from pathlib import Path

import pytest

from cookbook.miles_disagg import evaluation
from cookbook.miles_disagg.bench import configs, launch, sweep

BENCH_ENV = (
    "BENCH_CONFIG",
    "BENCH_VARIANTS",
    "BENCH_ENGINES",
    "BENCH_EXPERIMENT",
    "BENCH_RUN",
    "BENCH_VERSION",
    "BENCH_TAG",
)


def _import_app(monkeypatch, **env):
    for key in BENCH_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.bench.app", raising=False)
    return importlib.import_module("cookbook.miles_disagg.bench.app")


@pytest.mark.parametrize("key", list(configs.CONFIGS))
def test_app_defines_one_fixed_engine_fleet_per_variant(monkeypatch, key):
    bench = configs.CONFIGS[key]

    app = _import_app(monkeypatch, BENCH_CONFIG=key, BENCH_ENGINES="2")

    assert app.APP_NAME == f"stitch-bench-{key}"
    assert list(app.POOLS) == [variant.name for variant in bench.variants]
    for variant in bench.variants:
        pool = app.POOLS[variant.name]
        assert (pool.min_containers, pool.max_containers) == (2, 2)
        assert pool.gpu_request() == f"{bench.gpu}:{bench.gpus_per_engine}"
        assert app.ENGINE_SERVERS[variant.name] is getattr(app, pool.name)
        args = app.engine_args(pool)
        # What serve_startup adds for a rollout replica, booting the base as version 0.
        assert args["--weight-update-staging"] == app.exp.SGLANG_DELTA_UPDATE_MODE
        assert args["--weight-version"] == "0"
        assert args["--served-model-name"] == app.exp.miles.hf_checkpoint
        assert args["--max-running-requests"] == str(variant.max_batch)
        assert {k: args[k] for k in bench.server_args(variant)} == bench.server_args(
            variant
        )
    assert app.POINT.is_base and app.POINT.view == bench.weight_view
    assert evaluation.checkpoint_path(
        app.exp, app.POINT, source_run_root=evaluation.SOURCE_RUN_PATH
    ) == Path(app.exp.ROLLOUT_WEIGHT_VIEWS[bench.weight_view])
    assert str(evaluation.SOURCE_RUN_PATH) not in app.engine_volumes
    assert app.BENCH_ENVIRONMENT["BENCH_CONFIG"] == key
    assert app.BENCH_ENVIRONMENT["BENCH_VARIANTS"] == ",".join(app.POOLS)


def test_app_serves_a_runs_export_from_its_volume(monkeypatch):
    app = _import_app(
        monkeypatch,
        BENCH_CONFIG="h200-fp8",
        BENCH_VARIANTS="kv-bf16,base",
        BENCH_EXPERIMENT="qwen3_6_35b_a3b_hetero_score_centering_mis",
        BENCH_RUN="r03",
        BENCH_VERSION="100",
        BENCH_TAG="v100",
    )

    assert app.APP_NAME == "stitch-bench-h200-fp8-v100"
    assert list(app.POOLS) == ["kv-bf16", "base"]
    # The export saved at rollout 99 serves published version 100.
    assert evaluation.checkpoint_path(
        app.exp, app.POINT, source_run_root=evaluation.SOURCE_RUN_PATH
    ) == Path("/source-run/r03/hf_checkpoints/weight_v000099/fp8")
    assert str(evaluation.SOURCE_RUN_PATH) in app.engine_volumes
    assert app.engine_args(app.POOLS["base"])["--weight-version"] == "100"


def test_plan_bounds_each_variants_gpu_time_and_cost():
    bench = configs.CONFIGS["h100-bf16"]

    plan = launch.plan(bench, bench.variants, 2, sweep.SweepConfig())

    assert plan["app"] == "stitch-bench-h100-bf16"
    by_name = {row["variant"]: row for row in plan["variants"]}
    base = by_name["base"]
    assert base["sessions_per_engine"] == [8, 16, 20, 24, 28, 32, 48, 64]
    hours = (launch.BOOT_MINUTES * 60 + 8 * (120 + 180)) / 3600
    assert base["max_hours"] == round(hours, 2)
    # Two engines of two H100s each.
    assert base["max_gpu_hours"] == round(hours * 4, 2)
    assert base["max_usd"] == round(hours * 4 * 3.95, 2)
    assert by_name["cg-96"]["sessions_per_engine"] == [
        8,
        16,
        20,
        24,
        28,
        32,
        48,
        64,
        96,
    ]
    assert plan["max_usd"] == round(sum(row["max_usd"] for row in plan["variants"]), 2)


def test_launch_refuses_gvisor(monkeypatch):
    monkeypatch.delenv("MODAL_FUNCTION_RUNTIME", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["launch", "--config", "b200-bf16", "--traces", "bench-traces/x"]
    )

    with pytest.raises(SystemExit, match="MODAL_FUNCTION_RUNTIME=runc"):
        launch.main()


def test_launch_plan_prints_without_modal(monkeypatch, capsys):
    monkeypatch.delenv("MODAL_FUNCTION_RUNTIME", raising=False)
    monkeypatch.setattr(
        sys, "argv", ["launch", "--config", "b200-bf16", "--variants", "base", "--plan"]
    )

    launch.main()

    assert '"server": "BenchB200Bf16Base"' in capsys.readouterr().out
