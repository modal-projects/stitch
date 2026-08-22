from __future__ import annotations

import importlib
import sys


def _standalone_env(monkeypatch, experiment: str = "glm5_2_fp8") -> None:
    monkeypatch.setenv("EXPERIMENT_CONFIG", experiment)
    monkeypatch.setenv("RUN_ID", "run-42")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    sys.modules.pop("cookbook.standalone.app", None)


def test_engine_only_app_has_no_trainer(monkeypatch) -> None:
    _standalone_env(monkeypatch)

    app = importlib.import_module("cookbook.standalone.app")

    assert app.APP_NAME == "stitch-standalone-glm5-2-fp8-run-42"
    assert hasattr(app, "Server")
    assert hasattr(app, "RouterRegistry")
    assert hasattr(app, "Router")
    assert not hasattr(app, "Trainer")


def test_rl_serving_contract(monkeypatch) -> None:
    _standalone_env(monkeypatch)

    app = importlib.import_module("cookbook.standalone.app")

    # Routing replay and the sampling mask are the pool's RL contract: the
    # trainer replays routed experts and consumes the realized top-p mask.
    assert app.SGLANG_SERVER_ARGS["--enable-return-routed-experts"] == ""
    assert app.SGLANG_SERVER_ARGS["--sampling-mask-max-tokens"] == "8192"
    assert app.SGLANG_SERVER_ARGS["--moe-runner-backend"] == "flashinfer_trtllm_routed"
    # The DFlash draft is fixed and lives outside the target checkpoint.
    assert app.SGLANG_SERVER_ARGS["--speculative-algorithm"] == "DFLASH"
    assert app.draft_volume is not None
    # CPU staging must be declared exactly when the cpu update mode is selected.
    assert app.exp.SGLANG_DELTA_UPDATE_MODE == "cpu"
    assert "--enable-cpu-weight-cache" in app.SGLANG_SERVER_ARGS


def test_tp4_variant_keeps_the_rl_contract(monkeypatch) -> None:
    _standalone_env(monkeypatch, experiment="glm5_2_fp8_tp4")

    app = importlib.import_module("cookbook.standalone.app")

    assert app.APP_NAME == "stitch-standalone-glm5-2-fp8-tp4-run-42"
    assert app.exp.ROLLOUT_GPUS_PER_ENGINE == 4
    # Engine admission stops at the benchmarked 16-user KV ceiling; the
    # router targets the comfortable 12.
    assert app.SGLANG_SERVER_ARGS["--max-running-requests"] == "16"
    assert app.ROLLOUT_CONCURRENCY == 12
    assert app.SGLANG_SERVER_ARGS["--speculative-dflash-block-size"] == "8"
    assert app.SGLANG_SERVER_ARGS["--cuda-graph-max-bs"] == "16"
    assert app.SGLANG_SERVER_ARGS["--enable-return-routed-experts"] == ""
    assert app.SGLANG_SERVER_ARGS["--sampling-mask-max-tokens"] == "8192"
    assert app.SGLANG_SERVER_ARGS["--moe-runner-backend"] == "flashinfer_trtllm_routed"
    assert app.draft_volume is not None
