"""Import smoke tests for standalone experiment configs."""

from __future__ import annotations

import importlib
import sys

import pytest

CONFIG_MODULES = ("glm5_2_fp8", "glm5_2_fp8_tp4", "qwen3_0p6b")

# Attributes cookbook.standalone.app reads directly off the experiment module.
REQUIRED_ATTRS = (
    "modal",
    "APP_NAME",
    "EXPERIMENT_VOLUME_NAME",
    "CHECKPOINT_VOLUME_NAME",
    "LOCAL_CHECKPOINT_PATH",
    "SOURCE_MODEL",
    "SOURCE_REVISION",
    "BASE_CHECKPOINT_PATH",
    "ROLLOUT_GPUS_PER_ENGINE",
    "SIDECAR_COMMIT_MODE",
    "SIDECAR_FLUSH_CACHE_ON_COMMIT",
    "SGLANG_DELTA_UPDATE_MODE",
    "SGLANG_SERVER_ARGS",
)


@pytest.mark.parametrize("name", CONFIG_MODULES)
def test_config_imports_with_required_attrs(name: str) -> None:
    exp = importlib.import_module(f"cookbook.standalone.configs.{name}")

    for attr in REQUIRED_ATTRS:
        assert hasattr(exp, attr), f"{name} is missing {attr}"
    # app.py refuses to boot without the consolidation/autoscaler capacity unit.
    assert exp.modal.rollout_target_inputs is not None


def test_qwen3_0p6b_poc_shape() -> None:
    exp = importlib.import_module("cookbook.standalone.configs.qwen3_0p6b")

    # Cheap end-to-end pool: one small GPU per engine, three pinned replicas.
    assert exp.modal.gpu == "L40S"
    assert exp.ROLLOUT_GPUS_PER_ENGINE == 1
    assert exp.modal.rollout_min_containers == 3
    assert exp.modal.rollout_max_containers == 3
    # Offline-evals wiring opted in; the registry owns reconciliation timing.
    assert exp.OFFLINE_EVALS is True
    assert exp.SIDECAR_RECONCILE_INTERVAL == 365 * 24 * 3600.0
    assert exp.SGLANG_DELTA_UPDATE_MODE == "cpu"
    assert "--enable-cpu-weight-cache" in exp.SGLANG_SERVER_ARGS


def test_qwen3_0p6b_app_wiring(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_0p6b")
    monkeypatch.setenv("RUN_ID", "run-42")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    sys.modules.pop("cookbook.standalone.app", None)

    app = importlib.import_module("cookbook.standalone.app")

    assert app.APP_NAME == "stitch-standalone-qwen3-0p6b-run-42"
    # The config's rollout_target_inputs is the router's consolidation capacity.
    assert app.ROLLOUT_CONCURRENCY == 8
    assert app.SGLANG_SERVER_ARGS["--max-running-requests"] == "8"
    # OFFLINE_EVALS is set: the offline-evals wiring is active.
    assert hasattr(app, "rollout_marks")
