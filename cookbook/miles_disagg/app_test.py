from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import modal
import pytest

from cookbook.miles_disagg.config import MilesConfig


@pytest.fixture
def server_options(monkeypatch):
    observed = {}
    decorate = modal.App.server

    def capture(self, *args, **kwargs):
        observed.update(kwargs)
        return decorate(self, *args, **kwargs)

    monkeypatch.setattr(modal.App, "server", capture)
    return observed


@pytest.fixture
def app_module(monkeypatch, server_options):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "glm5_3_nvfp4")
    monkeypatch.setenv("RUN_ID", "run-42")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    name = "cookbook.miles_disagg.app"
    previous = sys.modules.pop(name, None)
    try:
        yield importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous


def test_glm53_limits_admission_at_modal_router(app_module, server_options):
    assert server_options["experimental_options"] == {
        "kv_aware_routing": True,
        "max_concurrency": 24,
    }
    assert server_options["target_concurrency"] == 16
    assert app_module.SGLANG_SERVER_ARGS["--max-running-requests"] == "24"
    assert "--max-queued-requests" not in app_module.SGLANG_SERVER_ARGS


@pytest.mark.parametrize("skip", [False, True])
def test_spawn_ships_readiness_bypass_outside_miles_arguments(
    monkeypatch, app_module, skip
) -> None:
    observed = {}
    call = SimpleNamespace(object_id="fc-trainer")

    def spawn(payload):
        observed["payload"] = payload
        return call

    def from_name(_cls, app_name, class_name):
        observed["target"] = (app_name, class_name)
        return lambda: SimpleNamespace(train=SimpleNamespace(spawn=spawn))

    def record(volume, run_id, call_id):
        observed["recorded"] = (volume, run_id, call_id)

    monkeypatch.setattr(app_module.modal.Cls, "from_name", classmethod(from_name))
    monkeypatch.setattr(app_module, "record_trainer_call", record)

    assert app_module.spawn_train(skip_rollout_ready_check=skip) is call

    payload = observed["payload"]
    assert payload["skip_rollout_ready_check"] is skip
    assert "skip_rollout_ready_check" not in payload["fields"]
    assert "--skip-rollout-ready-check" not in MilesConfig.from_payload(payload).cli_args()
    assert observed["target"] == (app_module.APP_NAME, "Trainer")
    assert observed["recorded"] == (app_module.run_volume, "run-42", call.object_id)
