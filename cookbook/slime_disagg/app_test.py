import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "recipe",
    [
        "qwen3_4b_delta_flash",
        "qwen3_4b_delta_flash_hillclimb",
        "moonlight",
        "moonlight_int4",
    ],
)
def test_hook_settings_travel_as_custom_config(recipe):
    cfg = importlib.import_module(f"cookbook.slime_disagg.configs.{recipe}").slime

    assert not any(arg.startswith("--rollout-request-") for arg in cfg.cli_args())
    assert cfg.custom_config_path["rollout_request_retry_attempts"] == 240


@pytest.mark.parametrize(
    "custom",
    [
        None,
        {
            "rollout_request_retry_attempts": 17,
            "run_id": "stale",
            "rollout_modal_flash_app_name": "stale",
        },
    ],
)
def test_trainer_preserves_hook_settings_and_owns_run_identity(monkeypatch, custom):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_4b_delta_flash")
    monkeypatch.setenv("RUN_ID", "current")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.slime_disagg.app", raising=False)
    app = importlib.import_module("cookbook.slime_disagg.app")
    monkeypatch.setattr(app, "train_volumes", {})
    monkeypatch.setattr(
        app,
        "ModalFlashLBPool",
        lambda *args: SimpleNamespace(gateway_url=lambda: "http://pool"),
    )
    resolve = Mock()
    monkeypatch.setattr(app.launch, "resolve_config", resolve)
    monkeypatch.setattr(app.subprocess, "run", Mock())
    from cookbook.common import hooks

    claim = Mock()
    monkeypatch.setattr(hooks, "claim_pool", claim)
    cfg = app.SlimeConfig.from_payload(app.slime_cfg.to_payload())
    cfg.custom_config_path = custom
    trainer = app.Trainer._get_user_cls()()
    trainer.rank = 0

    trainer.train(cfg.to_payload())

    resolved = resolve.call_args.args[0]
    settings = resolved.custom_config_path
    assert settings["run_id"] == "current"
    assert settings["rollout_modal_flash_app_name"] == app.APP_NAME
    assert settings.get("rollout_request_retry_attempts") == (17 if custom else None)
    assert vars(claim.call_args.args[0]) == {
        "update_weight_disk_dir": str(app.UPDATES_DIR),
        **settings,
    }
