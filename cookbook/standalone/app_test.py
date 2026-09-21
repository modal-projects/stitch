from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import modal
import pytest

from cookbook.common import serving_image
from cookbook.standalone.configs import glm5_3_fp8


def _standalone_env(monkeypatch) -> None:
    monkeypatch.setenv("EXPERIMENT_CONFIG", "glm5_3_fp8")
    monkeypatch.setenv("RUN_ID", "run-42")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    sys.modules.pop("cookbook.standalone.app", None)


def test_engine_only_app_has_no_trainer(monkeypatch) -> None:
    _standalone_env(monkeypatch)

    app = importlib.import_module("cookbook.standalone.app")

    assert app.APP_NAME == "stitch-standalone-glm5-3-fp8-run-42"
    assert hasattr(app, "Server")
    assert not hasattr(app, "Trainer")


def test_rl_serving_contract(monkeypatch) -> None:
    _standalone_env(monkeypatch)

    app = importlib.import_module("cookbook.standalone.app")

    assert app.SGLANG_SERVER_ARGS["--enable-return-routed-experts"] == ""
    assert app.SGLANG_SERVER_ARGS["--sampling-mask-max-tokens"] == "8192"
    assert app.SGLANG_SERVER_ARGS["--moe-runner-backend"] == "flashinfer_trtllm_routed"
    assert "--speculative-algorithm" not in app.SGLANG_SERVER_ARGS
    assert app.draft_volume is None
    assert app.exp.SGLANG_DELTA_UPDATE_MODE == "cpu"
    assert "--weight-update-staging" not in app.SGLANG_SERVER_ARGS
    assert app.exp.SIDECAR_COMMIT_MODE == "in_place"
    assert not app.exp.SIDECAR_FLUSH_CACHE_ON_COMMIT


def test_replica_topology_and_admission(monkeypatch) -> None:
    _standalone_env(monkeypatch)

    app = importlib.import_module("cookbook.standalone.app")

    assert app.modal_cfg.rollout_gpus(app.exp.ROLLOUT_GPUS_PER_ENGINE) == "B300:8"
    assert app.SGLANG_SERVER_ARGS["--tp"] == "8"
    assert app.ROLLOUT_CONCURRENCY == 16
    assert app.SGLANG_SERVER_ARGS["--max-running-requests"] == "32"
    assert app.SGLANG_SERVER_ARGS["--cuda-graph-max-bs-decode"] == "32"


def test_preparation_and_serving_share_the_pinned_checkpoint(monkeypatch) -> None:
    _standalone_env(monkeypatch)
    sys.modules.pop("cookbook.standalone.prep_app", None)

    app = importlib.import_module("cookbook.standalone.app")
    prep = importlib.import_module("cookbook.standalone.prep_app")

    assert prep.exp.SOURCE_MODEL == "zai-org/GLM-5.3"
    assert prep.exp.SOURCE_REVISION == "aca966e4e02791568aa6a4ced368624b3d897f42"
    assert prep.exp.BASE_CHECKPOINT_PATH == app.exp.BASE_CHECKPOINT_PATH
    assert app.SGLANG_SERVER_ARGS["--served-model-name"] == prep.exp.SOURCE_MODEL


@pytest.mark.parametrize("entrypoint", ["app", "prep_app"])
def test_invalid_recipe_fails_before_image_construction(monkeypatch, entrypoint):
    recipe = SimpleNamespace(**vars(glm5_3_fp8))
    recipe.SOURCE_REVISION = "main"
    monkeypatch.setitem(
        sys.modules, "cookbook.standalone.configs.invalid_test_recipe", recipe
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", "invalid_test_recipe")
    monkeypatch.setenv("RUN_ID", "test-run")
    module = f"cookbook.standalone.{entrypoint}"
    monkeypatch.delitem(sys.modules, module, raising=False)

    def build_image(**_kwargs):
        pytest.fail("invalid recipe reached image construction")

    monkeypatch.setattr(serving_image, "build_serving_image", build_image)
    monkeypatch.setattr(modal.Image, "debian_slim", build_image)
    with pytest.raises(ValueError, match="SOURCE_REVISION"):
        importlib.import_module(module)
