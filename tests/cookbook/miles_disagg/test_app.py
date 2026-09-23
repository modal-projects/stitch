import importlib
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    KERNEL_CACHE_PATH,
    STITCH_PATH,
)
from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.configs import qwen3_4b_math


@pytest.mark.parametrize("entrypoint", ["app", "prep_app"])
@pytest.mark.parametrize(
    "recipe", ["qwen3_4b_math", "qwen3_6_35b_a3b_nvfp4", "glm5_3_nvfp4"]
)
def test_maintained_recipes_import_through_deployment_entrypoints(
    monkeypatch, entrypoint, recipe
):
    monkeypatch.setenv("EXPERIMENT_CONFIG", recipe)
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    module = f"cookbook.miles_disagg.{entrypoint}"
    monkeypatch.delitem(sys.modules, module, raising=False)

    app = importlib.import_module(module)

    assert app.exp.__name__ == f"cookbook.miles_disagg.configs.{recipe}"
    assert app.modal_cfg is app.exp.modal
    assert app.miles_cfg is app.exp.miles


def _import_app(monkeypatch, recipe_name: str):
    monkeypatch.setenv("EXPERIMENT_CONFIG", recipe_name)
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)
    return importlib.import_module("cookbook.miles_disagg.app")


def test_trainer_mounts_kernel_cache_volume_by_default(monkeypatch):
    app = _import_app(monkeypatch, "qwen3_6_35b_a3b_nvfp4")

    volume = app.train_volumes[str(KERNEL_CACHE_PATH)]
    assert volume is app.kernel_cache_volume
    assert volume.name == "kernel-cache"


def test_recipe_can_disable_kernel_cache_volume(monkeypatch):
    recipe = SimpleNamespace(**vars(qwen3_4b_math))
    recipe.modal = replace(recipe.modal, kernel_cache_volume=None)
    monkeypatch.setitem(
        sys.modules, "cookbook.miles_disagg.configs.no_kernel_cache", recipe
    )

    app = _import_app(monkeypatch, "no_kernel_cache")

    assert app.kernel_cache_volume is None
    assert str(KERNEL_CACHE_PATH) not in app.train_volumes
    assert set(app.train_volumes) == {
        str(HF_CACHE_PATH),
        str(CHECKPOINTS_PATH),
        str(DATA_PATH),
        str(STITCH_PATH),
    }


@pytest.mark.parametrize("entrypoint", ["app", "prep_app"])
def test_invalid_recipe_fails_before_image_construction(monkeypatch, entrypoint):
    recipe = SimpleNamespace(**vars(qwen3_4b_math))
    recipe.SOURCE_REVISION = "main"
    monkeypatch.setitem(
        sys.modules, "cookbook.miles_disagg.configs.invalid_test_recipe", recipe
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", "invalid_test_recipe")
    monkeypatch.setenv("RUN_ID", "test-run")
    module = f"cookbook.miles_disagg.{entrypoint}"
    monkeypatch.delitem(sys.modules, module, raising=False)

    def build_image(**_kwargs):
        pytest.fail("invalid recipe reached image construction")

    monkeypatch.setattr(trainer_image, "build_trainer_image", build_image)
    with pytest.raises(ValueError, match="SOURCE_REVISION"):
        importlib.import_module(module)
