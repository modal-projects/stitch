import importlib
import sys
from types import SimpleNamespace

import modal
import pytest

from cookbook.common import serving_image
from cookbook.common.constants import KERNEL_CACHE_PATH
from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.configs import qwen3_4b_math


@pytest.mark.parametrize("entrypoint", ["app", "prep_app"])
@pytest.mark.parametrize(
    "recipe",
    [
        "qwen3_4b_math",
        "qwen3_6_35b_a3b_nvfp4",
        "glm5_3_nvfp4",
        "qwen3_8_27b_harbor_smoke",
        "qwen3_8_27b_harbor_mix_v0",
        "qwen3_8_27b_harbor_mix_v1",
        "qwen3_8_27b_harbor_mix_v1b",
        "qwen3_8_27b_harbor_mix_v2",
        "qwen3_8_27b_harbor_mix_v2b",
        "qwen3_8_27b_harbor_mix_v2b_h200",
    ],
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


def test_trainer_mounts_kernel_cache_volume(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_nvfp4")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)

    app = importlib.import_module("cookbook.miles_disagg.app")

    assert app.train_volumes[str(KERNEL_CACHE_PATH)].name == "kernel-cache"


def test_trainer_mounts_recipe_extra_volumes(monkeypatch):
    recipe = SimpleNamespace(**vars(qwen3_4b_math))
    extra = modal.Volume.from_name("recipe-data")
    recipe.TRAINER_EXTRA_VOLUMES = {"/recipe-data": extra}
    monkeypatch.setitem(
        sys.modules, "cookbook.miles_disagg.configs.extra_volume_test_recipe", recipe
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", "extra_volume_test_recipe")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)

    app = importlib.import_module("cookbook.miles_disagg.app")

    assert app.train_volumes["/recipe-data"] is extra
    assert app.train_volumes[str(KERNEL_CACHE_PATH)].name == "kernel-cache"


def test_trainer_gets_recipe_extra_secrets(monkeypatch):
    recipe = SimpleNamespace(**vars(qwen3_4b_math))
    extra = modal.Secret.from_name("recipe-settings")
    recipe.TRAINER_EXTRA_SECRETS = [extra]
    monkeypatch.setitem(
        sys.modules, "cookbook.miles_disagg.configs.extra_secret_test_recipe", recipe
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", "extra_secret_test_recipe")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)

    app = importlib.import_module("cookbook.miles_disagg.app")

    assert app.train_secrets[-1] is extra
    assert app.train_secrets[: len(app.STORE_SECRETS)] == app.STORE_SECRETS


@pytest.mark.parametrize("data_env", [None, "data-env"])
def test_data_env_reaches_both_images(monkeypatch, data_env):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_4b_math")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    if data_env:
        monkeypatch.setenv("STITCH_DATA_ENV", data_env)
    else:
        monkeypatch.delenv("STITCH_DATA_ENV", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)
    trainer_env, server_env = {}, {}
    build_trainer, build_server = (
        trainer_image.build_trainer_image,
        serving_image.build_serving_image,
    )

    def trainer(**kwargs):
        trainer_env.update(kwargs["extra_env"])
        return build_trainer(**kwargs)

    def server(**kwargs):
        server_env.update(kwargs["extra_env"])
        return build_server(**kwargs)

    monkeypatch.setattr(trainer_image, "build_trainer_image", trainer)
    monkeypatch.setattr(serving_image, "build_serving_image", server)

    app = importlib.import_module("cookbook.miles_disagg.app")

    assert app.DATA_ENV == data_env
    expected = {"STITCH_DATA_ENV": data_env} if data_env else {}
    for env in (trainer_env, server_env):
        assert {k: v for k, v in env.items() if k == "STITCH_DATA_ENV"} == expected


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
