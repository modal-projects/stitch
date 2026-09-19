from types import SimpleNamespace

import pytest

from cookbook.common.config import GPUType, ModalConfig, validate_serving_config
from cookbook.standalone.configs import glm5_3_fp8


def test_rejects_unknown_infrastructure_setting() -> None:
    with pytest.raises(TypeError, match="rollout_min_container"):
        ModalConfig(rollout_min_container=1)


def test_default_rollout_gpu_request() -> None:
    assert ModalConfig().rollout_gpus(4) == "B200:4"


@pytest.mark.parametrize(
    ("rollout_gpu", "expected"),
    [
        (None, "H100:8"),
        ("H200", "H200:8"),
        (["B200", "B300"], ["B200:8", "B300:8"]),
    ],
)
def test_rollout_gpu_selection(
    rollout_gpu: GPUType | list[GPUType] | None,
    expected: str | list[str],
) -> None:
    config = ModalConfig(gpu="H100", rollout_gpu=rollout_gpu)

    assert config.rollout_gpus(8) == expected


@pytest.fixture
def serving_recipe():
    recipe = SimpleNamespace(**vars(glm5_3_fp8))
    recipe.SGLANG_SERVER_ARGS = dict(recipe.SGLANG_SERVER_ARGS)
    return recipe


@pytest.mark.parametrize("revision", [None, "", "main", "aca966e4"])
def test_serving_requires_an_immutable_source_revision(serving_recipe, revision):
    serving_recipe.SOURCE_REVISION = revision

    with pytest.raises(ValueError, match="SOURCE_REVISION"):
        validate_serving_config(serving_recipe, gpus_per_engine=8)


def test_serving_requires_a_source_model(serving_recipe):
    del serving_recipe.SOURCE_MODEL

    with pytest.raises(ValueError, match="SOURCE_MODEL"):
        validate_serving_config(serving_recipe, gpus_per_engine=8)


@pytest.mark.parametrize("gpus_per_engine", [0, 4])
def test_serving_tensor_parallelism_must_match_gpu_request(
    serving_recipe, gpus_per_engine
):
    with pytest.raises(ValueError, match="--tp.*GPU count"):
        validate_serving_config(serving_recipe, gpus_per_engine=gpus_per_engine)


def test_cpu_updates_require_the_server_weight_cache(serving_recipe):
    del serving_recipe.SGLANG_SERVER_ARGS["--enable-cpu-weight-cache"]

    with pytest.raises(ValueError, match="--enable-cpu-weight-cache"):
        validate_serving_config(serving_recipe, gpus_per_engine=8)


def test_disk_updates_require_the_server_weight_cache_to_be_disabled(serving_recipe):
    serving_recipe.SGLANG_DELTA_UPDATE_MODE = "disk"

    with pytest.raises(ValueError, match="--enable-cpu-weight-cache"):
        validate_serving_config(serving_recipe, gpus_per_engine=8)

    del serving_recipe.SGLANG_SERVER_ARGS["--enable-cpu-weight-cache"]
    validate_serving_config(serving_recipe, gpus_per_engine=8)
