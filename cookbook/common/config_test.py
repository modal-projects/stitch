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


@pytest.mark.parametrize(
    "argument",
    [
        "--weight-update-staging",
        "--weight-update-local-checkpoint-dir",
        "--weight-version",
    ],
)
def test_serving_rejects_stitch_managed_server_arguments(serving_recipe, argument):
    serving_recipe.SGLANG_SERVER_ARGS[argument] = "value"

    with pytest.raises(ValueError, match="managed server arguments"):
        validate_serving_config(serving_recipe, gpus_per_engine=8)


def test_serving_requires_an_explicit_update_mode(serving_recipe):
    del serving_recipe.SGLANG_DELTA_UPDATE_MODE

    with pytest.raises(ValueError, match="SGLANG_DELTA_UPDATE_MODE"):
        validate_serving_config(serving_recipe, gpus_per_engine=8)


@pytest.mark.parametrize("path", ["", 1])
def test_serving_rejects_invalid_local_checkpoint_path(serving_recipe, path):
    serving_recipe.LOCAL_CHECKPOINT_PATH = path

    with pytest.raises(ValueError, match="LOCAL_CHECKPOINT_PATH"):
        validate_serving_config(serving_recipe, gpus_per_engine=8)


def test_disk_updates_require_a_local_checkpoint(serving_recipe):
    serving_recipe.SGLANG_DELTA_UPDATE_MODE = "disk"
    serving_recipe.LOCAL_CHECKPOINT_PATH = None

    with pytest.raises(ValueError, match="disk.*LOCAL_CHECKPOINT_PATH"):
        validate_serving_config(serving_recipe, gpus_per_engine=8)
