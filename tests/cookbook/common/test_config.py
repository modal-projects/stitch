from types import SimpleNamespace

import pytest

from cookbook.common.config import (
    GPUType,
    ModalConfig,
    RolloutPoolConfig,
    validate_serving_config,
)
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


def test_single_rollout_pool_preserves_existing_config() -> None:
    config = ModalConfig(
        gpu="B300",
        rollout_gpu="H200",
        rollout_min_containers=2,
        rollout_max_containers=5,
    )

    args = {"--tp": "2"}
    assert config.resolved_rollout_pools(
        default_gpus_per_engine=2,
        default_target_inputs=7,
        default_sglang_args=args,
    ) == (
        RolloutPoolConfig(
            name="Server",
            gpu="H200",
            gpus_per_engine=2,
            target_inputs=7,
            sglang_args=args,
            min_containers=2,
            max_containers=5,
        ),
    )
    assert config.rollout_replica_floor == 2


def test_heterogeneous_rollout_floor_is_sum_of_independent_pools() -> None:
    config = ModalConfig(
        rollout_pools=(
            RolloutPoolConfig(
                name="ServerH100",
                gpu="H100",
                gpus_per_engine=2,
                target_inputs=8,
                sglang_args={"--tp": "2", "--max-running-requests": "16"},
                max_containers=2,
            ),
            RolloutPoolConfig(
                name="ServerB300",
                gpu="B300",
                gpus_per_engine=1,
                target_inputs=8,
                sglang_args={"--tp": "1", "--max-running-requests": "16"},
                max_containers=2,
            ),
        )
    )

    assert config.rollout_replica_floor == 2


def test_heterogeneous_pools_reject_fallback_gpu_list() -> None:
    config = ModalConfig(
        rollout_gpu=["H100", "B300"],
        rollout_pools=(
            RolloutPoolConfig(
                name="ServerH100",
                gpu="H100",
                gpus_per_engine=2,
                target_inputs=8,
                sglang_args={"--tp": "2"},
            ),
        ),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        config.resolved_rollout_pools(
            default_gpus_per_engine=1,
            default_target_inputs=1,
            default_sglang_args={},
        )


def test_explicit_rollout_pool_requires_one_exact_gpu_type() -> None:
    config = ModalConfig(
        rollout_pools=(
            RolloutPoolConfig(
                name="Server",
                gpu=["H100", "B300"],
                gpus_per_engine=1,
                target_inputs=8,
                sglang_args={"--tp": "1"},
            ),
        )
    )

    with pytest.raises(ValueError, match="one exact GPU type"):
        config.resolved_rollout_pools(
            default_gpus_per_engine=1,
            default_target_inputs=1,
            default_sglang_args={},
        )


def test_same_gpu_can_have_independent_engine_configurations() -> None:
    pools = (
        RolloutPoolConfig(
            name="ServerB200TP1",
            gpu="B200",
            gpus_per_engine=1,
            target_inputs=8,
            sglang_args={"--tp": "1", "--max-running-requests": "16"},
        ),
        RolloutPoolConfig(
            name="ServerB200TP2",
            gpu="B200",
            gpus_per_engine=2,
            target_inputs=16,
            sglang_args={"--tp": "2", "--max-running-requests": "32"},
        ),
    )
    config = ModalConfig(rollout_pools=pools)

    assert (
        config.resolved_rollout_pools(
            default_gpus_per_engine=8,
            default_target_inputs=1,
            default_sglang_args={},
        )
        == pools
    )
