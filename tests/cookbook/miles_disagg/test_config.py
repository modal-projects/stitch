import json
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

import pytest

from cookbook.miles_disagg import nvfp4
from cookbook.miles_disagg.config import MilesConfig, validate_recipe
from cookbook.miles_disagg.resume import validate_resumable_config


def _recipe(name):
    recipe = SimpleNamespace(
        **vars(import_module(f"cookbook.miles_disagg.configs.{name}"))
    )
    recipe.miles = deepcopy(recipe.miles)
    for name in (
        "SGLANG_SERVER_ARGS",
        "PREP_ENV",
        "SGLANG_SERVER_ENV",
        "NVFP4_TRAINING_ENV",
        "NVFP4_SERVING_ENV",
    ):
        if hasattr(recipe, name):
            setattr(recipe, name, dict(getattr(recipe, name)))
    return recipe


def _rollout_server_args(recipe):
    if recipe.modal.rollout_pools:
        return [pool.sglang_args for pool in recipe.modal.rollout_pools]
    return [recipe.SGLANG_SERVER_ARGS]


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_4b_math",
        "qwen3_6_35b_a3b_swebench_pro",
        "qwen3_6_35b_a3b_nvfp4",
        "glm5_3_nvfp4",
    ],
)
def test_maintained_training_recipes_have_consistent_checkpoints_and_resume(name):
    recipe = _recipe(name)

    validate_recipe(recipe)
    validate_resumable_config(recipe.miles)


def test_training_and_serving_must_use_the_same_checkpoint():
    recipe = _recipe("qwen3_4b_math")
    recipe.miles.hf_checkpoint = "/checkpoints/another-model"

    with pytest.raises(ValueError, match="hf_checkpoint.*ROLLOUT_CHECKPOINT_PATH"):
        validate_recipe(recipe)


def test_bf16_serving_uses_the_prepared_master_checkpoint():
    recipe = _recipe("qwen3_4b_math")
    recipe.ROLLOUT_CHECKPOINT_PATH = recipe.BF16_CHECKPOINT_PATH / "unprepared-copy"
    recipe.miles.hf_checkpoint = str(recipe.ROLLOUT_CHECKPOINT_PATH)

    with pytest.raises(ValueError, match="BF16 serving"):
        validate_recipe(recipe)


def test_quantized_serving_cannot_overwrite_bf16_masters():
    recipe = _recipe("qwen3_6_35b_a3b_nvfp4")
    recipe.ROLLOUT_CHECKPOINT_PATH = recipe.BF16_CHECKPOINT_PATH
    recipe.miles.hf_checkpoint = str(recipe.ROLLOUT_CHECKPOINT_PATH)

    with pytest.raises(ValueError, match="must differ from BF16_CHECKPOINT_PATH"):
        validate_recipe(recipe)


@pytest.mark.parametrize(
    "name", ["qwen3_6_35b_a3b_swebench_pro", "qwen3_6_35b_a3b_nvfp4"]
)
def test_qwen36_target_only_recipe_omits_mtp_from_training_and_conversion(name):
    recipe = _recipe(name)

    assert recipe.miles.mtp_num_layers == 0
    assert "--mtp-num-layers 0" in recipe.modal.torch_dist_convert_extra_args
    assert all(
        "--speculative-algorithm" not in args for args in _rollout_server_args(recipe)
    )
    assert "mtp." in recipe.miles.hf_export_source_tensor_prefixes


def test_qwen36_bf16_swebench_recipe_sizes_async_rollout_to_the_fleet():
    recipe = _recipe("qwen3_6_35b_a3b_swebench_pro")
    cfg = recipe.miles

    assert recipe.SERVED_CHECKPOINT_FORMAT == "bf16"
    assert recipe.ROLLOUT_CHECKPOINT_PATH == recipe.BF16_CHECKPOINT_PATH
    assert all("--quantization" not in args for args in _rollout_server_args(recipe))
    assert not hasattr(cfg, "fp4_recipe")
    assert cfg.async_mode
    assert cfg.fully_async
    assert cfg.async_max_concurrent_samples == recipe.ROLLOUT_CONCURRENT_SAMPLES
    assert cfg.async_max_concurrent_samples == 640
    assert cfg.sglang_server_concurrency == cfg.async_max_concurrent_samples
    assert len(recipe.modal.rollout_pools) == 4
    assert sum(pool.min_containers for pool in recipe.modal.rollout_pools) == 112
    assert sum(pool.max_containers for pool in recipe.modal.rollout_pools) == 224
    assert (
        sum(
            pool.min_containers * pool.gpus_per_engine
            for pool in recipe.modal.rollout_pools
        )
        == 128
    )
    assert (
        sum(
            pool.max_containers * pool.gpus_per_engine
            for pool in recipe.modal.rollout_pools
        )
        == 256
    )
    assert all(
        pool.min_containers * pool.gpus_per_engine == 32
        for pool in recipe.modal.rollout_pools
    )
    assert all(
        pool.max_containers * pool.gpus_per_engine == 64
        for pool in recipe.modal.rollout_pools
    )
    assert {pool.gpu for pool in recipe.modal.rollout_pools} == {
        "H100",
        "H200",
        "B200",
        "B300",
    }
    assert cfg.tito_model == "qwen36"
    assert cfg.miles_router_timeout == 1800
    assert cfg.max_weight_staleness is None
    assert cfg.async_unused_samples_handler == "drop"
    assert cfg.async_keep_partial_groups_on_abort
    assert cfg.use_dynamic_global_batch_size
    assert (
        cfg.custom_rollout_request_hook_args["rollout_request_weight_version_mode"]
        == "min"
    )
    assert (
        cfg.custom_rollout_request_hook_args["rollout_request_weight_version_lag"] == 1
    )


@pytest.mark.parametrize("ref_load", [None, "/checkpoints/another-model-torch-dist"])
def test_raw_export_uses_the_declared_torch_dist_reference(ref_load):
    recipe = _recipe("qwen3_4b_math")
    recipe.miles.ref_load = ref_load

    with pytest.raises(ValueError, match="Raw export.*TORCH_DIST_CHECKPOINT_PATH"):
        validate_recipe(recipe)


@pytest.mark.parametrize(
    ("environment", "key"),
    [
        ("PREP_ENV", "NVTE_NVFP4_DISABLE_RHT"),
        ("PREP_ENV", "NVTE_NVFP4_4OVER6_ERR_MODE"),
        ("miles.environment", "NVTE_NVFP4_ROW_SCALED_ACTIVATION"),
        ("miles.environment", "NVTE_NVFP4_4OVER6_ERR_MODE"),
        ("SGLANG_SERVER_ENV", "FLASHINFER_NVFP4_4OVER6_ERR_MODE"),
        ("SGLANG_SERVER_ENV", "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION"),
    ],
)
@pytest.mark.parametrize("value", [None, "wrong"])
def test_nvfp4_contract_is_required_during_preparation_training_and_serving(
    environment, key, value
):
    recipe = _recipe("qwen3_6_35b_a3b_nvfp4")
    actual = (
        recipe.miles.environment
        if environment == "miles.environment"
        else getattr(recipe, environment)
    )
    if value is None:
        del actual[key]
    else:
        actual[key] = value

    with pytest.raises(ValueError, match=key):
        validate_recipe(recipe)


def test_nvfp4_error_metric_is_a_recipe_choice():
    recipe = _recipe("qwen3_6_35b_a3b_nvfp4")
    training, serving = nvfp4.environments(error_mode="MAE")
    recipe.NVFP4_TRAINING_ENV = training
    recipe.NVFP4_SERVING_ENV = serving
    recipe.PREP_ENV.update(training)
    recipe.miles.environment.update(training)
    recipe.SGLANG_SERVER_ENV.update(serving)

    validate_recipe(recipe)


def test_nvfp4_actual_environments_must_match_the_declared_metric():
    recipe = _recipe("qwen3_6_35b_a3b_nvfp4")
    training, serving = nvfp4.environments(error_mode="MAE")
    recipe.PREP_ENV.update(training)
    recipe.miles.environment.update(training)
    recipe.SGLANG_SERVER_ENV.update(serving)

    with pytest.raises(ValueError, match="NVTE_NVFP4_4OVER6_ERR_MODE"):
        validate_recipe(recipe)


def test_mapping_arguments_are_encoded_as_json() -> None:
    config = MilesConfig()
    config.custom_rollout_request_hook_args = {
        "minimum_version": 7,
        "retry": True,
    }

    args = config.cli_args()
    index = args.index("--custom-rollout-request-hook-args")

    assert json.loads(args[index + 1]) == {"minimum_version": 7, "retry": True}
