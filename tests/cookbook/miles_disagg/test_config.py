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


@pytest.mark.parametrize(
    "name", ["qwen3_4b_math", "qwen3_6_35b_a3b_nvfp4", "glm5_3_nvfp4"]
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


def test_qwen36_target_only_recipe_omits_mtp_from_training_and_conversion():
    recipe = _recipe("qwen3_6_35b_a3b_nvfp4")

    assert recipe.miles.mtp_num_layers == 0
    assert "--mtp-num-layers 0" in recipe.modal.torch_dist_convert_extra_args
    assert "--speculative-algorithm" not in recipe.SGLANG_SERVER_ARGS
    assert "mtp." in recipe.miles.hf_export_source_tensor_prefixes


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


def test_external_agents_require_fixed_session_server_ports() -> None:
    recipe = _recipe("qwen3_6_35b_a3b_nvfp4")
    recipe.miles.session_server_port = None

    with pytest.raises(ValueError, match="fixed session_server_port"):
        validate_recipe(recipe)


def test_swe_recipe_uses_harbor_without_legacy_modal_swe_hooks() -> None:
    recipe = _recipe("qwen3_6_35b_a3b_nvfp4")
    fields = recipe.miles._fields()

    assert fields["custom_agent_function_path"] == "harbor_agent_function.run"
    assert fields["custom_rm_path"] == "generate.reward_func"
    assert fields.get("rollout_function_path") is None
    assert fields["use_session_server"] is True
    assert recipe.modal.forward_session_server_ports is True
    assert not any("modal_swe" in str(value) for value in fields.values())
    assert not any(key.startswith("MODAL_SWE_") for key in recipe.miles.environment)
