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
        "qwen3_6_35b_a3b_mimo_code_heterogeneous",
        "qwen3_6_35b_a3b_mimo_code_heterogeneous_tis",
        "qwen3_6_35b_a3b_mimo_code",
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


def test_mixed_rollout_views_use_generic_checkpoint_paths() -> None:
    recipe = _recipe("qwen3_6_35b_a3b_mimo_code_heterogeneous")

    validate_recipe(recipe)

    assert recipe.ROLLOUT_WEIGHT_VIEWS == {
        "fp8": recipe.FP8_CHECKPOINT_PATH,
        "nvfp4": recipe.NVFP4_CHECKPOINT_PATH,
    }


def test_mixed_rollout_views_require_matching_pool_assignments() -> None:
    recipe = _recipe("qwen3_6_35b_a3b_mimo_code_heterogeneous")
    recipe.ROLLOUT_WEIGHT_VIEWS = {"fp8": recipe.FP8_CHECKPOINT_PATH}

    with pytest.raises(ValueError, match="must select every configured weight view"):
        validate_recipe(recipe)


def test_mixed_rollout_view_name_must_be_a_safe_path_component() -> None:
    recipe = _recipe("qwen3_6_35b_a3b_mimo_code_heterogeneous")
    recipe.ROLLOUT_WEIGHT_VIEWS = {"../fp8": recipe.FP8_CHECKPOINT_PATH}

    with pytest.raises(ValueError, match="invalid weight_view"):
        validate_recipe(recipe)


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_6_35b_a3b_mimo_code",
        "qwen3_6_35b_a3b_swebench_pro",
        "qwen3_6_35b_a3b_nvfp4",
    ],
)
def test_qwen36_target_only_recipe_omits_mtp_from_training_and_conversion(name):
    recipe = _recipe(name)

    assert recipe.miles.mtp_num_layers == 0
    assert "--mtp-num-layers 0" in recipe.modal.torch_dist_convert_extra_args
    assert all(
        "--speculative-algorithm" not in args for args in _rollout_server_args(recipe)
    )
    assert recipe.miles.hf_export_source_tensor_prefixes == [
        "model.visual.",
        "mtp.",
    ]


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_6_35b_a3b_mimo_code",
        "qwen3_6_35b_a3b_swebench_pro",
        "qwen3_6_35b_a3b_nvfp4",
    ],
)
def test_qwen36_recipes_use_qwen36_tito_template(name):
    assert _recipe(name).miles.tito_model == "qwen36"


def test_glm53_recipe_uses_glm53_tito_template():
    assert _recipe("glm5_3_nvfp4").miles.tito_model == "glm53"


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_4b_math",
        "qwen3_6_35b_a3b_mimo_code",
        "qwen3_6_35b_a3b_swebench_pro",
        "qwen3_6_35b_a3b_nvfp4",
        "glm5_3_nvfp4",
    ],
)
def test_request_retry_policy_uses_miles_transport_arguments(name):
    cfg = _recipe(name).miles

    assert cfg.rollout_request_max_attempts > 1
    assert cfg.rollout_request_retry_interval == 1.0
    assert "rollout_request_max_attempts" not in cfg.custom_rollout_request_hook_args
    assert "rollout_request_retry_interval" not in cfg.custom_rollout_request_hook_args


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_4b_math",
        "qwen3_6_35b_a3b_mimo_code",
        "qwen3_6_35b_a3b_swebench_pro",
        "qwen3_6_35b_a3b_nvfp4",
        "glm5_3_nvfp4",
    ],
)
def test_grouped_agentic_recipes_retain_usable_partial_groups(name):
    cfg = _recipe(name).miles

    assert cfg.keep_partial_groups_on_abort
    assert cfg.use_dynamic_global_batch_size
    assert not getattr(cfg, "use_fault_tolerance", False)
    assert not hasattr(cfg, "rollout_health_check_first_wait")


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
    assert cfg.async_max_concurrent_samples == 1024
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
    assert cfg.keep_partial_groups_on_abort
    assert cfg.use_dynamic_global_batch_size
    assert (
        cfg.custom_rollout_request_hook_args["rollout_request_weight_version_mode"]
        == "min"
    )
    assert (
        cfg.custom_rollout_request_hook_args["rollout_request_weight_version_lag"] == 1
    )


def test_qwen36_mimo_code_uses_validated_context_and_code_tasks():
    recipe = _recipe("qwen3_6_35b_a3b_mimo_code")
    cfg = recipe.miles

    assert cfg.prompt_data.endswith("/mimo-v2-6-rl-oss/code.jsonl")
    assert cfg.max_seq_len == 262_144
    assert cfg.rollout_top_p == 0.95
    assert cfg.rollout_top_k == 1024
    assert cfg.rollout_max_response_len == 32_768
    assert cfg.rollout_batch_size == 64
    assert cfg.n_samples_per_prompt == 8
    assert cfg.global_batch_size == 512
    assert cfg.num_rollout == 10
    assert cfg.context_parallel_size == 2
    assert cfg.max_tokens_per_gpu == 16_384
    assert cfg.log_probs_max_tokens_per_gpu == 16_384
    assert cfg.use_tis
    assert (
        cfg.custom_tis_function_path
        == "miles.backends.training_utils.loss_hub.corrections.icepop_function"
    )
    assert cfg.tis_clip_low == 0.5
    assert cfg.tis_clip == 5.0
    assert cfg.environment["MODAL_SWE_MAX_STEPS"] == "500"
    assert cfg.environment["MODAL_SWE_EPISODE_TIMEOUT"] == "4800"
    assert cfg.environment["MODAL_SWE_MODEL_REQUEST_TIMEOUT"] == "3600"
    assert cfg.miles_router_timeout == 3600
    assert cfg.environment["MODAL_SWE_EXEC_TIMEOUT"] == "300"
    assert cfg.environment["MODAL_SWE_AGENT_PROFILE"] == "mimo-code-bash"
    assert cfg.environment["MODAL_SWE_AGENT_PROCESSES"] == "64"
    assert cfg.environment["MODAL_SWE_AGENT_THREADS_PER_PROCESS"] == "32"
    assert "MODAL_SWE_SANDBOX_BOOT_CONCURRENCY_PER_PROCESS" not in cfg.environment
    assert cfg.async_max_concurrent_samples == 2048
    assert cfg.sglang_server_concurrency == 2048
    assert cfg.session_server_workers == 64
    assert cfg.session_samples_timeout == 600
    assert len(recipe.modal.rollout_pools) == 4
    assert {pool.gpu for pool in recipe.modal.rollout_pools} == {
        "H100",
        "H200",
        "B200",
        "B300",
    }
    assert {
        pool.gpu: (pool.min_containers, pool.max_containers)
        for pool in recipe.modal.rollout_pools
    } == {
        "H100": (32, 48),
        "H200": (64, 96),
        "B200": (32, 48),
        "B300": (32, 48),
    }
    assert {pool.gpu: pool.gpus_per_engine for pool in recipe.modal.rollout_pools} == {
        "H100": 2,
        "H200": 1,
        "B200": 1,
        "B300": 1,
    }
    assert {pool.gpu: pool.target_inputs for pool in recipe.modal.rollout_pools} == {
        "H100": 16,
        "H200": 8,
        "B200": 16,
        "B300": 16,
    }
    assert (
        sum(
            pool.min_containers * pool.target_inputs
            for pool in recipe.modal.rollout_pools
        )
        == cfg.sglang_server_concurrency
    )
    assert {
        pool.gpu: pool.sglang_args["--tp"] for pool in recipe.modal.rollout_pools
    } == {
        "H100": "2",
        "H200": "1",
        "B200": "1",
        "B300": "1",
    }
    assert all(
        pool.sglang_args["--context-length"] == "262144"
        for pool in recipe.modal.rollout_pools
    )
    assert all(
        pool.sglang_args["--kv-cache-dtype"] == "fp8_e4m3"
        for pool in recipe.modal.rollout_pools
    )
    assert all(
        pool.sglang_args["--max-running-requests"] == "24"
        for pool in recipe.modal.rollout_pools
    )
    assert all(
        pool.sglang_args["--cuda-graph-max-bs-decode"] == "24"
        for pool in recipe.modal.rollout_pools
    )


def test_qwen36_mimo_heterogeneous_base_is_uncorrected_grpo():
    recipe = _recipe("qwen3_6_35b_a3b_mimo_code_heterogeneous")
    cfg = recipe.miles

    validate_recipe(recipe)
    validate_resumable_config(cfg, weight_views=recipe.ROLLOUT_WEIGHT_VIEWS)

    assert cfg.hf_checkpoint == str(recipe.BF16_CHECKPOINT_PATH)
    assert cfg.bf16
    assert not hasattr(cfg, "fp4_recipe")
    assert not hasattr(cfg, "te_precision_config_file")
    assert "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG" not in cfg.environment
    assert recipe.ROLLOUT_WEIGHT_VIEWS == {
        "fp8": recipe.FP8_CHECKPOINT_PATH,
        "nvfp4": recipe.NVFP4_CHECKPOINT_PATH,
    }
    assert cfg.rollout_top_p == 1.0
    assert cfg.rollout_top_k == -1
    assert cfg.advantage_estimator == "grpo"
    assert not cfg.use_tis
    assert not cfg.get_mismatch_metrics
    assert cfg.custom_tis_function_path is None
    assert not cfg.use_rollout_routing_replay
    assert all(
        "--enable-return-routed-experts" not in pool.sglang_args
        for pool in recipe.modal.rollout_pools
    )

    pools = {pool.name: pool for pool in recipe.modal.rollout_pools}
    assert {
        name: (pool.gpu, pool.gpus_per_engine, pool.weight_view)
        for name, pool in pools.items()
    } == {
        "ServerH100FP8": ("H100", 1, "fp8"),
        "ServerH200FP8": ("H200", 1, "fp8"),
        "ServerB200NVFP4W4A16": ("B200", 1, "nvfp4"),
        "ServerB300NVFP4W4A16": ("B300", 1, "nvfp4"),
    }
    assert all(
        "--quantization" not in pool.sglang_args
        and "--moe-runner-backend" not in pool.sglang_args
        for name, pool in pools.items()
        if "FP8" in name
    )
    for name in ("ServerB200NVFP4W4A16", "ServerB300NVFP4W4A16"):
        pool = pools[name]
        assert pool.sglang_args["--quantization"] == "modelopt_fp4"
        assert pool.sglang_args["--moe-runner-backend"] == "flashinfer_cutedsl"
        assert pool.environment == {"SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1"}
    assert {
        pool.gpu: (
            pool.min_containers,
            pool.max_containers,
            pool.target_inputs,
            pool.sglang_args["--max-running-requests"],
            pool.sglang_args["--cuda-graph-max-bs-decode"],
        )
        for pool in pools.values()
    } == {
        "H100": (16, 48, 16, "24", "24"),
        "H200": (16, 48, 64, "64", "64"),
        "B200": (16, 48, 64, "64", "64"),
        "B300": (16, 48, 64, "64", "64"),
    }
    assert (
        sum(
            pool.min_containers * pool.target_inputs
            for pool in recipe.modal.rollout_pools
        )
        >= cfg.sglang_server_concurrency
        == cfg.async_max_concurrent_samples
    )


def test_qwen36_mimo_heterogeneous_tis_is_a_thin_control():
    base = _recipe("qwen3_6_35b_a3b_mimo_code_heterogeneous")
    recipe = _recipe("qwen3_6_35b_a3b_mimo_code_heterogeneous_tis")
    cfg = recipe.miles

    validate_recipe(recipe)
    validate_resumable_config(cfg, weight_views=recipe.ROLLOUT_WEIGHT_VIEWS)

    assert recipe.ROLLOUT_WEIGHT_VIEWS == base.ROLLOUT_WEIGHT_VIEWS
    assert all(
        pool.sglang_args.get("--enable-return-routed-experts") == ""
        for pool in recipe.modal.rollout_pools
    )
    assert cfg.use_tis
    assert cfg.num_rollout == 5
    assert cfg.tis_clip_low == 0.5
    assert cfg.tis_clip == 2.0
    assert cfg.use_rollout_routing_replay
    assert not cfg.get_mismatch_metrics
    assert cfg.custom_tis_function_path is None
    assert cfg.rollout_top_p == 0.95
    assert cfg.rollout_top_k == 4096
    assert base.miles.rollout_top_p == 1.0
    assert base.miles.rollout_top_k == -1
    assert all(
        pool.sglang_args["--sampling-mask-max-tokens"] == "8192"
        for pool in recipe.modal.rollout_pools
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
