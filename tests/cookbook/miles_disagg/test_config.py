import json
from copy import deepcopy
from dataclasses import replace
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
        "qwen3_6_35b_a3b_hetero_grpo",
        "qwen3_6_35b_a3b_hetero_icepop",
        "qwen3_6_35b_a3b_hetero_score_centering",
        "qwen3_6_35b_a3b_hetero_score_centering_mis",
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
    recipe = _recipe("qwen3_6_35b_a3b_hetero_grpo")

    validate_recipe(recipe)

    assert recipe.ROLLOUT_WEIGHT_VIEWS == {
        "bf16": recipe.BF16_CHECKPOINT_PATH,
        "fp8": recipe.FP8_CHECKPOINT_PATH,
        "nvfp4": recipe.NVFP4_CHECKPOINT_PATH,
    }


def test_mixed_rollout_views_require_matching_pool_assignments() -> None:
    recipe = _recipe("qwen3_6_35b_a3b_hetero_grpo")
    recipe.ROLLOUT_WEIGHT_VIEWS = {"fp8": recipe.FP8_CHECKPOINT_PATH}

    with pytest.raises(ValueError, match="must select every configured weight view"):
        validate_recipe(recipe)


def test_mixed_rollout_view_name_must_be_a_safe_path_component() -> None:
    recipe = _recipe("qwen3_6_35b_a3b_hetero_grpo")
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


def test_qwen36_hetero_fleet_and_trainer():
    recipe = _recipe("qwen3_6_35b_a3b_hetero_grpo")
    cfg = recipe.miles

    validate_recipe(recipe)
    validate_resumable_config(cfg, weight_views=recipe.ROLLOUT_WEIGHT_VIEWS)

    assert cfg.hf_checkpoint == str(recipe.BF16_CHECKPOINT_PATH)
    assert cfg.bf16
    assert cfg.actor_num_nodes == 4
    assert not hasattr(cfg, "fp4_recipe")
    assert not hasattr(cfg, "te_precision_config_file")
    assert "OPEN_TRAINING_NVFP4_FAKE_QAT_FLAG" not in cfg.environment
    assert recipe.ROLLOUT_WEIGHT_VIEWS == {
        "bf16": recipe.BF16_CHECKPOINT_PATH,
        "fp8": recipe.FP8_CHECKPOINT_PATH,
        "nvfp4": recipe.NVFP4_CHECKPOINT_PATH,
    }
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
        "ServerH100FP8": ("H100!", 1, "fp8"),
        "ServerH200FP8": ("H200", 1, "fp8"),
        "ServerB200NVFP4W4A16": ("B200", 1, "nvfp4"),
        "ServerB300NVFP4W4A16": ("B300", 1, "nvfp4"),
        "ServerA100BF16TP2": ("A100-80GB", 2, "bf16"),
        "ServerRTXPRO6000BF16TP2": ("RTX-PRO-6000", 2, "bf16"),
        "ServerH100BF16TP2": ("H100!", 2, "bf16"),
        "ServerH200BF16": ("H200", 1, "bf16"),
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
    for pool in pools.values():
        if pool.weight_view == "bf16":
            assert "--quantization" not in pool.sglang_args
            assert pool.sglang_args["--moe-runner-backend"] == "triton"
    # fp8 KV everywhere. SM120 trtllm_mha rejects fp8 KV, so RTX uses flashinfer.
    assert all(
        pool.sglang_args["--kv-cache-dtype"] == "fp8_e4m3" for pool in pools.values()
    )
    assert {
        name: pools[name].sglang_args["--attention-backend"]
        for name in ("ServerA100BF16TP2", "ServerRTXPRO6000BF16TP2")
    } == {"ServerA100BF16TP2": "flashinfer", "ServerRTXPRO6000BF16TP2": "flashinfer"}
    # FlashInfer GDN requires SM90+, so A100 uses Triton linear attention.
    assert pools["ServerA100BF16TP2"].sglang_args["--linear-attn-decode-backend"] == (
        "triton"
    )
    assert {
        name: (
            pool.min_containers,
            pool.max_containers,
            pool.target_inputs,
            pool.sglang_args["--max-running-requests"],
            pool.sglang_args["--cuda-graph-max-bs-decode"],
        )
        for name, pool in pools.items()
    } == {
        "ServerH100FP8": (8, 8, 16, "24", "24"),
        "ServerH200FP8": (8, 8, 32, "32", "32"),
        "ServerB200NVFP4W4A16": (8, 8, 32, "32", "32"),
        "ServerB300NVFP4W4A16": (8, 8, 48, "64", "64"),
        "ServerA100BF16TP2": (4, 4, 12, "16", "16"),
        "ServerRTXPRO6000BF16TP2": (4, 4, 12, "16", "16"),
        "ServerH100BF16TP2": (4, 4, 32, "48", "48"),
        "ServerH200BF16": (8, 8, 16, "24", "24"),
    }
    # Every pool gets the same 8-GPU floor.
    assert {pool.min_containers * pool.gpus_per_engine for pool in pools.values()} == {
        8
    }
    assert (
        sum(
            pool.min_containers * pool.target_inputs
            for pool in recipe.modal.rollout_pools
        )
        == 1376
        >= cfg.sglang_server_concurrency
        == cfg.async_max_concurrent_samples
    )
    assert recipe.modal.rollout_cpu == 10.0
    assert {
        name: tuple(size // 1024 for size in pool.memory_mib)
        for name, pool in pools.items()
    } == {
        "ServerH100FP8": (120, 512),
        "ServerH200FP8": (120, 512),
        "ServerB200NVFP4W4A16": (88, 512),
        "ServerB300NVFP4W4A16": (88, 512),
        "ServerA100BF16TP2": (224, 512),
        "ServerRTXPRO6000BF16TP2": (224, 512),
        "ServerH100BF16TP2": (224, 512),
        "ServerH200BF16": (216, 512),
    }
    assert (cfg.rollout_batch_size, cfg.n_samples_per_prompt) == (128, 8)
    assert cfg.global_batch_size == 1024
    assert cfg.max_seq_len == 262_144
    assert cfg.context_parallel_size == 4
    assert cfg.save_interval == 10
    assert cfg.environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert cfg.fully_async_drain_during_weight_update
    # Miles sizes the fully-async buffer as factor * rollout_batch_size groups.
    assert cfg.async_data_buffer_capacity_factor * cfg.rollout_batch_size == 256
    assert cfg.sglang_server_concurrency == 64 * 21 == 1344
    assert cfg.environment["MODAL_SWE_AGENT_PROCESSES"] == "64"
    assert cfg.environment["MODAL_SWE_AGENT_THREADS_PER_PROCESS"] == "21"


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


_HETERO_RECIPES = (
    "grpo",
    "icepop",
    "score_centering",
    "score_centering_mis",
)
# Frozen recipes pin runs started before the clean arms, sharing each arm's names.
_FROZEN_RECIPES = {
    "icepop_advanced": "icepop",
    "score_centering_advanced": "score_centering",
}


def _hetero(name):
    return _recipe(f"qwen3_6_35b_a3b_hetero_{name}")


@pytest.mark.parametrize("name", _HETERO_RECIPES + tuple(_FROZEN_RECIPES))
def test_hetero_recipes_share_the_fleet_data_and_run_shape(name):
    recipe = _hetero(name)
    base = import_module("cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero")
    cfg = recipe.miles

    validate_recipe(recipe)
    validate_resumable_config(cfg, weight_views=recipe.ROLLOUT_WEIGHT_VIEWS)

    # One rollout fleet and trainer shape for every arm, on the base's B200 trainer;
    # only the frozen IcePop run keeps the B300 trainer it started on.
    assert replace(recipe.modal, gpu=base.modal.gpu) == base.modal
    assert base.modal.gpu == "B200"
    assert recipe.modal.gpu == ("B300" if name == "icepop_advanced" else "B200")
    assert recipe.ROLLOUT_WEIGHT_VIEWS == base.ROLLOUT_WEIGHT_VIEWS
    assert cfg.rollout_temperature == 1.0
    assert cfg.advantage_estimator == "grpo"
    assert cfg.prompt_data.endswith("/mimo-v2-6-rl-oss/code.jsonl")
    assert not cfg.use_rollout_routing_replay
    assert (
        cfg.rollout_batch_size,
        cfg.n_samples_per_prompt,
        cfg.global_batch_size,
    ) == (128, 8, 1024)
    assert (cfg.num_rollout, cfg.save_interval, cfg.lr) == (500, 10, 1e-6)
    assert cfg.context_parallel_size == 4
    assert cfg.fully_async_drain_during_weight_update
    # Unbounded staleness: no age filter, but every request still asks for the
    # latest published weights of its precision.
    assert cfg.max_weight_staleness is None
    assert cfg.custom_rollout_request_hook_path == (
        "cookbook.common.hooks.gated_rollout_request_hook"
    )
    assert cfg.custom_rollout_request_hook_args == {
        "rollout_request_weight_version_mode": "min"
    }
    slug = _FROZEN_RECIPES.get(name, name).replace("_", "-")
    assert (
        recipe.APP_NAME
        == recipe.EXPERIMENT_VOLUME_NAME
        == f"stitch-qwen36-hetero-{slug}"
    )
    assert cfg.wandb_group == cfg.prometheus_run_name == f"qwen36-hetero-{slug}"
    assert cfg.environment["MODAL_SWE_SANDBOX_APP"] == "stitch-qwen36-hetero-sandbox"
    # Views that finish encoding together share one commit round, and the first wave
    # of sessions starts over five minutes rather than at once. The trainer checkpoint
    # is not written asynchronously: an async writer would hold files open on the
    # mount each publish reloads.
    assert not getattr(cfg, "async_save", False)
    # Saves go to local disk, with room for a second one while the first uploads.
    assert recipe.modal.trainer_local_checkpoint_dir == "/tmp/stitch-local-checkpoints"
    assert recipe.modal.trainer_ephemeral_disk_mib == 1_048_576
    assert cfg.custom_update_weight_post_write_views_path == (
        "cookbook.common.hooks.commit_and_wake_views"
    )
    assert cfg.environment["MODAL_SWE_START_RAMP_SECONDS"] == "300"
    # A replayed support must fit the engine's mask, with room for ties at the cutoff.
    if cfg.rollout_top_k > 0:
        assert all(
            int(pool.sglang_args["--sampling-mask-max-tokens"]) > cfg.rollout_top_k
            for pool in recipe.modal.rollout_pools
        )
    # Modal caps Volume and app names at 64 characters; the app name also carries a run id.
    assert len(recipe.APP_NAME) + len("-r01") < 64


def _public_fields(cfg) -> dict:
    return {
        key: getattr(cfg, key)
        for key in dir(cfg)
        if not key.startswith("_") and not callable(getattr(cfg, key))
    }


_NAMES = {"wandb_group", "prometheus_run_name"}
_SCORE_CENTERING = {
    "disable_grpo_std_normalization": True,
    "loss_type": "score_centering",
    "score_centering_top_k": 128,
    "score_centering_is": "none",
    "skip_actor_forward_only": False,
    "use_rollout_logprobs": True,
}
# Exactly what each arm changes from vanilla GRPO.
_ALGORITHMS = {
    "icepop": {
        "use_tis": True,
        "custom_tis_function_path": (
            "miles.backends.training_utils.loss_hub.corrections.icepop_function"
        ),
        "tis_clip_low": 0.5,
        "tis_clip": 5.0,
    },
    "score_centering": _SCORE_CENTERING,
    "score_centering_mis": {
        **_SCORE_CENTERING,
        "score_centering_is": "mis",
        "score_centering_mis_low": 0.5,
        "score_centering_mis_high": 5.0,
    },
}


def test_vanilla_grpo_samples_the_full_vocabulary_with_a_batch_token_mean():
    grpo = _hetero("grpo").miles
    assert (grpo.rollout_top_p, grpo.rollout_top_k) == (1.0, -1)
    assert not getattr(grpo, "disable_grpo_std_normalization", False)
    assert grpo.calculate_per_token_loss
    assert not getattr(grpo, "prompt_mean_loss", False)
    assert not getattr(grpo, "freeze_moe_router", False)
    assert getattr(grpo, "loss_type", None) is None
    # One optimizer step per batch against the trainer's own detached log-probs.
    assert (grpo.use_tis, grpo.use_rollout_logprobs) == (False, False)
    assert grpo.skip_actor_forward_only


@pytest.mark.parametrize("name", sorted(_ALGORITHMS))
def test_each_arm_changes_only_its_algorithm_from_vanilla_grpo(name):
    grpo = _public_fields(_hetero("grpo").miles)
    arm = _public_fields(_hetero(name).miles)
    changed = {
        key: arm.get(key)
        for key in grpo.keys() | arm.keys()
        if arm.get(key) != grpo.get(key)
    }
    assert {key: value for key, value in changed.items() if key not in _NAMES} == (
        _ALGORITHMS[name]
    )
    assert _hetero(name).modal is _hetero("grpo").modal


def test_frozen_recipes_keep_their_runs_configuration():
    advanced = _hetero("icepop_advanced").miles
    assert (advanced.rollout_top_p, advanced.rollout_top_k) == (0.97, 4096)
    assert (advanced.tis_clip_low, advanced.tis_clip) == (0.2, 5.0)
    assert advanced.disable_grpo_std_normalization
    assert advanced.prompt_mean_loss and not advanced.calculate_per_token_loss
    assert advanced.freeze_moe_router
    args = advanced.cli_args()
    assert "--prompt-mean-loss" in args and "--freeze-moe-router" in args
    assert "--calculate-per-token-loss" not in args
    sc_advanced = _hetero("score_centering_advanced").miles
    assert (sc_advanced.rollout_top_p, sc_advanced.rollout_top_k) == (0.97, 64)
    assert sc_advanced.rollout_top_k <= sc_advanced.score_centering_top_k
    assert {
        key: getattr(sc_advanced, key) for key in _SCORE_CENTERING
    } == _SCORE_CENTERING


# The mismatch ladder: one B200 fleet per rung, each serving one weight view.
_LADDER_FLEETS = {"b200_bf16": "bf16", "b200_nvfp4": "nvfp4"}


def _ladder(fleet, name):
    return _recipe(f"qwen3_6_35b_a3b_{fleet}_{name}")


@pytest.mark.parametrize("fleet", sorted(_LADDER_FLEETS))
@pytest.mark.parametrize("name", _HETERO_RECIPES)
def test_ladder_recipes_change_only_the_fleet_from_their_hetero_arm(fleet, name):
    recipe = _ladder(fleet, name)
    hetero = _hetero(name)
    cfg = recipe.miles

    validate_recipe(recipe)
    validate_resumable_config(cfg, weight_views=recipe.ROLLOUT_WEIGHT_VIEWS)

    # The trainer publishes only the view its fleet serves, but every save exports all
    # three precisions, as the heterogeneous runs' saves do.
    view = _LADDER_FLEETS[fleet]
    assert recipe.ROLLOUT_WEIGHT_VIEWS == {view: hetero.ROLLOUT_WEIGHT_VIEWS[view]}
    assert recipe.EXPORT_WEIGHT_VIEWS == hetero.ROLLOUT_WEIGHT_VIEWS
    assert not getattr(hetero, "EXPORT_WEIGHT_VIEWS", None)
    assert replace(recipe.modal, rollout_pools=hetero.modal.rollout_pools) == (
        hetero.modal
    )
    (pool,) = recipe.modal.rollout_pools
    assert (pool.gpu, pool.gpus_per_engine, pool.weight_view) == ("B200", 1, view)
    # A fixed fleet that holds every concurrent session at the engine's load.
    assert pool.min_containers == pool.max_containers
    assert pool.min_containers * pool.target_inputs == (
        cfg.async_max_concurrent_samples
    )
    assert _public_fields(cfg).keys() - _NAMES == (
        _public_fields(hetero.miles).keys() - _NAMES
    )
    assert {
        key: value for key, value in _public_fields(cfg).items() if key not in _NAMES
    } == {
        key: value
        for key, value in _public_fields(hetero.miles).items()
        if key not in _NAMES
    }
    slug = f"{fleet}-{name}".replace("_", "-")
    assert recipe.APP_NAME == recipe.EXPERIMENT_VOLUME_NAME == f"stitch-qwen36-{slug}"
    assert cfg.wandb_group == cfg.prometheus_run_name == f"qwen36-{slug}"
    assert len(recipe.APP_NAME) + len("-r01") < 64


def test_export_views_must_include_every_rollout_view():
    recipe = _ladder("b200_nvfp4", "grpo")
    views = recipe.EXPORT_WEIGHT_VIEWS
    recipe.EXPORT_WEIGHT_VIEWS = {"bf16": views["bf16"], "fp8": views["fp8"]}
    with pytest.raises(ValueError, match="must include every rollout weight view"):
        validate_recipe(recipe)
    recipe.EXPORT_WEIGHT_VIEWS = {**views, "nvfp4": views["bf16"]}
    with pytest.raises(ValueError, match="must include every rollout weight view"):
        validate_recipe(recipe)
    recipe.EXPORT_WEIGHT_VIEWS = {**views, "../bf16": views["bf16"]}
    with pytest.raises(ValueError):
        validate_recipe(recipe)


def test_b200_nvfp4_fleet_is_the_hetero_b200_engine():
    hetero_pool = next(
        pool
        for pool in _hetero("grpo").modal.rollout_pools
        if pool.name == "ServerB200NVFP4W4A16"
    )
    (pool,) = _ladder("b200_nvfp4", "grpo").modal.rollout_pools
    assert replace(pool, min_containers=8, max_containers=8) == hetero_pool
    assert pool.min_containers == 42


def test_b200_bf16_fleet_serves_bf16_weights_and_kv_cache():
    (pool,) = _ladder("b200_bf16", "grpo").modal.rollout_pools
    args = pool.sglang_args
    assert args["--kv-cache-dtype"] == "bf16"
    assert "--quantization" not in args
    assert args["--moe-runner-backend"] == "triton"
    assert args["--attention-backend"] == "trtllm_mha"
    assert (pool.target_inputs, args["--max-running-requests"]) == (32, "32")
    assert pool.min_containers == 42


def test_ladder_and_hetero_recipes_have_their_own_app_volume_and_wandb_group():
    recipes = [_hetero(name) for name in _HETERO_RECIPES] + [
        _ladder(fleet, name) for fleet in _LADDER_FLEETS for name in _HETERO_RECIPES
    ]
    for field in ("APP_NAME", "EXPERIMENT_VOLUME_NAME"):
        assert len({getattr(recipe, field) for recipe in recipes}) == len(recipes)
    assert len({recipe.miles.wandb_group for recipe in recipes}) == len(recipes)


def test_study_trainers_and_pools_may_run_on_any_cloud():
    """The AWS-only trainer pin left trainers queued for GPUs; every study recipe now
    lets Modal place its trainer and pools on any provider."""
    recipes = [_hetero(name) for name in _HETERO_RECIPES] + [
        _ladder(fleet, name) for fleet in _LADDER_FLEETS for name in _HETERO_RECIPES
    ]
    for recipe in recipes:
        assert recipe.modal.trainer_cloud is None
        assert recipe.modal.cloud is None
        assert all(pool.cloud is None for pool in recipe.modal.rollout_pools)
