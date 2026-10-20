import dataclasses
import json
import sys
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace

import pytest

from cookbook.common import serving_image
from cookbook.miles_disagg import harbor_config, nvfp4, trainer_image
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


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_8_27b_harbor_smoke",
        "qwen3_8_27b_harbor_mix_v0",
        "qwen3_8_27b_harbor_mix_v1",
        "qwen3_8_27b_harbor_mix_v1_h200",
        "qwen3_8_27b_harbor_mix_v1b",
        "qwen3_8_27b_harbor_mix_v1b_h200",
    ],
)
def test_harbor_recipes_are_valid(name):
    validate_recipe(_recipe(name))


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_8_27b_harbor_smoke",
        "qwen3_8_27b_harbor_mix_v0",
        "qwen3_8_27b_harbor_mix_v1",
        "qwen3_8_27b_harbor_mix_v1_h200",
        "qwen3_8_27b_harbor_mix_v1b",
        "qwen3_8_27b_harbor_mix_v1b_h200",
    ],
)
def test_harbor_recipes_train_with_score_centering_without_dflash(name):
    recipe = _recipe(name)
    args = recipe.miles.cli_args()

    assert args[args.index("--loss-type") + 1] == "score_centering"
    assert args[args.index("--score-centering-top-k") + 1] == "128"
    assert args[args.index("--score-centering-is") + 1] == "mis"
    assert args[args.index("--score-centering-mis-low") + 1] == "0.5"
    assert args[args.index("--score-centering-mis-high") + 1] == "5.0"
    assert "--use-tis" not in args and "--custom-tis-function-path" not in args
    assert args[args.index("--rollout-top-p") + 1] == "0.97"
    assert args[args.index("--rollout-top-k") + 1] == "64"
    assert "--sglang-speculative-algorithm" not in args
    assert not [k for k in recipe.SGLANG_SERVER_ARGS if k.startswith("--speculative-")]
    assert args[args.index("--rollout-request-max-attempts") + 1] == "1200"
    assert "--get-mismatch-metrics" not in args
    kwargs = json.loads(args[args.index("--apply-chat-template-kwargs") + 1])
    assert kwargs == {"reasoning_effort": "medium"}


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_8_27b_harbor_smoke",
        "qwen3_8_27b_harbor_mix_v0",
        "qwen3_8_27b_harbor_mix_v1",
        "qwen3_8_27b_harbor_mix_v1_h200",
        "qwen3_8_27b_harbor_mix_v1b",
        "qwen3_8_27b_harbor_mix_v1b_h200",
    ],
)
def test_only_harbor_recipes_pin_the_score_centering_stack(name):
    recipe = _recipe(name)

    assert recipe.MILES_REPO_REF == harbor_config.MILES_REPO_REF
    assert recipe.MILES_IMAGE_TAG == harbor_config.MILES_IMAGE_TAG
    assert recipe.SGLANG_RUNTIME.branch == "stitch-sglang-hetero"
    assert recipe.SIDECAR_RESPONSE_KEEPALIVE_AFTER == 60.0
    assert trainer_image.MILES_REPO_REF != harbor_config.MILES_REPO_REF
    assert serving_image.DEFAULT_SGLANG_RUNTIME.branch != "stitch-sglang-hetero"


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_8_27b_harbor_mix_v0",
        "qwen3_8_27b_harbor_mix_v1",
        "qwen3_8_27b_harbor_mix_v1_h200",
        "qwen3_8_27b_harbor_mix_v1b",
        "qwen3_8_27b_harbor_mix_v1b_h200",
    ],
)
def test_harbor_mix_drops_groups_without_signal(name):
    recipe = _recipe(name)

    validate_resumable_config(recipe.miles)
    assert recipe.miles.dynamic_sampling_filter_path.endswith(
        "check_reward_nonzero_std"
    )
    assert recipe.miles.environment["HARBOR_TASKS_DIR"] == f"{recipe.TASKS_ROOT}/tasks"
    assert recipe.miles.prompt_data == f"{recipe.TASKS_ROOT}/train.jsonl"
    assert recipe.miles.environment["AGENT_MAX_OUTPUT_TOKENS"] == "16384"
    assert recipe.miles.rollout_max_response_len == 16384
    sandbox = json.loads(recipe.miles.environment["HARBOR_ENV_KWARGS"])
    trial_timeout = int(recipe.miles.environment["AGENT_TRIAL_TIMEOUT"])
    assert trial_timeout < sandbox["sandbox_timeout_secs"] < 24 * 3600


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_8_27b_harbor_smoke",
        "qwen3_8_27b_harbor_mix_v0",
        "qwen3_8_27b_harbor_mix_v1",
        "qwen3_8_27b_harbor_mix_v1_h200",
        "qwen3_8_27b_harbor_mix_v1b",
        "qwen3_8_27b_harbor_mix_v1b_h200",
    ],
)
def test_harbor_rollout_workers_run_without_uvloop(name):
    assert _recipe(name).miles.environment["RAY_USE_UVLOOP"] == "0"


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_8_27b_harbor_mix_v0",
        "qwen3_8_27b_harbor_mix_v1",
        "qwen3_8_27b_harbor_mix_v1_h200",
        "qwen3_8_27b_harbor_mix_v1b",
        "qwen3_8_27b_harbor_mix_v1b_h200",
    ],
)
def test_harbor_mix_saves_every_step_and_waits_for_long_replies(name):
    args = _recipe(name).miles.cli_args()

    def value(flag):
        return args[args.index(flag) + 1]

    assert (value("--save-interval"), value("--save-retain-interval")) == ("1", "5")
    assert value("--miles-router-timeout") == "3600"


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_8_27b_harbor_mix_v0",
        "qwen3_8_27b_harbor_mix_v1",
        "qwen3_8_27b_harbor_mix_v1b",
    ],
)
def test_harbor_mix_serves_and_trains_at_128k(name):
    recipe = _recipe(name)

    assert recipe.MAX_SEQ_LEN == recipe.miles.max_seq_len == 131_072
    assert recipe.SGLANG_SERVER_ARGS["--context-length"] == str(131_072 + 8)
    assert recipe.miles.environment["HARBOR_MAX_SEQ_LEN"] == "131072"
    assert recipe.miles.environment["AGENT_MAX_INPUT_TOKENS"] == str(131_072 - 16_384)
    assert recipe.miles.context_parallel_size == 2
    assert (
        recipe.modal.rollout_min_containers == recipe.modal.rollout_max_containers == 32
    )
    assert recipe.miles.sglang_server_concurrency == 256


def test_harbor_mix_v1_h200_trainer_splits_each_sample_over_four_gpus():
    recipe = _recipe("qwen3_8_27b_harbor_mix_v1_h200")
    b300 = _recipe("qwen3_8_27b_harbor_mix_v1")

    assert (recipe.modal.gpu, recipe.miles.context_parallel_size) == ("H200", 4)
    assert recipe.modal == dataclasses.replace(b300.modal, gpu="H200")
    assert (recipe.APP_NAME, recipe.EXPERIMENT_VOLUME_NAME) == (
        b300.APP_NAME,
        b300.EXPERIMENT_VOLUME_NAME,
    )

    def without_cp(args):
        i = args.index("--context-parallel-size")
        return args[:i] + args[i + 2 :]

    assert without_cp(recipe.miles.cli_args()) == without_cp(b300.miles.cli_args())


def test_harbor_mix_v1_caps_agents_at_one_hour():
    # Miles' Harbor function passes timeout_multiplier=2.0, and Harbor multiplies the override by it.
    env = _recipe("qwen3_8_27b_harbor_mix_v1").miles.environment

    assert env["AGENT_TIMEOUT"] == "1800"
    assert env["HARBOR_TASKS_DIR"] == "/harbor-envs/mixture-v1/tasks"


def test_harbor_mix_v1b_changes_only_the_task_mix_and_names():
    # Runs v1 and v1b compare mixture weights, so every other setting must match, and no name is shared.
    v1, v1b = (
        _recipe("qwen3_8_27b_harbor_mix_v1"),
        _recipe("qwen3_8_27b_harbor_mix_v1b"),
    )
    args = zip(v1.miles.cli_args(), v1b.miles.cli_args(), strict=True)
    env, env_b = v1.miles.environment, v1b.miles.environment

    # The W&B group and the Prometheus run name, then the prompt data.
    assert {(a, b) for a, b in args if a != b} == {
        ("qwen3-8-27b-harbor-mix-v1", "qwen3-8-27b-harbor-mix-v1b"),
        ("/harbor-envs/mixture-v1/train.jsonl", "/harbor-envs/mixture-v1b/train.jsonl"),
    }
    assert {k for k in env | env_b if env.get(k) != env_b.get(k)} == {
        "HARBOR_TASKS_DIR",
        "HARBOR_ENV_KWARGS",
    }
    assert env_b["HARBOR_TASKS_DIR"] == "/harbor-envs/mixture-v1b/tasks"
    assert json.loads(env_b["HARBOR_ENV_KWARGS"]) == {
        "app_name": "qwen3-8-27b-harbor-mix-v1b-sandboxes",
        "sandbox_timeout_secs": 7800,
    }
    assert v1b.modal == v1.modal
    assert v1b.APP_NAME == "stitch-qwen3-8-27b-harbor-mix-v1b" != v1.APP_NAME
    assert v1b.EXPERIMENT_VOLUME_NAME == "stitch-miles-qwen3-8-27b-harbor-mix-v1b"
    assert v1b.EXPERIMENT_VOLUME_NAME != v1.EXPERIMENT_VOLUME_NAME
    assert v1b.TRAINER_EXTRA_VOLUMES["/harbor-trials"].name == "harbor-mix-v1b-trials"



def test_harbor_mix_v1b_h200_is_v1b_on_the_h200_trainer():
    # A and B must train on the same GPU type, so v1b on H200 matches v1 on H200.
    recipe = _recipe("qwen3_8_27b_harbor_mix_v1b_h200")
    b300 = _recipe("qwen3_8_27b_harbor_mix_v1b")
    v1 = _recipe("qwen3_8_27b_harbor_mix_v1_h200")

    assert recipe.modal == v1.modal == dataclasses.replace(b300.modal, gpu="H200")
    assert recipe.miles.context_parallel_size == v1.miles.context_parallel_size == 4
    assert (recipe.APP_NAME, recipe.EXPERIMENT_VOLUME_NAME) == (
        b300.APP_NAME,
        b300.EXPERIMENT_VOLUME_NAME,
    )
    assert recipe.miles.environment == b300.miles.environment

    def without_cp(args):
        i = args.index("--context-parallel-size")
        return args[:i] + args[i + 2 :]

    assert without_cp(recipe.miles.cli_args()) == without_cp(b300.miles.cli_args())

@dataclasses.dataclass(frozen=True)
class _Binding:
    kwargs: object
    env: object


def test_harbor_terminus_gets_no_env_in_the_task_shell(monkeypatch):
    miles_harbor = SimpleNamespace(
        HARNESS_BINDINGS={
            name: _Binding(kwargs="k", env=lambda url, key: {"OPENAI_API_KEY": key})
            for name in ("terminus-2", "terminus-1", "terminus", "claude-code")
        },
        run="miles-run",
    )
    monkeypatch.setitem(sys.modules, "harbor_agent_function", miles_harbor)
    monkeypatch.delitem(
        sys.modules, "cookbook.miles_disagg.harbor_agent", raising=False
    )

    agent = import_module("cookbook.miles_disagg.harbor_agent")

    bindings = miles_harbor.HARNESS_BINDINGS
    assert bindings["terminus-2"].env("url", "key") == {}
    assert bindings["terminus-2"].kwargs == "k"
    assert bindings["claude-code"].env("url", "key") == {"OPENAI_API_KEY": "key"}
    assert agent.run == "miles-run"
    recipe = _recipe("qwen3_8_27b_harbor_mix_v0")
    assert (
        recipe.miles.custom_agent_function_path
        == "cookbook.miles_disagg.harbor_agent.run"
    )


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
