import asyncio
import base64
import importlib
import itertools
import json
import sys
import types
from enum import Enum
from types import SimpleNamespace

import pytest

from cookbook.common import server
from cookbook.miles_disagg import evaluation
from cookbook.miles_disagg.eval_configs import swebench_pro_hetero as spec
from cookbook.miles_disagg.resume import export_version

HETERO_ARMS = [
    "qwen3_6_35b_a3b_hetero_grpo",
    "qwen3_6_35b_a3b_hetero_icepop",
    "qwen3_6_35b_a3b_hetero_score_centering",
    "qwen3_6_35b_a3b_hetero_score_centering_mis",
    "qwen3_6_35b_a3b_hetero_grpo_top_p",
    "qwen3_6_35b_a3b_hetero_score_centering_mis_top_p",
]
# The harness: what the policy sees and is scored by. Eval must not change any of it.
HARNESS_FIELDS = [
    "custom_generate_function_path",
    "custom_agent_function_path",
    "custom_rm_path",
    "use_session_server",
    "session_sample_picker_path",
    "session_sample_postprocessor_path",
    "tito_model",
    "apply_chat_template",
    "input_key",
    "metadata_key",
    "max_seq_len",
    "rollout_max_response_len",
    "rollout_temperature",
]
HARNESS_ENVIRONMENT = [
    "MODAL_SWE_AGENT_PROFILE",
    "MODAL_SWE_MAX_STEPS",
    "MODAL_SWE_EPISODE_TIMEOUT",
    "MODAL_SWE_MODEL_REQUEST_TIMEOUT",
    "MODAL_SWE_EXEC_TIMEOUT",
    "MODAL_SWE_MEMORY_MIB",
    "MODAL_SWE_SETUP_TIMEOUT",
    "MODAL_SWE_VERIFY_TIMEOUT",
    "MODAL_SWE_AGENT_THREADS_PER_PROCESS",
]


def _recipe(name):
    return importlib.import_module(f"cookbook.miles_disagg.configs.{name}")


def _eval_config(miles_cfg, concurrency=384):
    return evaluation.eval_miles_config(
        miles_cfg,
        dataset=spec.DATASET,
        tasks_dir=spec.TASKS_DIR,
        sandbox_app=spec.SANDBOX_APP,
        concurrency=concurrency,
        dump_template="/stitch/eval/dump/{rollout_id}.pt",
    )


def test_pass_at_k_matches_enumerating_every_k_subset():
    n = 8
    for c, k in itertools.product(range(n + 1), range(1, n + 1)):
        outcomes = [1] * c + [0] * (n - c)
        subsets = list(itertools.combinations(outcomes, k))
        expected = sum(any(subset) for subset in subsets) / len(subsets)
        assert evaluation.pass_at_k(n, c, k) == pytest.approx(expected)


def test_summary_reports_pass_at_1_to_n_only_once_every_task_is_complete():
    rewards = {"a": [1, 0, 0, 0], "b": [0, 0, 0, 0], "c": [1, 1, 1, 1]}
    records = [
        {"instance_id": task, "sample_index": index, "reward": reward}
        for task, values in rewards.items()
        for index, reward in enumerate(values)
    ]

    partial = evaluation.summarize(records[:-1], n_samples=4, n_tasks=3)
    complete = evaluation.summarize(records, n_samples=4, n_tasks=3)

    assert partial["complete"] is False and "pass@1" not in partial
    assert partial["tasks_complete"] == 2 and partial["samples_scored"] == 11
    assert complete["complete"] is True
    # pass@1 is the mean reward; pass@n is the fraction of tasks solved at least once.
    assert complete["pass@1"] == pytest.approx((0.25 + 0 + 1) / 3)
    assert complete["pass@4"] == pytest.approx(2 / 3)
    assert [f"pass@{k}" in complete for k in range(1, 5)] == [True] * 4


def test_episodes_that_kept_aborting_count_as_failures_so_the_point_completes():
    scored = [
        {"instance_id": "a", "sample_index": index, "reward": 1.0} for index in range(3)
    ]
    failure = {
        "instance_id": "a",
        "sample_index": 3,
        "attempts": 4,
        "status": "aborted",
    }

    zeroed = evaluation.scored_as_failures([failure])
    summary = evaluation.summarize(scored + zeroed, n_samples=4, n_tasks=1)

    assert zeroed == [{**failure, "reward": 0.0, "exit_status": "retries_exhausted"}]
    assert summary["complete"] is True
    assert summary["pass@1"] == pytest.approx(0.75)
    assert summary["pass@4"] == pytest.approx(1.0)


def test_points_name_their_checkpoint_by_published_version(tmp_path):
    exp = _recipe("qwen3_6_35b_a3b_hetero_score_centering")
    point = evaluation.EvalPoint(exp.__name__.rsplit(".", 1)[1], "r03", 50, "nvfp4")

    path = evaluation.checkpoint_path(exp, point, source_run_root=tmp_path)

    # Version 50 is the export saved at rollout 49.
    assert export_version(49) == 50
    assert path == tmp_path / "r03" / "hf_checkpoints" / "weight_v000049" / "nvfp4"
    with pytest.raises(FileNotFoundError, match=".complete"):
        evaluation.checkpoint_dir(exp, point, source_run_root=tmp_path)
    path.mkdir(parents=True)
    (path / ".complete").touch()
    assert evaluation.checkpoint_dir(exp, point, source_run_root=tmp_path) == path


def test_base_points_serve_the_static_views_and_are_shared_by_arms():
    exp = _recipe(spec.BASE_EXPERIMENT)
    for view in spec.POOLS:
        point = evaluation.EvalPoint(spec.BASE_EXPERIMENT, None, 0, view)
        assert str(point.relative_dir) == f"base/{view}"
        assert (
            evaluation.checkpoint_path(
                exp, point, source_run_root=evaluation.SOURCE_RUN_PATH
            )
            == (exp.ROLLOUT_WEIGHT_VIEWS[view])
        )
    with pytest.raises(ValueError, match="base model"):
        evaluation.EvalPoint(spec.BASE_EXPERIMENT, "r03", 0, "bf16")
    with pytest.raises(ValueError, match="base model"):
        evaluation.EvalPoint(spec.BASE_EXPERIMENT, None, 50, "bf16")


@pytest.mark.parametrize("arm", HETERO_ARMS)
def test_eval_config_differs_from_training_only_where_an_eval_must(arm):
    train = _recipe(arm).miles

    evaluated = _eval_config(train)

    assert evaluation.config_drift(train, evaluated) == []
    assert evaluated.custom_rollout_request_hook_path is None
    assert evaluated.custom_rollout_request_hook_args is None
    assert evaluated.async_max_concurrent_samples == 384
    # Training's ratios: sessions per session server and threads per agent process.
    assert evaluated.session_server_workers == 19
    assert evaluated.environment["MODAL_SWE_AGENT_PROCESSES"] == "19"
    assert evaluated.environment["MODAL_SWE_SANDBOX_APP"] == spec.SANDBOX_APP
    assert evaluated.use_wandb is False
    assert evaluated.save_debug_rollout_data == "/stitch/eval/dump/{rollout_id}.pt"
    # Every eval dataset generates through the retry hook, at the spec's sampler.
    document = json.loads(
        base64.b64decode(evaluated.eval_config.removeprefix("base64:"))
    )
    (dataset,) = document["eval"]["datasets"]
    assert dataset == {
        **spec.DATASET,
        "custom_generate_function_path": evaluation.EVAL_GENERATE_FUNCTION,
    }
    # The recipe itself is untouched.
    assert train.custom_rollout_request_hook_path is not None


def test_drift_check_names_any_other_change():
    train = _recipe("qwen3_6_35b_a3b_hetero_icepop").miles
    evaluated = _eval_config(train)
    evaluated.rollout_max_response_len = 1024
    evaluated.environment = {**evaluated.environment, "MODAL_SWE_MAX_STEPS": "100"}

    assert evaluation.config_drift(train, evaluated) == [
        "rollout_max_response_len",
        "environment.MODAL_SWE_MAX_STEPS",
    ]


def test_every_hetero_arm_evaluates_with_the_same_harness():
    configs = [_eval_config(_recipe(arm).miles) for arm in HETERO_ARMS]
    for field in HARNESS_FIELDS:
        assert len({repr(getattr(cfg, field, None)) for cfg in configs}) == 1, field
    for key in HARNESS_ENVIRONMENT:
        assert len({cfg.environment.get(key) for cfg in configs}) == 1, key


def test_spec_samples_four_times_at_the_agreed_sampler():
    assert spec.DATASET["n_samples_per_eval_prompt"] == spec.N_SAMPLES == 4
    assert (spec.DATASET["temperature"], spec.DATASET["top_p"]) == (1.0, 1.0)
    assert spec.DATASET["top_k"] == -1


def test_low_precision_views_are_served_by_their_training_pools():
    hetero = _recipe("qwen3_6_35b_a3b_hetero")
    pools = {pool.name: pool for pool in hetero.modal.rollout_pools}

    assert spec.POOLS["fp8"] == pools["ServerH200FP8"]
    assert spec.POOLS["nvfp4"] == pools["ServerB300NVFP4W4A16"]


def test_bf16_reference_is_the_b300_pool_at_bf16_weights_and_kv():
    hetero = _recipe("qwen3_6_35b_a3b_hetero")
    nvfp4 = next(p for p in hetero.modal.rollout_pools if p.gpu == "B300")
    bf16 = spec.POOLS["bf16"]

    assert (bf16.gpu, bf16.weight_view, bf16.gpus_per_engine) == ("B300", "bf16", 1)
    assert bf16.sglang_args["--kv-cache-dtype"] == "bfloat16"
    assert "--quantization" not in bf16.sglang_args
    precision_args = {"--kv-cache-dtype", "--quantization", "--moe-runner-backend"}
    shared = set(nvfp4.sglang_args) - precision_args
    assert {k: bf16.sglang_args[k] for k in shared} == {
        k: nvfp4.sglang_args[k] for k in shared
    }
    assert (bf16.min_containers, bf16.max_containers) == (8, 8)


@pytest.mark.parametrize("version", [0, 50, 500])
@pytest.mark.parametrize("view", ["bf16", "fp8", "nvfp4"])
def test_eval_app_imports_for_each_point(monkeypatch, version, view):
    arm = (
        spec.BASE_EXPERIMENT
        if version == 0
        else "qwen3_6_35b_a3b_hetero_score_centering_mis"
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", arm)
    monkeypatch.setenv("EVAL_CONFIG", "swebench_pro_hetero")
    monkeypatch.setenv("EVAL_RUN", "" if version == 0 else "r03")
    monkeypatch.setenv("EVAL_VERSION", str(version))
    monkeypatch.setenv("EVAL_VIEW", view)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.eval_app", raising=False)

    eval_app = importlib.import_module("cookbook.miles_disagg.eval_app")

    assert eval_app.POOL is spec.POOLS[view]
    assert len(eval_app.APP_NAME) <= 64
    assert str(eval_app.POINT_DIR).startswith(f"/stitch/{evaluation.task_set(spec)}/")
    # Training volumes are mounted read-only; only the eval volume is writable.
    assert (str(evaluation.SOURCE_RUN_PATH) in eval_app.pool_volumes) == (version > 0)
    assert eval_app.pool_volumes["/stitch"] is eval_app.eval_volume


def test_eval_pools_run_with_their_training_pools_modal_options():
    hetero = _recipe("qwen3_6_35b_a3b_hetero")
    for view in ("fp8", "nvfp4"):
        pool = spec.POOLS[view]
        assert server.modal_server_options(hetero.modal, pool)["gpu"] == f"{pool.gpu}:1"
        assert server.modal_server_options(hetero.modal, pool) == (
            server.modal_server_options(
                hetero.modal,
                next(p for p in hetero.modal.rollout_pools if p.name == pool.name),
            )
        )


def test_smoke_subset_is_fixed_and_spread_across_repositories():
    ids = [
        f"instance_{repo}__{repo}-{index:04d}"
        for repo in "abcdefghijk"
        for index in range(60)
    ]

    chosen = evaluation.smoke_tasks(ids, 50)

    assert chosen == evaluation.smoke_tasks(reversed(ids), 50)
    assert len(set(chosen)) == 50
    assert len({task.split("__")[0] for task in chosen}) >= 8
    assert evaluation.smoke_dir((50, 1)) == "smoke-50x1"
    with pytest.raises(ValueError, match="smoke needs"):
        evaluation.smoke_tasks(ids, 0)


def test_launcher_builds_shared_base_points_and_arm_points():
    from argparse import Namespace

    from cookbook.miles_disagg import eval_launch

    args = Namespace(
        spec="swebench_pro_hetero",
        experiment="qwen3_6_35b_a3b_hetero_icepop",
        run="r03",
        versions="100,0,50,50",
        views="bf16,nvfp4",
    )

    points = eval_launch._points(spec, args)

    assert [(p.experiment, p.run_id, p.version, p.view) for p in points] == [
        (spec.BASE_EXPERIMENT, None, 0, "bf16"),
        (spec.BASE_EXPERIMENT, None, 0, "nvfp4"),
        ("qwen3_6_35b_a3b_hetero_icepop", "r03", 50, "bf16"),
        ("qwen3_6_35b_a3b_hetero_icepop", "r03", 50, "nvfp4"),
        ("qwen3_6_35b_a3b_hetero_icepop", "r03", 100, "bf16"),
        ("qwen3_6_35b_a3b_hetero_icepop", "r03", 100, "nvfp4"),
    ]
    assert eval_launch._parse_smoke("50x1") == (50, 1)
    with pytest.raises(SystemExit):
        eval_launch._parse_smoke("50")


def test_launcher_refuses_to_run_without_runc(monkeypatch):
    from cookbook.miles_disagg import eval_launch

    monkeypatch.delenv("MODAL_FUNCTION_RUNTIME", raising=False)
    monkeypatch.setattr(sys, "argv", ["eval_launch", "--spec", "swebench_pro_hetero"])

    with pytest.raises(SystemExit, match="runc"):
        eval_launch.main()


def test_results_split_scored_samples_from_episodes_that_kept_aborting():
    def sample(task, index, status, reward):
        return SimpleNamespace(
            metadata={
                "instance_id": task,
                "eval_sample_index": index,
                "eval_attempts": 1,
                "exit_status": "Submitted",
            },
            status=SimpleNamespace(value=status),
            reward=reward,
            response_length=10,
        )

    samples = [
        sample("a", 0, "completed", 1),
        sample("a", 1, "truncated", 0),
        sample("b", 2, "aborted", None),
        sample("b", 3, "completed", 1.0),
    ]

    records, failures = evaluation.results_from_samples(samples, n_samples=2)

    assert [(r["instance_id"], r["sample_index"], r["reward"]) for r in records] == [
        ("a", 0, 1.0),
        ("a", 1, 0.0),
        ("b", 1, 1.0),
    ]
    assert failures == [
        {
            "instance_id": "b",
            "sample_index": 0,
            "attempts": 1,
            "aborts": [],
            "status": "aborted",
        }
    ]


class _Status(Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"


def _stub_miles(monkeypatch, outcomes):
    """Stand in for the two Miles modules the hook imports; ``outcomes`` are the
    statuses successive attempts of the recipe's generate function return."""
    calls = []

    async def recipe_generate(input):
        calls.append(input.sample)
        status = outcomes[len(calls) - 1]
        metadata = {"instance_id": "a"}
        if status is _Status.ABORTED:
            # What the agent's failure carries for an infrastructure abort.
            metadata |= {
                "exit_status": "agent_error",
                "failure_phase": "model_generation",
                "agent_error": f"BadGatewayError: deadline exceeded ({len(calls)})",
            }
        return SimpleNamespace(
            samples=[SimpleNamespace(metadata=metadata, status=status)]
        )

    compatibility = types.ModuleType("miles.rollout.inference_rollout.compatibility")
    compatibility.load_generate_function = lambda path: recipe_generate
    miles_types = types.ModuleType("miles.utils.types")
    miles_types.Sample = SimpleNamespace(Status=_Status)
    monkeypatch.setitem(sys.modules, compatibility.__name__, compatibility)
    monkeypatch.setitem(sys.modules, miles_types.__name__, miles_types)
    return calls


def _hook_input(retries):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Input:
        sample: object
        args: object

    return Input(
        sample=SimpleNamespace(index=13, metadata={"instance_id": "a"}),
        args=SimpleNamespace(
            custom_generate_function_path="recipe.generate", eval_infra_retries=retries
        ),
    )


def test_retry_hook_reruns_aborted_episodes_from_a_fresh_copy(monkeypatch):
    from cookbook.miles_disagg import eval_hooks

    calls = _stub_miles(
        monkeypatch, [_Status.ABORTED, _Status.ABORTED, _Status.COMPLETED]
    )
    hook_input = _hook_input(retries=3)

    output = asyncio.run(eval_hooks.generate(hook_input))

    (episode,) = output.samples
    assert episode.status is _Status.COMPLETED
    assert episode.metadata["eval_attempts"] == 3
    assert episode.metadata["eval_sample_index"] == 13
    assert len(calls) == 3
    assert all(call is not hook_input.sample for call in calls)
    # The scored sample keeps why each earlier attempt aborted.
    assert episode.metadata["eval_aborts"] == [
        {
            "attempt": attempt,
            "exit_status": "agent_error",
            "failure_phase": "model_generation",
            "agent_error": f"BadGatewayError: deadline exceeded ({attempt})",
            "root_error": None,
        }
        for attempt in (1, 2)
    ]


def test_retry_hook_returns_the_abort_once_retries_run_out(monkeypatch):
    from cookbook.miles_disagg import eval_hooks

    calls = _stub_miles(monkeypatch, [_Status.ABORTED] * 3)

    output = asyncio.run(eval_hooks.generate(_hook_input(retries=1)))

    assert output.samples[0].status is _Status.ABORTED
    assert output.samples[0].metadata["eval_attempts"] == 2
    assert [a["attempt"] for a in output.samples[0].metadata["eval_aborts"]] == [1, 2]
    assert len(calls) == 2


def test_failed_samples_keep_their_abort_reasons():
    aborts = [{"attempt": 1, "exit_status": "agent_error", "agent_error": "x"}]
    sample = SimpleNamespace(
        metadata={
            "instance_id": "t1",
            "eval_sample_index": 2,
            "eval_attempts": 1,
            "eval_aborts": aborts,
        },
        status="aborted",
        reward=None,
        response_length=0,
    )

    records, (failure,) = evaluation.results_from_samples([sample], n_samples=4)
    (scored,) = evaluation.scored_as_failures([failure])

    assert records == []
    assert failure["aborts"] == aborts
    assert scored["aborts"] == aborts and scored["exit_status"] == "retries_exhausted"


def test_eval_engines_resizes_the_pool_fleet(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_hetero_icepop")
    monkeypatch.setenv("EVAL_CONFIG", "swebench_pro_hetero")
    monkeypatch.setenv("EVAL_RUN", "r03")
    monkeypatch.setenv("EVAL_VERSION", "50")
    monkeypatch.setenv("EVAL_VIEW", "fp8")
    monkeypatch.setenv("EVAL_ENGINES", "32")
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.eval_app", raising=False)

    eval_app = importlib.import_module("cookbook.miles_disagg.eval_app")

    assert (eval_app.POOL.min_containers, eval_app.POOL.max_containers) == (32, 32)
    assert eval_app.POOL.sglang_args == spec.POOLS["fp8"].sglang_args
    assert eval_app.POINT_ENVIRONMENT["EVAL_ENGINES"] == "32"


def test_eval_driver_keeps_every_workers_log_lines(monkeypatch):
    from cookbook.miles_disagg import trainer_image

    built = []
    real = trainer_image.build_trainer_image
    monkeypatch.setattr(
        trainer_image,
        "build_trainer_image",
        lambda **kwargs: built.append(kwargs) or real(**kwargs),
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_hetero_icepop")
    monkeypatch.setenv("EVAL_CONFIG", "swebench_pro_hetero")
    monkeypatch.setenv("EVAL_RUN", "r03")
    monkeypatch.setenv("EVAL_VERSION", "50")
    monkeypatch.setenv("EVAL_VIEW", "bf16")
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.eval_app", raising=False)

    importlib.import_module("cookbook.miles_disagg.eval_app")

    # Ray reads it at import, so it must be in the driver's environment from the start.
    (driver,) = built
    assert driver["extra_env"]["RAY_DEDUP_LOGS"] == "0"
    assert driver["extra_env"]["EVAL_VERSION"] == "50"


def test_results_live_under_the_prepared_task_set():
    point = evaluation.EvalPoint("qwen3_6_35b_a3b_hetero_icepop", "r03", 50, "fp8")
    tasks = spec.DATASET_PATH.name

    assert evaluation.task_set(spec) == tasks
    assert str(evaluation.results_path(spec, point)) == (
        f"{tasks}/qwen3_6_35b_a3b_hetero_icepop/r03/v000050/fp8"
    )


def test_eval_set_is_v2_less_the_tasks_that_do_not_run_reliably():
    assert spec.DATASET_PATH.name == "swebench-pro-scale-v2"
    assert spec.TASKS == 642 - len(spec.EXCLUDED_TASKS) == 641
    assert set(spec.EXCLUDED_TASKS.values()) == {"unreliable"}


def test_v2_patches_are_graded_in_a_fresh_sandbox_as_v2_grades_them():
    assert spec.GRADE_IN_FRESH_SANDBOX is True
    cfg = evaluation.eval_miles_config(
        importlib.import_module(
            "cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_grpo"
        ).miles,
        dataset=spec.DATASET,
        tasks_dir=spec.TASKS_DIR,
        sandbox_app=spec.SANDBOX_APP,
        concurrency=256,
        dump_template="/tmp/{rollout_id}.pt",
        fresh_sandbox_grading=spec.GRADE_IN_FRESH_SANDBOX,
    )

    assert cfg.environment["MODAL_SWE_GRADE_IN_FRESH_SANDBOX"] == "1"
    assert cfg.environment["MODAL_SWE_TASKS_DIR"] == "/data/swebench-pro-scale-v2/tasks"


@pytest.mark.parametrize("arm", HETERO_ARMS)
def test_a_turn_past_the_time_limit_fails_and_other_failures_are_resent(arm):
    train = _recipe(arm).miles
    cfg = evaluation.eval_miles_config(
        train,
        dataset=spec.DATASET,
        tasks_dir=spec.TASKS_DIR,
        sandbox_app=spec.SANDBOX_APP,
        concurrency=384,
        dump_template="/tmp/{rollout_id}.pt",
        request_attempts=spec.REQUEST_ATTEMPTS,
        turn_time_limit=spec.TURN_TIME_LIMIT_SECONDS,
    )

    # Training sends each turn once; the eval resends a failed one.
    assert train.environment["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "1"
    assert cfg.environment["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "3"
    # The agent learns the limit, so a turn the session server gives up on fails its
    # episode instead of being resent.
    assert cfg.environment["MODAL_SWE_TURN_TIME_LIMIT_SECONDS"] == "300"
    assert "MODAL_SWE_TURN_TIME_LIMIT_SECONDS" not in train.environment
    assert evaluation.config_drift(train, cfg) == []
    # The session server gives up at the limit, before the agent's own request timeout,
    # and before the GPU pools' serving path drops a silent response (~343 s observed).
    assert evaluation.request_deadline(cfg, spec.TURN_TIME_LIMIT_SECONDS) == 300
    assert 300 < float(cfg.environment["MODAL_SWE_MODEL_REQUEST_TIMEOUT"])
    assert spec.TURN_TIME_LIMIT_SECONDS < 343
    assert not hasattr(spec, "REQUEST_DEADLINE_SECONDS")


def test_the_turn_time_limit_must_be_positive():
    with pytest.raises(ValueError, match="turn_time_limit"):
        evaluation.eval_miles_config(
            _recipe("qwen3_6_35b_a3b_hetero_grpo").miles,
            dataset=spec.DATASET,
            tasks_dir=spec.TASKS_DIR,
            sandbox_app=spec.SANDBOX_APP,
            concurrency=384,
            dump_template="/tmp/{rollout_id}.pt",
            turn_time_limit=0,
        )


def test_request_deadline_must_fall_before_the_agent_timeout():
    cfg = _eval_config(_recipe("qwen3_6_35b_a3b_hetero_grpo").miles)
    agent_timeout = float(cfg.environment["MODAL_SWE_MODEL_REQUEST_TIMEOUT"])

    for seconds in (0, agent_timeout, agent_timeout + 1):
        with pytest.raises(ValueError, match="request deadline"):
            evaluation.request_deadline(cfg, seconds)
    with pytest.raises(ValueError, match="request_attempts"):
        evaluation.eval_miles_config(
            _recipe("qwen3_6_35b_a3b_hetero_grpo").miles,
            dataset=spec.DATASET,
            tasks_dir=spec.TASKS_DIR,
            sandbox_app=spec.SANDBOX_APP,
            concurrency=384,
            dump_template="/tmp/{rollout_id}.pt",
            request_attempts=0,
        )


def test_driver_writes_the_eval_set_less_excluded_tasks(tmp_path):
    from cookbook.miles_disagg import eval_driver

    source = tmp_path / "test.jsonl"
    ids = [f"instance_{repo}__{repo}-{i}" for repo in "abc" for i in range(10)]
    source.write_text(
        "".join(
            json.dumps({"prompt": i, "metadata": {"instance_id": i}}) + "\n"
            for i in ids
        )
    )
    excluded = {ids[0]: "needs_network", ids[5]: "flaky_reference"}

    full = eval_driver._write_eval_set(
        source, tmp_path / "full.jsonl", excluded=excluded, count=28, smoke=False
    )
    smoke = eval_driver._write_eval_set(
        source, tmp_path / "smoke.jsonl", excluded=excluded, count=5, smoke=True
    )

    kept = [
        json.loads(line)["metadata"]["instance_id"]
        for line in full.read_text().splitlines()
    ]
    assert kept == [i for i in ids if i not in excluded]
    chosen = [
        json.loads(line)["metadata"]["instance_id"]
        for line in smoke.read_text().splitlines()
    ]
    assert len(chosen) == 5 and not set(chosen) & set(excluded)
    with pytest.raises(RuntimeError, match="expected 29"):
        eval_driver._write_eval_set(
            source, tmp_path / "bad.jsonl", excluded=excluded, count=29, smoke=False
        )


def test_patches_are_kept_apart_from_scores_for_regrading():
    records = [
        {
            "instance_id": "t1",
            "sample_index": 0,
            "reward": 1.0,
            "policy_patch_b64": "cA==",
        },
        {
            "instance_id": "t1",
            "sample_index": 1,
            "reward": 0.0,
            "policy_patch_b64": None,
        },
    ]

    scores, patches = evaluation.split_patches(records)

    assert all("policy_patch_b64" not in row for row in scores)
    assert [row["reward"] for row in scores] == [1.0, 0.0]
    assert patches == [
        {"instance_id": "t1", "sample_index": 0, "reward": 1.0, "patch_b64": "cA=="}
    ]
    assert records[0]["policy_patch_b64"] == "cA=="


def test_sample_records_keep_the_verifier_output_and_patch():
    sample = SimpleNamespace(
        metadata={
            "instance_id": "t1",
            "eval_sample_index": 9,
            "eval_attempts": 1,
            "exit_status": "Submitted",
            "agent_metrics": {},
            "verifier_output_tail": "RESULT: PASSED",
            "policy_patch_b64": "cA==",
        },
        status="completed",
        reward=1.0,
        response_length=10,
    )

    (record,), failures = evaluation.results_from_samples([sample], n_samples=8)

    assert failures == []
    assert record["sample_index"] == 1
    assert record["verifier_output_tail"] == "RESULT: PASSED"
    assert record["policy_patch_b64"] == "cA=="


def test_app_names_fit_modal_and_stay_unchanged_when_they_already_fit():
    point = evaluation.EvalPoint("recipe", "r01", 20, "bf16")
    late = evaluation.EvalPoint("recipe", "r01", 480, "bf16")
    fits = "stitch-qwen36-hetero-score-centering-mis-top-p"  # 64 characters at step 20
    long = "stitch-qwen36-b200-bf16-score-centering-mis-top-p"

    assert evaluation.app_name("spec", fits, point) == f"{fits}-eval-r01-v20-bf16"
    assert (
        evaluation.app_name("spec", long, point)
        == "stitch-qwen36-b200-bf16-sc-mis-top-p-eval-r01-v20-bf16"
    )
    assert (
        evaluation.app_name("spec", fits, late)
        == "stitch-qwen36-hetero-sc-mis-top-p-eval-r01-v480-bf16"
    )
    with pytest.raises(ValueError, match="64 characters"):
        evaluation.app_name("spec", "x" * 60, point)


@pytest.mark.parametrize(
    "arm",
    [
        "qwen3_6_35b_a3b_hetero_score_centering_mis_top_p",
        "qwen3_6_35b_a3b_b200_bf16_score_centering_mis_top_p",
    ],
)
def test_an_eval_pins_the_training_only_grading_rules_off(arm):
    """A SWE-bench Pro verifier applies the patch itself; a recipe that applies it in
    training must not make an eval apply it twice, and eval scores stay as graded."""
    train = _recipe(arm).miles
    assert train.environment["MODAL_SWE_FRESH_GRADE_APPLY_PATCH"] == "1"

    graded = evaluation.eval_miles_config(
        train,
        dataset=spec.DATASET,
        tasks_dir=spec.TASKS_DIR,
        sandbox_app=spec.SANDBOX_APP,
        concurrency=256,
        dump_template="/tmp/{rollout_id}.pt",
        fresh_sandbox_grading=True,
    )
    in_place = _eval_config(train)

    for key, value in evaluation.TRAINING_ONLY_ENVIRONMENT.items():
        assert graded.environment[key] == value == "0"
    assert graded.environment["MODAL_SWE_GRADE_IN_FRESH_SANDBOX"] == "1"
    assert in_place.environment["MODAL_SWE_GRADE_IN_FRESH_SANDBOX"] == "0"
