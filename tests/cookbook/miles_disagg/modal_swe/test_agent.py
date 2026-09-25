import asyncio
import subprocess
import sys
import time
from importlib import import_module

import pytest

try:
    import_module("pytest_asyncio")
    from jinja2 import Template
    from miles.utils.function_registry import load_function
    from miles.utils.types import Sample

    from cookbook.miles_disagg.modal_swe import agent as agent_function_module
    from cookbook.miles_disagg.modal_swe.agent import (
        _OBSERVATION_TEMPLATE,
        _AgentWorker,
        _attach_client_model_timings,
        _environment_metrics,
        _EnvironmentSnapshot,
        _EpisodeAbortState,
        _exception_metadata,
        _failure,
        _instrument_model_requests,
        _is_context_limit_error,
        _is_infrastructure_error,
        _is_sandbox_not_found_error,
        _is_truncated_generation_error,
        _parse_reward,
        _prepare_environment,
        _RayAgentWorkerPool,
        _sandbox_boot_semaphore,
        _task_cwd,
        pick_latest_leaf,
        postprocess_samples,
        reward_func,
    )
    from cookbook.miles_disagg.modal_swe.metrics import log_rollout_data
    from cookbook.miles_disagg.modal_swe.sandbox import (
        _BOUNDED_COMMAND_RUNNER,
        ModalSWEEnvironment,
        SandboxExecResult,
        SandboxTransportError,
        _parse_bounded_command_response,
        sandbox_settings,
    )
    from cookbook.miles_disagg.swebench_config import arguments
except ModuleNotFoundError as error:
    if error.name not in {"jinja2", "miles", "pytest_asyncio"}:
        raise
    pytest.skip(
        "Modal SWE adapter tests require the Miles trainer environment",
        allow_module_level=True,
    )


def test_swebench_callbacks_resolve_from_the_cookbook() -> None:
    config = arguments()

    for key in (
        "custom_rollout_log_function_path",
        "custom_agent_function_path",
        "custom_rm_path",
        "session_sample_picker_path",
        "session_sample_postprocessor_path",
    ):
        assert callable(load_function(config[key]))


def _run_bounded(
    command: str,
    *,
    timeout: float = 30,
    output_hard_limit: int = 16 * 1024 * 1024,
):
    started = time.perf_counter()
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            _BOUNDED_COMMAND_RUNNER,
            "5000",
            "5000",
            str(timeout),
            str(output_hard_limit),
        ],
        input=command.encode(),
        capture_output=True,
        check=True,
    )
    return _parse_bounded_command_response(
        stdout=process.stdout,
        stderr=process.stderr,
        fallback_return_code=process.returncode,
        client_seconds=time.perf_counter() - started,
    )


def test_sandbox_settings_rejects_invalid_modal_app_name(monkeypatch):
    invalid_name = "x" * 65
    monkeypatch.setenv("MODAL_SWE_SANDBOX_APP", invalid_name)

    with pytest.raises(ValueError, match=r"1-64 characters.*65 chars"):
        sandbox_settings()


def test_sandbox_settings_accepts_64_character_modal_app_name(monkeypatch):
    valid_name = "x" * 64
    monkeypatch.setenv("MODAL_SWE_SANDBOX_APP", valid_name)

    assert sandbox_settings()["app_name"] == valid_name


def test_task_cwd_defaults_to_swegym_and_accepts_absolute_override():
    assert _task_cwd({}) == "/testbed"
    assert _task_cwd({"sandbox_cwd": "/app"}) == "/app"


@pytest.mark.parametrize("cwd", ["app", "", "/app\x00other"])
def test_task_cwd_rejects_invalid_override(cwd):
    with pytest.raises(ValueError, match="absolute path"):
        _task_cwd({"sandbox_cwd": cwd})


def test_optional_task_setup_is_hidden_after_execution(tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "setup.sh").write_text("printf prepared")

    class Environment:
        cwd = "/app"

        def __init__(self):
            self.uploads = []
            self.commands = []
            self.ready = False

        def upload_tree(self, source, destination):
            self.uploads.append((source, destination))

        def exec(self, command, *, cwd, timeout):
            self.commands.append((command, cwd, timeout))
            return 0, "prepared"

        def mark_ready(self):
            self.ready = True

    env = Environment()
    _prepare_environment(env, tmp_path)

    assert env.uploads == [(environment, "/tmp/miles-task-environment")]
    command, cwd, timeout = env.commands[0]
    assert "setup.sh" in command
    assert "rm -rf /tmp/miles-task-environment" in command
    assert cwd == "/app"
    assert timeout == 240
    assert env.ready is True


def test_task_setup_timeout_must_fit_modal_readiness_window(monkeypatch, tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "setup.sh").write_text("printf prepared")
    monkeypatch.setenv("MODAL_SWE_SETUP_TIMEOUT", "300")

    class Environment:
        def upload_tree(self, _source, _destination):
            pass

    with pytest.raises(ValueError, match="between 1 and 240"):
        _prepare_environment(Environment(), tmp_path)


def test_environment_without_setup_is_marked_ready(tmp_path):
    class Environment:
        def __init__(self):
            self.ready = False

        def mark_ready(self):
            self.ready = True

    env = Environment()
    _prepare_environment(env, tmp_path)

    assert env.ready is True


def test_environment_snapshot_excludes_setup_from_agent_metrics():
    setup = _EnvironmentSnapshot(
        command_count=1,
        exec_time=2.0,
        exec_remote_time=1.5,
        upload_time=0.5,
        upload_bytes=100,
        command_timeout_count=0,
        output_bytes=20,
        transferred_bytes=30,
        output_truncated_count=0,
        output_hard_limit_count=0,
    )
    after_agent = _EnvironmentSnapshot(
        command_count=3,
        exec_time=7.0,
        exec_remote_time=5.5,
        upload_time=0.5,
        upload_bytes=100,
        command_timeout_count=1,
        output_bytes=220,
        transferred_bytes=330,
        output_truncated_count=1,
        output_hard_limit_count=0,
    )

    agent = after_agent.since(setup)

    assert agent.command_count == 2
    assert agent.exec_time == 5.0
    assert agent.exec_remote_time == 4.0
    assert agent.upload_time == 0.0
    assert agent.output_bytes == 200
    assert agent.transferred_bytes == 300
    assert agent.command_timeout_count == 1
    assert agent.output_truncated_count == 1


def test_environment_metrics_do_not_count_setup_twice():
    class Environment:
        boot_time = 10.0
        schedule_time = 2.0
        readiness_time = 8.0
        exec_time = 5.0
        exec_remote_time = 4.0
        exec_durations = [3.0]
        exec_remote_durations = [2.5]
        exec_transport_durations = [0.5]
        command_input_sizes = [10]
        upload_time = 3.0
        upload_bytes = 100
        command_output_bytes = 20
        command_transferred_bytes = 30

    setup = _EnvironmentSnapshot(
        command_count=1,
        exec_time=3.0,
        exec_remote_time=2.5,
        upload_time=2.0,
        upload_bytes=100,
        command_timeout_count=0,
        output_bytes=20,
        transferred_bytes=30,
        output_truncated_count=0,
        output_hard_limit_count=0,
    )

    metrics = _environment_metrics(
        Environment(),
        agent_queue_time=0.0,
        total_time=20.0,
        agent_start_snapshot=setup,
        agent_snapshot=setup,
    )

    assert metrics["sandbox_boot_time"] == 10.0
    assert metrics["sandbox_setup_time"] == 5.0
    assert metrics["total_tool_time"] == 13.0


def test_invalid_command_wrapper_response_is_infrastructure_failure():
    with pytest.raises(SandboxTransportError, match="invalid response"):
        _parse_bounded_command_response(
            stdout=b"not-json",
            stderr=b"wrapper traceback",
            fallback_return_code=1,
            client_seconds=0.1,
        )


def test_policy_failure_keeps_trajectory_trainable_with_zero_reward():
    result = _failure("command_timeout", infrastructure=False)

    assert "_miles_abort" not in result
    assert result["reward"] == 0.0
    assert result["agent_metrics"]["policy_failure"] == 1
    assert result["agent_metrics"]["infra_error"] == 0


def test_infrastructure_failure_requests_scheduler_abort():
    result = _failure("sandbox_infra_error")

    assert result["_miles_abort"] is True
    assert result["agent_metrics"]["infra_error"] == 1


def test_v2_postprocessor_marks_infrastructure_failure_aborted():
    sample = Sample(
        tokens=[1],
        response_length=1,
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        metadata={"leaf": {"node_id": 0, "path_node_ids": [0]}},
    )
    metadata = {
        "tree": {"nodes": [{"id": 0, "completion_span": [0, 1]}]},
        "agent": _failure("sandbox_infra_error"),
    }

    [processed] = postprocess_samples([sample], metadata)

    assert processed.status == Sample.Status.ABORTED
    assert processed.reward is None
    assert processed.metadata["exit_status"] == "sandbox_infra_error"
    assert "_miles_abort" not in processed.metadata


def test_v2_picker_returns_only_latest_committed_leaf():
    older_deep_branch = Sample(
        metadata={"leaf": {"node_id": 7, "path_node_ids": [0, 3, 7]}}
    )
    latest_shallow_branch = Sample(
        metadata={"leaf": {"node_id": 9, "path_node_ids": [0, 9]}}
    )

    assert pick_latest_leaf([older_deep_branch, latest_shallow_branch], {}) == [
        latest_shallow_branch
    ]


def test_v2_picker_accepts_empty_session_and_rejects_missing_commit_order():
    assert pick_latest_leaf([], {}) == []
    with pytest.raises(ValueError, match="leaf.node_id"):
        pick_latest_leaf([Sample(metadata={})], {})


def test_v2_postprocessor_keeps_policy_failure_trainable():
    sample = Sample(
        tokens=[1],
        response_length=1,
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        metadata={"leaf": {"node_id": 0, "path_node_ids": [0]}},
    )
    metadata = {
        "tree": {"nodes": [{"id": 0, "completion_span": [0, 1]}]},
        "agent": _failure("command_timeout", infrastructure=False),
    }

    [processed] = postprocess_samples([sample], metadata)

    assert processed.status == Sample.Status.COMPLETED
    assert processed.reward == 0.0


def test_infrastructure_failure_preserves_root_cause_metadata():
    service_error_type = type(
        "ServiceError", (Exception,), {"__module__": "modal.exception"}
    )
    root = service_error_type("temporarily unavailable")
    wrapped = RuntimeError("worker failed")
    wrapped.__cause__ = root

    metadata = _exception_metadata(wrapped)
    assert metadata["error_type"] == "builtins.RuntimeError"
    assert metadata["root_error_type"] == "modal.exception.ServiceError"
    assert metadata["error_chain"] == [
        "builtins.RuntimeError",
        "modal.exception.ServiceError",
    ]


def test_sandbox_boot_concurrency_is_bounded_per_controller(monkeypatch):
    _sandbox_boot_semaphore.cache_clear()
    monkeypatch.setenv("MODAL_SWE_SANDBOX_BOOT_CONCURRENCY_PER_PROCESS", "1")
    semaphore = _sandbox_boot_semaphore()
    assert semaphore.acquire(blocking=False)
    assert not semaphore.acquire(blocking=False)
    semaphore.release()
    _sandbox_boot_semaphore.cache_clear()


@pytest.mark.asyncio
async def test_reward_hook_returns_verifier_reward():
    sample = Sample(metadata={"reward": 1.0})

    assert await reward_func(None, sample) == 1.0
    assert await reward_func(None, [sample, Sample(metadata={"reward": 0})]) == [
        1.0,
        0.0,
    ]


@pytest.mark.asyncio
async def test_reward_hook_rejects_missing_or_invalid_verifier_reward():
    with pytest.raises(ValueError, match="no verifier reward"):
        await reward_func(None, Sample(metadata={}))
    with pytest.raises(TypeError, match="must be numeric"):
        await reward_func(None, Sample(metadata={"reward": "1"}))
    with pytest.raises(ValueError, match="must be finite"):
        await reward_func(None, Sample(metadata={"reward": float("nan")}))
    with pytest.raises(ValueError, match="must be binary"):
        await reward_func(None, Sample(metadata={"reward": 0.5}))


def test_client_model_request_instrumentation_records_exact_attempts():
    class Model:
        def _query(self, value):
            return value * 2

    model = Model()
    durations = []
    phases = []
    _instrument_model_requests(model, durations, phases.append)

    assert model._query(3) == 6
    assert len(durations) == 1
    assert durations[0] >= 0
    assert phases == ["model_generation", "interaction"]

    metrics = _attach_client_model_timings({}, durations)
    assert metrics["client_model_request_durations_seconds"] == durations
    assert metrics["model_request_count"] == 1
    assert metrics["model_request_time"] == pytest.approx(sum(durations))


def test_client_model_timings_split_generation_from_interaction():
    metrics = _attach_client_model_timings(
        {"total_time": 10.0, "total_tool_time": 3.0},
        [2.0, 3.0],
    )

    assert metrics["model_request_count"] == 2
    assert metrics["model_request_time"] == 5.0
    assert metrics["interaction_time"] == 5.0
    assert metrics["interaction_sandbox_time"] == 3.0
    assert metrics["interaction_unattributed_time"] == 2.0
    assert metrics["generation_time_ratio"] == 0.5
    assert metrics["interaction_time_ratio"] == 0.5
    assert metrics["generation_bound"] == 0


def test_client_model_timings_classify_pre_generation_failure():
    metrics = _attach_client_model_timings({"total_time": 10.0}, [])

    assert metrics["model_request_count"] == 0
    assert metrics["model_request_time"] == 0
    assert metrics["interaction_time"] == 10.0
    assert metrics["interaction_sandbox_time"] == 0.0
    assert metrics["interaction_unattributed_time"] == 10.0
    assert metrics["generation_time_ratio"] == 0.0
    assert metrics["interaction_time_ratio"] == 1.0


def test_rollout_metrics_aggregate_adapter_owned_timings():
    samples = [
        Sample(
            response_length=40,
            loss_mask=[1] * 30 + [0] * 10,
            metadata={
                "exit_status": "Submitted",
                "agent_metrics": {
                    "total_time": 10.0,
                    "total_tool_time": 4.0,
                    "agent_tool_output_hard_limit_count": 1,
                    "client_model_request_durations_seconds": [2.0, 3.0],
                },
                "session_collect/total_seconds": 0.5,
                "lifecycle": [
                    {"t0": 1.0, "t1": 2.5, "turn": 1},
                    {"t0": 3.0, "t1": 5.5, "turn": 2},
                ],
            },
        ),
        Sample(
            response_length=20,
            loss_mask=[1] * 10 + [0] * 10,
            metadata={
                "exit_status": "LimitsExceeded",
                "agent_metrics": {
                    "total_time": 20.0,
                    "total_tool_time": 8.0,
                    "agent_tool_output_hard_limit_count": 0,
                    "context_limit_exceeded": 1,
                    "client_model_request_durations_seconds": [4.0],
                },
                "session_collect/total_seconds": 1.0,
                "lifecycle": {"t0": 10.0, "t1": 13.5, "turn": 1},
            },
        ),
    ]
    metrics = {}

    assert log_rollout_data(0, None, samples, metrics, 0.0) is False

    assert metrics["rollout_agent/total_time_mean"] == 15
    assert metrics["rollout_session/total_seconds_mean"] == 0.75
    assert metrics["rollout_model/request_count"] == 3
    assert metrics["rollout_model/trainable_completion_tokens"] == 40
    assert metrics[
        "rollout_model/trainable_tokens_per_backend_request_second"
    ] == pytest.approx(40 / 7.5)
    assert metrics["rollout_model/client_minus_backend_request_count"] == 0
    assert metrics[
        "rollout_model/client_minus_backend_seconds_signed"
    ] == pytest.approx(1.5)
    assert metrics["rollout_agent/agent_tool_output_hard_limit_count_mean"] == 0.5
    assert metrics["rollout_agent/context_limit_exit_ratio"] == 0.5
    assert (
        "client_model_request_durations_seconds"
        not in samples[0].metadata["agent_metrics"]
    )


def test_rollout_metrics_include_masked_infrastructure_attempts():
    failed = Sample(
        response_length=8,
        loss_mask=[0] * 8,
        remove_sample=True,
        metadata={
            "exit_status": "sandbox_infra_error",
            "agent_metrics": {
                "total_time": 12.0,
                "infra_error": 1,
            },
            "lifecycle": {"t0": 1.0, "t1": 4.0, "turn": 1},
        },
    )
    metrics = {}

    assert log_rollout_data(0, None, [failed], metrics, 0.0) is False

    assert metrics["rollout_agent/total_time_mean"] == 12.0
    assert metrics["rollout_agent/exit_status/sandbox_infra_error_ratio"] == 1.0
    assert metrics["rollout_model/request_count"] == 1


@pytest.mark.parametrize(
    "exit_status",
    ["TimeExceeded", "RepeatedFormatError", "FormatError", "UserInterruption"],
)
def test_rollout_metrics_preserve_mini_swe_terminal_statuses(exit_status):
    metrics = {}

    assert (
        log_rollout_data(
            0,
            None,
            [Sample(metadata={"exit_status": exit_status})],
            metrics,
            0.0,
        )
        is False
    )

    assert metrics[f"rollout_agent/exit_status/{exit_status}_ratio"] == 1.0
    assert "rollout_agent/exit_status/other_ratio" not in metrics


def test_reward_parser_rejects_ambiguous_mapping():
    output = "\n".join(
        [
            "__MILES_SWEGYM_REWARD_START__",
            '{"passed": 1, "failed": 0}',
            "__MILES_SWEGYM_REWARD_END__",
        ]
    )

    assert _parse_reward(output) is None


@pytest.mark.parametrize("reward", ["nan", "inf", "true", "0.5"])
def test_reward_parser_rejects_non_binary_values(reward):
    output = "\n".join(
        [
            "__MILES_SWEGYM_REWARD_START__",
            reward,
            "__MILES_SWEGYM_REWARD_END__",
        ]
    )

    assert _parse_reward(output) is None


def test_bounded_runner_preserves_small_stdout_stderr_and_return_code():
    result = _run_bounded("printf stdout; printf stderr >&2; exit 7")

    assert result.return_code == 7
    assert result.output == "stdoutstderr"
    assert result.output_total_bytes == 12
    assert not result.output_truncated
    assert result.remote_seconds > 0


def test_bounded_runner_caps_output_before_transport():
    result = _run_bounded(
        "python -c \"import sys; sys.stdout.write('a' * 600000); sys.stderr.write('b' * 400000)\""
    )

    assert result.return_code == 0
    assert result.output == ""
    assert result.output_total_bytes == 1_000_000
    assert result.output_truncated
    assert result.output_head == "a" * 5000
    assert result.output_tail == "b" * 5000
    assert result.transferred_bytes < 20_000
    assert not result.output_limited


def test_bounded_runner_stops_commands_that_produce_runaway_output():
    result = _run_bounded(
        "python -c \"import sys; chunk=b'x'*65536; [sys.stdout.buffer.write(chunk) for _ in range(1000)]\"",
        output_hard_limit=200_000,
    )

    assert result.return_code == 125
    assert result.output_truncated
    assert result.output_limited
    assert result.output_total_bytes < 1_000_000
    assert result.transferred_bytes < 20_000
    assert result.remote_seconds < 5


def test_output_limit_is_explicit_agent_feedback(monkeypatch):
    env = ModalSWEEnvironment.__new__(ModalSWEEnvironment)
    env.cwd = "/testbed"
    result = SandboxExecResult(
        return_code=125,
        output="",
        output_head="first lines\n",
        output_tail="\nlast lines",
        output_total_bytes=200_000,
        output_truncated=True,
        remote_seconds=0.1,
        client_seconds=0.2,
        transferred_bytes=10_000,
        output_limited=True,
    )
    monkeypatch.setattr(env, "exec_detailed", lambda *_args, **_kwargs: result)

    observation = env.execute({"command": "yes"})

    assert observation["returncode"] == 125
    assert observation["output_limited"]
    rendered = Template(_OBSERVATION_TEMPLATE).render(output=observation)
    assert "terminated after producing too much output" in rendered
    assert "first lines" in rendered
    assert "last lines" in rendered


def test_bounded_runner_enforces_deadline_and_keeps_partial_output():
    result = _run_bounded(
        "printf started; sleep 30",
        timeout=0.2,
    )

    assert result.return_code == 124
    assert result.timed_out
    assert result.output == "started"
    assert result.remote_seconds < 5


def test_bounded_runner_streams_commands_larger_than_modal_cmd_limit():
    command = "#" + ("x" * 200_000) + "\nprintf streamed"
    result = _run_bounded(command)

    assert result.return_code == 0
    assert result.output == "streamed"


class _FakeWorker:
    pass


def test_worker_pool_round_robins_ties_and_prefers_least_loaded():
    workers = [_FakeWorker() for _ in range(3)]
    pool = _RayAgentWorkerPool(workers)

    acquired = [pool._acquire()[0] for _ in range(6)]
    assert acquired == [0, 1, 2, 0, 1, 2]
    assert pool.in_flight == [2, 2, 2]

    pool._release(1)
    index, worker = pool._acquire()
    assert index == 1
    assert worker is workers[1]


def test_empty_worker_pool_is_rejected():
    pool = _RayAgentWorkerPool([])
    with pytest.raises(ValueError):
        pool._acquire()


@pytest.mark.asyncio
async def test_cancelled_dispatch_retains_capacity_until_remote_episode_finishes():
    finish = asyncio.Event()

    class RemoteMethod:
        def remote(self, _payload):
            async def run():
                await finish.wait()
                return {"reward": 0}

            return run()

    worker = _FakeWorker()
    worker.run_episode = RemoteMethod()
    pool = _RayAgentWorkerPool([worker])

    dispatch = asyncio.create_task(pool.run_episode({}))
    await asyncio.sleep(0)
    assert pool.in_flight == [1]

    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    assert pool.in_flight == [1]

    finish.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert pool.in_flight == [0]


@pytest.mark.asyncio
async def test_worker_pool_queues_excess_episodes_before_ray_submission():
    finishes = [asyncio.Event(), asyncio.Event()]
    submitted = 0

    class RemoteMethod:
        def remote(self, _payload):
            nonlocal submitted
            index = submitted
            submitted += 1

            async def run():
                await finishes[index].wait()
                return {"reward": 0}

            return run()

    worker = _FakeWorker()
    worker.run_episode = RemoteMethod()
    pool = _RayAgentWorkerPool([worker], per_worker_capacity=1)

    first = asyncio.create_task(pool.run_episode({}))
    second = asyncio.create_task(pool.run_episode({}))
    await asyncio.sleep(0)
    assert submitted == 1
    assert pool.in_flight == [1]

    finishes[0].set()
    await first
    await asyncio.sleep(0)
    assert submitted == 2
    assert pool.in_flight == [1]

    finishes[1].set()
    await second
    assert pool.in_flight == [0]


@pytest.mark.asyncio
async def test_worker_pool_abort_stops_active_and_rejects_queued_episodes():
    finish = asyncio.Event()
    submitted = []
    aborted = []

    class RemoteMethod:
        def remote(self, payload):
            submitted.append(payload["_abort_generation"])

            async def run():
                await finish.wait()
                return {"reward": 0}

            return run()

    class AbortMethod:
        def remote(self, generation):
            async def run():
                aborted.append(generation)
                finish.set()

            return run()

    worker = _FakeWorker()
    worker.run_episode = RemoteMethod()
    worker.abort_episodes = AbortMethod()
    pool = _RayAgentWorkerPool([worker], per_worker_capacity=1)

    first = asyncio.create_task(pool.run_episode({}))
    second = asyncio.create_task(pool.run_episode({}))
    await asyncio.sleep(0)

    await pool.abort()
    first_result, second_result = await asyncio.gather(first, second)

    assert submitted == [0]
    assert aborted == [1]
    assert first_result == {"reward": 0}
    assert second_result["exit_status"] == "rollout_cancelled"
    assert pool.in_flight == [0]


@pytest.mark.asyncio
async def test_agent_worker_abort_cancels_active_sandboxes(monkeypatch):
    terminated = []

    async def fake_terminate(sandbox_ids):
        terminated.extend(sandbox_ids)

    monkeypatch.setattr(agent_function_module, "_terminate_sandboxes", fake_terminate)
    worker = _AgentWorker(worker_index=0, threads=1)
    episode_id, cancelled = worker._abort_state.start()
    worker._abort_state.attach_sandbox(episode_id, "sb-123")

    await worker.abort_episodes(4)

    assert cancelled.is_set()
    assert terminated == ["sb-123"]
    assert worker._abort_generation == 4

    await worker.abort_episodes(3)

    assert terminated == ["sb-123"]
    assert worker._abort_generation == 4
    worker._abort_state.finish(episode_id)


def test_episode_abort_state_cancels_pre_sandbox_work():
    state = _EpisodeAbortState()
    episode_id, cancelled = state.start()

    assert state.cancel_all() == []
    assert cancelled.is_set()
    state.finish(episode_id)


@pytest.mark.asyncio
async def test_agent_worker_records_live_phase_accounting(monkeypatch):
    def fake_episode(*, phase_callback, **_kwargs):
        phase_callback("sandbox_boot")
        phase_callback("model_generation")
        phase_callback("interaction")
        return {"agent_metrics": {}}

    monkeypatch.setattr(
        agent_function_module,
        "_run_episode_sync",
        fake_episode,
    )
    worker = _AgentWorker(worker_index=3, threads=1)
    result = await worker.run_episode(
        {
            "submitted_at_unix": time.time(),
            "base_url": "http://example",
            "prompt": "prompt",
            "request_kwargs": {},
            "metadata": {},
        }
    )

    metrics = result["agent_metrics"]
    assert metrics["agent_worker_index"] == 3
    assert metrics["phase_accounted_seconds"] >= 0
    assert "phase_executor_queue_seconds" in metrics
    assert "phase_sandbox_boot_seconds" in metrics
    assert "phase_model_generation_seconds" in metrics
    assert "phase_interaction_seconds" in metrics
    assert (await worker.stats())["active"] == 0


@pytest.mark.asyncio
async def test_agent_worker_preserves_failure_phase_and_root_cause(monkeypatch):
    def fake_episode(*, phase_callback, **_kwargs):
        phase_callback("sandbox_setup")
        raise ConnectionError("sandbox connection reset")

    monkeypatch.setattr(agent_function_module, "_run_episode_sync", fake_episode)
    worker = _AgentWorker(worker_index=2, threads=1)
    result = await worker.run_episode(
        {
            "submitted_at_unix": time.time(),
            "base_url": "http://example",
            "prompt": "prompt",
            "request_kwargs": {},
            "metadata": {},
        }
    )

    assert result["_miles_abort"] is True
    assert result["failure_phase"] == "sandbox_setup"
    assert result["root_error_type"] == "builtins.ConnectionError"
    assert result["agent_metrics"]["agent_worker_index"] == 2


@pytest.mark.asyncio
async def test_agent_worker_propagates_unknown_adapter_bug(monkeypatch):
    def fake_episode(**_kwargs):
        raise RuntimeError("adapter invariant broke")

    monkeypatch.setattr(agent_function_module, "_run_episode_sync", fake_episode)
    worker = _AgentWorker(worker_index=2, threads=1)

    with pytest.raises(
        RuntimeError,
        match="Modal SWE episode failed during executor_queue: RuntimeError: adapter invariant broke",
    ) as error:
        await worker.run_episode(
            {
                "submitted_at_unix": time.time(),
                "base_url": "http://example",
                "prompt": "prompt",
                "request_kwargs": {},
                "metadata": {},
            }
        )

    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(
            "TITO context limit reached: prompt has 65536 tokens, configured max_seq_len is 65536"
        ),
        RuntimeError(
            "Requested token count exceeds the model's maximum context length"
        ),
        RuntimeError(
            "Input length (65530 tokens) exceeds the maximum allowed length (65530 tokens)"
        ),
        RuntimeError(
            "The input (65692 tokens) is longer than the model's context length (65544 tokens)"
        ),
    ],
)
def test_context_limit_errors_are_recognized(error):
    assert _is_context_limit_error(error)


def test_typed_context_limit_errors_are_recognized():
    error_type = type(
        "ContextWindowExceededError",
        (RuntimeError,),
        {"__module__": "litellm.exceptions"},
    )
    coded_error_type = type(
        "BadRequestError",
        (RuntimeError,),
        {"code": "context_length_exceeded"},
    )

    assert _is_context_limit_error(error_type("request rejected"))
    assert _is_context_limit_error(coded_error_type("request rejected"))


def test_unrelated_bad_request_is_not_a_context_limit():
    assert not _is_context_limit_error(
        RuntimeError("appended message has role='assistant'")
    )


def test_truncated_generation_error_is_recognized_through_wrappers():
    class APIError(RuntimeError):
        status_code = 409

    root = APIError(
        "truncated generation cannot be extended: the matched node ended "
        "with finish_reason='length'"
    )
    wrapper = RuntimeError("episode cleanup failed")
    wrapper.__cause__ = root

    assert _is_truncated_generation_error(wrapper)
    assert _is_truncated_generation_error(
        RuntimeError(
            "APIError: Error code: 409 - truncated generation cannot be extended"
        )
    )


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Error code: 409 - session ownership conflict"),
        RuntimeError("truncated generation cannot be extended"),
        RuntimeError("Error code: 500 - truncated generation cannot be extended"),
    ],
)
def test_other_protocol_errors_are_not_truncated_generation(error):
    assert not _is_truncated_generation_error(error)


def test_modal_sandbox_not_found_is_a_distinct_infra_error():
    class SandboxNotFoundError(RuntimeError):
        __module__ = "modal.exception"

    assert _is_sandbox_not_found_error(SandboxNotFoundError("Sandbox not found"))
    assert not _is_sandbox_not_found_error(RuntimeError("file not found"))


def test_only_recognized_external_failures_are_infrastructure():
    service_error_type = type(
        "ServiceError", (Exception,), {"__module__": "modal.exception"}
    )
    litellm_timeout_type = type(
        "Timeout", (Exception,), {"__module__": "litellm.exceptions"}
    )

    assert _is_infrastructure_error(ConnectionError("connection reset"))
    assert _is_infrastructure_error(SandboxTransportError("invalid response"))
    assert _is_infrastructure_error(service_error_type("temporarily unavailable"))
    assert _is_infrastructure_error(litellm_timeout_type("request expired"))
    assert not _is_infrastructure_error(RuntimeError("adapter invariant broke"))
