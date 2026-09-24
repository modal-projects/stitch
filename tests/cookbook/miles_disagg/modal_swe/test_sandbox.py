import asyncio
from pathlib import Path

import pytest

from cookbook.miles_disagg.modal_swe import sandbox as sandbox_module


def _task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    environment = task_dir / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n")
    return task_dir


def test_environment_uses_v2_and_waits_for_task_setup_readiness(monkeypatch, tmp_path):
    monkeypatch.setenv("MODAL_SWE_MEMORY_MIB", "2048")

    class Sandbox:
        def __init__(self):
            self.wait_calls = 0
            self.exec_calls = []

        def exec(self, *args):
            self.exec_calls.append(args)

            class Process:
                def wait(self):
                    return 0

            return Process()

        def wait_until_ready(self):
            self.wait_calls += 1

    sandbox = Sandbox()
    create_calls = []
    probe = object()
    clock = iter([10.0, 12.5, 15.0])

    monkeypatch.setattr(sandbox_module, "_cached_image", lambda _name: object())
    monkeypatch.setattr(
        sandbox_module, "_cached_app", lambda _name, **_kwargs: object()
    )
    monkeypatch.setattr(
        sandbox_module.modal.Probe,
        "with_exec",
        lambda *args, **kwargs: (args, kwargs, probe),
    )

    def create(*args, **kwargs):
        create_calls.append((args, kwargs))
        return sandbox

    monkeypatch.setattr(sandbox_module.modal.Sandbox, "_experimental_create", create)
    monkeypatch.setattr(
        sandbox_module.modal.Sandbox,
        "create",
        lambda *_args, **_kwargs: pytest.fail("Sandbox v1 create must not be used"),
    )
    monkeypatch.setattr(sandbox_module.time, "perf_counter", lambda: next(clock))

    environment = sandbox_module.ModalSWEEnvironment(_task_dir(tmp_path))

    assert sandbox.wait_calls == 0
    assert environment.schedule_time == 2.5
    assert environment.readiness_time is None
    assert environment.boot_time is None

    environment.mark_ready()

    assert sandbox.wait_calls == 1
    assert sandbox.exec_calls == [("touch", sandbox_module._SANDBOX_READY_PATH)]
    assert environment.schedule_time == 2.5
    assert environment.readiness_time == 2.5
    assert environment.boot_time == 5.0
    args, kwargs = create_calls[0]
    assert args == ("sleep", "infinity")
    assert kwargs["readiness_probe"] == (
        ("test", "-f", sandbox_module._SANDBOX_READY_PATH),
        {"interval_ms": 250},
        probe,
    )
    assert kwargs["block_network"] is True
    assert kwargs["cpu"] is None
    assert kwargs["memory"] == 2048


def test_lifecycle_diagnostics_include_v2_identity_and_exit_state(
    monkeypatch, tmp_path
):
    class Sandbox:
        object_id = "sb-test"

        def poll(self):
            return 137

    sandbox = Sandbox()
    monkeypatch.setattr(sandbox_module, "_cached_image", lambda _name: object())
    monkeypatch.setattr(
        sandbox_module, "_cached_app", lambda _name, **_kwargs: object()
    )
    monkeypatch.setattr(
        sandbox_module.modal.Probe, "with_exec", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        sandbox_module.modal.Sandbox,
        "_experimental_create",
        lambda *_args, **_kwargs: sandbox,
    )

    environment = sandbox_module.ModalSWEEnvironment(_task_dir(tmp_path))
    diagnostics = environment.lifecycle_diagnostics()

    assert diagnostics["sandbox_id"] == "sb-test"
    assert diagnostics["sandbox_return_code"] == 137
    assert diagnostics["sandbox_schedule_time"] >= 0
    assert diagnostics["sandbox_readiness_time"] is None
    assert diagnostics["sandbox_boot_time"] is None


def test_readiness_failure_reclaims_v2_sandbox(monkeypatch, tmp_path):
    class Sandbox:
        def __init__(self):
            self.terminated = 0
            self.detached = 0

        def wait_until_ready(self):
            raise TimeoutError("not ready")

        def exec(self, *_args):
            class Process:
                def wait(self):
                    return 0

            return Process()

        def terminate(self):
            self.terminated += 1

        def detach(self):
            self.detached += 1

    sandbox = Sandbox()
    monkeypatch.setattr(sandbox_module, "_cached_image", lambda _name: object())
    monkeypatch.setattr(
        sandbox_module, "_cached_app", lambda _name, **_kwargs: object()
    )
    monkeypatch.setattr(
        sandbox_module.modal.Probe, "with_exec", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        sandbox_module.modal.Sandbox,
        "_experimental_create",
        lambda *_args, **_kwargs: sandbox,
    )

    environment = sandbox_module.ModalSWEEnvironment(_task_dir(tmp_path))
    with pytest.raises(TimeoutError, match="not ready"):
        environment.mark_ready()

    assert sandbox.terminated == 1
    assert sandbox.detached == 1


def test_app_is_created_once_then_workers_only_look_it_up(monkeypatch):
    async_calls = []
    sync_calls = []

    class Lookup:
        def __call__(self, name, *, create_if_missing):
            sync_calls.append((name, create_if_missing))
            return object()

        async def aio(self, name, *, create_if_missing):
            async_calls.append((name, create_if_missing))
            return object()

    monkeypatch.setattr(sandbox_module, "_APP_CACHE", {})
    monkeypatch.setattr(sandbox_module.modal.App, "lookup", Lookup())

    async def ensure_twice() -> None:
        await asyncio.gather(
            sandbox_module.ensure_sandbox_app("run-sandboxes"),
            sandbox_module.ensure_sandbox_app("run-sandboxes"),
        )

    asyncio.run(ensure_twice())
    assert async_calls == [("run-sandboxes", True)]

    # A Ray worker has an independent process-local cache. Its first access
    # must resolve the app without racing to create it again.
    sandbox_module._APP_CACHE.clear()
    sandbox_module._cached_app("run-sandboxes")
    assert sync_calls == [("run-sandboxes", False)]


def test_stop_detaches_once_even_when_termination_fails(caplog):
    class Sandbox:
        def __init__(self):
            self.terminate_calls = 0
            self.detach_calls = 0

        def terminate(self):
            self.terminate_calls += 1
            raise RuntimeError("terminate failed")

        def detach(self):
            self.detach_calls += 1

    sandbox = Sandbox()
    environment = sandbox_module.ModalSWEEnvironment.__new__(
        sandbox_module.ModalSWEEnvironment
    )
    environment.sandbox = sandbox
    environment._stopped = False

    environment.stop()
    environment.stop()

    assert sandbox.terminate_calls == 1
    assert sandbox.detach_calls == 1
    assert "Failed to terminate Modal Sandbox" in caplog.text
