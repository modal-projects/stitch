from __future__ import annotations

import urllib.error

from cookbook.common import process, storage
from stitch.sidecar import SidecarConfig, _load_store_factory
from stitch.stores.s3 import S3Store


class _Process:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


def test_http_health_error_checks_process_and_http(monkeypatch) -> None:
    monkeypatch.setattr(
        process.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response()
    )

    assert process.http_health_error("http://sidecar/server_info", _Process()) is None
    assert (
        process.http_health_error("http://sidecar/server_info", _Process(returncode=7))
        == "process exited with code 7"
    )


def test_http_health_error_reports_request_failure(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise TimeoutError("sidecar stalled")

    monkeypatch.setattr(process.urllib.request, "urlopen", fail)

    assert (
        process.http_health_error("http://sidecar/server_info", _Process())
        == "TimeoutError: sidecar stalled"
    )


def test_http_health_error_reports_http_status(monkeypatch) -> None:
    def fail(url, **_kwargs):
        raise urllib.error.HTTPError(url, 503, "not ready", {}, None)

    monkeypatch.setattr(process.urllib.request, "urlopen", fail)

    assert (
        process.http_health_error("http://sidecar/server_info", _Process())
        == "HTTP 503 from http://sidecar/server_info"
    )


def test_start_sidecar_passes_s3_store_settings(monkeypatch) -> None:
    commands: list[list[str]] = []

    def popen(command: list[str], **_kwargs):
        commands.append(command)
        return object()

    monkeypatch.setattr(process.subprocess, "Popen", popen)

    process.start_sidecar(
        sidecar_port=8000,
        sglang_port=8001,
        bulletin_root="/cache/run-a",
        base_checkpoint_dir="/model",
        local_checkpoint_dir=None,
        delta_update_mode="cpu",
        disk_load_format="auto",
        store_backend=storage.S3,
        volume_name="",
        s3_root="s3://bucket/experiment/run-a",
        s3_endpoint_url="https://s3.example.test",
        run_id="run-a",
        boot_version=0,
        commit_mode="in_place",
    )

    command = commands[0]
    assert command[:3] == ["python3", "-m", "stitch.sidecar"]
    assert command[command.index("--store-factory") + 1] == process.STORE_FACTORY
    assert command[command.index("--boot-version") + 1] == "0"

    # The launched argv must reconstruct the same store the recipe chose: parse
    # it back and drive core's factory seam exactly as ``stitch.sidecar.main``
    # does. The empty volume_name never reaches the flags.
    config = SidecarConfig.from_argv(command[3:])
    assert config.store_options == {
        "backend": storage.S3,
        "s3_root": "s3://bucket/experiment/run-a",
        "s3_endpoint_url": "https://s3.example.test",
    }
    factory = _load_store_factory(config.store_factory)
    assert factory is storage.create_store
    store = factory(
        local_root=config.bulletin_root,
        run_id=config.run_id,
        **config.store_options,
    )
    assert isinstance(store, S3Store)
    assert store.run_id == "run-a"
