from __future__ import annotations

import io
import json

import pytest

from cookbook.common import process, storage
from stitch.sidecar import SidecarConfig, _load_store_factory
from stitch.stores.s3 import S3Store


class _JsonResponse:
    status = 200

    def __init__(self, payload: dict) -> None:
        self._body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self._body.close()

    def read(self, *args, **kwargs) -> bytes:
        return self._body.read(*args, **kwargs)


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


def test_wait_sidecar_ready_waits_for_catchup_and_destination(monkeypatch) -> None:
    statuses = iter(
        [
            {"ready": False, "update_destination_ready": False},
            {"ready": True, "update_destination_ready": False},
            {"ready": True, "update_destination_ready": True},
        ]
    )
    calls = 0

    def urlopen(_url: str, timeout: int) -> _JsonResponse:
        nonlocal calls
        assert timeout == 5
        calls += 1
        return _JsonResponse(next(statuses))

    monkeypatch.setattr(process.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(process.time, "sleep", lambda _seconds: None)

    process.wait_sidecar_ready("http://sidecar/server_info", None, timeout=10)

    assert calls == 3


def test_wait_sidecar_ready_surfaces_destination_failure(monkeypatch) -> None:
    def urlopen(_url: str, timeout: int) -> _JsonResponse:
        return _JsonResponse(
            {
                "ready": True,
                "update_destination_ready": False,
                "update_destination_error": "layout failed",
            }
        )

    monkeypatch.setattr(process.urllib.request, "urlopen", urlopen)

    with pytest.raises(RuntimeError, match="layout failed"):
        process.wait_sidecar_ready("http://sidecar/server_info", None, timeout=10)
