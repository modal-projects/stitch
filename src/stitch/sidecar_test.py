"""``SidecarConfig`` flag round-trips and ``main``'s Store/Engine/serve wiring."""

from __future__ import annotations

from typing import Any

import pytest

from stitch import sidecar
from stitch.sidecar import SidecarConfig

_BASE = dict(
    bulletin_root="/cache/run-a",
    base_checkpoint_dir="/model",
    run_id="run-a",
)


def test_sidecar_config_requires_run_id() -> None:
    with pytest.raises(ValueError, match="run_id is required"):
        SidecarConfig(**{**_BASE, "run_id": ""}, delta_update_mode="cpu")


def test_sidecar_config_requires_nonnegative_boot_version() -> None:
    with pytest.raises(ValueError, match="boot_version must be non-negative"):
        SidecarConfig(**_BASE, delta_update_mode="cpu", boot_version=-1)


def test_sidecar_config_requires_local_checkpoint_dir_in_disk_mode() -> None:
    with pytest.raises(
        ValueError, match="local_checkpoint_dir is required in disk mode"
    ):
        SidecarConfig(**_BASE, delta_update_mode="disk")


_MINIMAL = SidecarConfig(**_BASE, delta_update_mode="cpu")

_FULL = SidecarConfig(
    **_BASE,
    host="127.0.0.1",
    port=9000,
    upstream="http://127.0.0.1:18001",
    local_checkpoint_dir="/cache/weights",
    delta_update_mode="disk",
    disk_load_format="safetensors",
    store_backend="s3",
    volume_name="weights",
    s3_root="s3://bucket/experiment/run-a",
    s3_endpoint_url="https://s3.example.test",
    commit_mode="quiesce",
    flush_cache_on_commit=True,
    boot_version=3,
    debug_requests=True,
    reconcile_interval=2.5,
    watchdog_interval=1.0,
    watchdog_failure_threshold=7,
)


@pytest.mark.parametrize("config", [_MINIMAL, _FULL])
def test_argv_round_trips_losslessly(config: SidecarConfig) -> None:
    assert SidecarConfig.from_argv(config.to_argv()) == config


def test_from_argv_fills_unflagged_fields_with_defaults() -> None:
    parsed = SidecarConfig.from_argv(
        [
            "--bulletin-root",
            "/cache/run-a",
            "--base-checkpoint-dir",
            "/model",
            "--delta-update-mode",
            "cpu",
            "--run-id",
            "run-a",
        ]
    )
    assert parsed == _MINIMAL


def test_disk_mode_requires_local_checkpoint_dir() -> None:
    argv = _MINIMAL.to_argv()
    argv[argv.index("cpu")] = "disk"
    with pytest.raises(SystemExit):
        SidecarConfig.from_argv(argv)


def test_main_builds_store_engine_and_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    store_out = object()
    engine_out = object()

    def _create_store(backend: str, **kwargs: Any) -> object:
        calls["store"] = (backend, kwargs)
        return store_out

    def _engine(
        base_url: str,
        base_checkpoint_dir: str,
        local_checkpoint_dir: str | None = None,
        **kwargs: Any,
    ) -> object:
        calls["engine"] = (base_url, base_checkpoint_dir, local_checkpoint_dir, kwargs)
        return engine_out

    def _serve(store: Any, engine: Any, **kwargs: Any) -> None:
        calls["serve"] = (store, engine, kwargs)

    monkeypatch.setattr(sidecar, "create_store", _create_store)
    monkeypatch.setattr(sidecar, "SGLangEngine", _engine)
    monkeypatch.setattr(sidecar, "serve", _serve)

    config = SidecarConfig(
        **_BASE,
        local_checkpoint_dir="/cache/weights",
        delta_update_mode="disk",
        store_backend="s3",
        s3_root="s3://bucket/experiment/run-a",
        commit_mode="quiesce",
        flush_cache_on_commit=True,
        boot_version=2,
        debug_requests=True,
        reconcile_interval=0.0,
        watchdog_interval=1.0,
        watchdog_failure_threshold=7,
    )
    sidecar.main(config.to_argv())

    assert calls["store"] == (
        "s3",
        {
            "local_root": "/cache/run-a",
            "volume_name": None,
            "run_id": "run-a",
            "s3_root": "s3://bucket/experiment/run-a",
            "s3_endpoint_url": None,
        },
    )
    assert calls["engine"] == (
        "http://127.0.0.1:8001",
        "/model",
        "/cache/weights",
        {"delta_update_mode": "disk", "disk_load_format": "auto"},
    )
    store, engine, serve_kwargs = calls["serve"]
    assert store is store_out
    assert engine is engine_out
    assert serve_kwargs == {
        "run_id": "run-a",
        "boot_version": 2,
        "commit_mode": "quiesce",
        "flush_cache_on_commit": True,
        "host": "0.0.0.0",
        "port": 8000,
        "debug_requests": True,
        "reconcile_interval": 0.0,
        "watchdog_interval": 1.0,
        "watchdog_failure_threshold": 7,
    }
