"""``SidecarConfig`` flag round-trips, ``main``'s store-factory seam, and
``run``'s Engine/serve wiring."""

from __future__ import annotations

from typing import Any

import pytest

from stitch import sidecar
from stitch.sidecar import SidecarConfig
from stitch.stores.base import Store

_FACTORY = "stitch.sidecar_test:_recording_store_factory"

_BASE = dict(
    bulletin_root="/cache/run-a",
    base_checkpoint_dir="/model",
    run_id="run-a",
    # Opaque to core's config: only ``main`` imports and calls it.
    store_factory=_FACTORY,
)


class _RecordingStore(Store):
    """A Store that records the factory kwargs it was built with."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _recording_store_factory(**kwargs: Any) -> _RecordingStore:
    return _RecordingStore(**kwargs)


def _not_a_store_factory(**kwargs: Any) -> object:
    del kwargs
    return object()


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


def test_sidecar_config_requires_a_module_callable_factory_reference() -> None:
    with pytest.raises(ValueError, match="package.module:callable"):
        SidecarConfig(**{**_BASE, "store_factory": "no-colon"}, delta_update_mode="cpu")


def test_sidecar_config_rejects_factory_reserved_store_options() -> None:
    with pytest.raises(ValueError, match="run_id"):
        SidecarConfig(
            **_BASE, delta_update_mode="cpu", store_options={"run_id": "other"}
        )


_MINIMAL = SidecarConfig(**_BASE, delta_update_mode="cpu")

_FULL = SidecarConfig(
    **_BASE,
    host="127.0.0.1",
    port=9000,
    upstream="http://127.0.0.1:18001",
    local_checkpoint_dir="/cache/weights",
    delta_update_mode="disk",
    disk_load_format="safetensors",
    store_options={
        "backend": "s3",
        "s3_root": "s3://bucket/experiment/run-a",
        "s3_endpoint_url": "https://s3.example.test",
    },
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
            "--store-factory",
            _FACTORY,
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


@pytest.mark.parametrize("option", ["no-equals-sign", "=value-without-key"])
def test_from_argv_rejects_malformed_store_opts(option: str) -> None:
    with pytest.raises(SystemExit):
        SidecarConfig.from_argv([*_MINIMAL.to_argv(), "--store-opt", option])


def test_from_argv_rejects_duplicate_store_opts() -> None:
    argv = [
        *_MINIMAL.to_argv(),
        "--store-opt",
        "backend=s3",
        "--store-opt",
        "backend=modal-volume",
    ]
    with pytest.raises(SystemExit):
        SidecarConfig.from_argv(argv)


def test_run_builds_engine_and_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    store_out = object()
    engine_out = object()

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

    monkeypatch.setattr(sidecar, "SGLangEngine", _engine)
    monkeypatch.setattr(sidecar, "serve", _serve)

    config = SidecarConfig(
        **_BASE,
        local_checkpoint_dir="/cache/weights",
        delta_update_mode="disk",
        commit_mode="quiesce",
        flush_cache_on_commit=True,
        boot_version=2,
        debug_requests=True,
        reconcile_interval=0.0,
        watchdog_interval=1.0,
        watchdog_failure_threshold=7,
    )
    sidecar.run(config, store_out)

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


def test_main_builds_the_store_via_the_configured_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        sidecar, "run", lambda config, store: calls.update(run=(config, store))
    )

    argv = SidecarConfig(
        **_BASE,
        delta_update_mode="cpu",
        store_options={"backend": "s3", "s3_root": "s3://bucket/experiment/run-a"},
    ).to_argv()
    sidecar.main(argv)

    config, store = calls["run"]
    assert isinstance(store, _RecordingStore)
    assert store.kwargs == {
        "local_root": "/cache/run-a",
        "run_id": "run-a",
        "backend": "s3",
        "s3_root": "s3://bucket/experiment/run-a",
    }
    assert config.store_factory == _FACTORY


def test_main_rejects_a_factory_that_returns_a_non_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        sidecar, "run", lambda config, store: calls.update(run=(config, store))
    )

    argv = SidecarConfig(
        **{**_BASE, "store_factory": "stitch.sidecar_test:_not_a_store_factory"},
        delta_update_mode="cpu",
    ).to_argv()
    with pytest.raises(TypeError, match="not a stitch Store"):
        sidecar.main(argv)
    assert "run" not in calls


@pytest.mark.parametrize(
    ("spec", "error", "message"),
    [
        ("no.such.module:factory", ModuleNotFoundError, "No module named"),
        ("stitch.sidecar_test:missing", ImportError, "has no attribute 'missing'"),
        ("stitch.sidecar_test:_BASE", TypeError, "not callable"),
    ],
)
def test_load_store_factory_rejects_unresolvable_references(
    spec: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        sidecar._load_store_factory(spec)
