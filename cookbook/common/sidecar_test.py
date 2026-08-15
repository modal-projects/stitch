"""The cookbook sidecar entrypoint: recipes decide the backend via storage."""

from __future__ import annotations

from typing import Any

import pytest

from cookbook.common import sidecar as sidecar_entry
from cookbook.common import storage


def test_main_builds_store_via_storage_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    store_out = object()

    def _create_store(backend: str, **kwargs: Any) -> object:
        calls["store"] = (backend, kwargs)
        return store_out

    def _run(config: Any, store: Any) -> None:
        calls["run"] = (config, store)

    monkeypatch.setattr(storage, "create_store", _create_store)
    monkeypatch.setattr(sidecar_entry, "run", _run)

    sidecar_entry.main(
        [
            "--bulletin-root",
            "/cache/run-a",
            "--base-checkpoint-dir",
            "/model",
            "--delta-update-mode",
            "cpu",
            "--store-backend",
            "s3",
            "--s3-root",
            "s3://bucket/experiment/run-a",
            "--run-id",
            "run-a",
        ]
    )

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
    config, store = calls["run"]
    assert config.store_backend == "s3"
    assert store is store_out


def test_unknown_backend_is_storage_error() -> None:
    with pytest.raises(ValueError, match="unsupported store backend"):
        storage.create_store("no-such-backend", local_root="/x", run_id="r")
