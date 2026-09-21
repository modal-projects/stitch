from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from cookbook.common import server


@pytest.mark.parametrize(
    ("update_mode", "local_checkpoint_dir", "expected_local_dir"),
    [
        ("disk", "/local-checkpoint/run-a", "/local-checkpoint/run-a"),
        ("cpu", "/local-checkpoint/run-a", "/local-checkpoint/run-a"),
        ("cpu", None, None),
    ],
)
def test_serve_startup_owns_weight_staging_args(
    monkeypatch,
    update_mode: str,
    local_checkpoint_dir: str | None,
    expected_local_dir: str | None,
) -> None:
    endpoint_args: dict = {}
    sidecar_args: dict = {}

    class Endpoint:
        def __init__(self, **kwargs) -> None:
            endpoint_args.update(kwargs)

        def start(self) -> None:
            return None

    endpoint_module = SimpleNamespace(
        SGLangEndpoint=Endpoint,
        start_heartbeat_thread=lambda *_args, **_kwargs: None,
        warmup_chat_completions=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "autoinference_utils.endpoint", endpoint_module)

    def start_sidecar(**kwargs):
        sidecar_args.update(kwargs)
        return object()

    monkeypatch.setattr(server.process, "start_sidecar", start_sidecar)
    monkeypatch.setattr(server.process, "wait_http", lambda *_args: None)

    replica = SimpleNamespace()
    server.serve_startup(
        replica,
        model_name="model-a",
        boot_version=7,
        sglang_args={"--weight-update-local-checkpoint-dir": "/stale"},
        concurrency=4,
        bulletin_root="/bulletin/run-a",
        local_checkpoint_dir=local_checkpoint_dir,
        delta_update_mode=update_mode,
        store_backend="modal_volume",
        volume_name="weights",
        s3_root=None,
        s3_endpoint_url=None,
        run_id="run-a",
        commit_mode="in_place",
        flush_cache_on_commit=True,
        startup_timeout=60,
    )

    args = endpoint_args["extra_server_args"]
    assert args["--weight-update-staging"] == update_mode
    assert args["--weight-version"] == "7"
    if expected_local_dir is None:
        assert "--weight-update-local-checkpoint-dir" not in args
    else:
        assert args["--weight-update-local-checkpoint-dir"] == expected_local_dir
    assert sidecar_args["flush_cache_on_commit"] is True
