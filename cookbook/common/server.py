"""Shared rollout-Server logic: boot sglang + the stitch sidecar, and tear them down.

Modal requires an ``@app.cls`` class to live at module (global) scope, so each framework's
``app.py`` defines a thin ``Server`` class there; its ``@enter``/``@exit`` delegate to these
functions. The behavior is shared here; only the class skeleton + its per-app decorators
(image / gpu / volumes) stay local. The public container port is the sidecar (it fronts the
private sglang on SGLANG_PORT).
"""

from __future__ import annotations

from typing import Any

from stitch.service import sync_in_progress

from . import process
from .constants import SGLANG_PORT, SIDECAR_PORT


def serve_startup(
    replica: Any,
    *,
    model_name: str,
    sglang_args: dict,
    tp: int,
    concurrency: int,
    bulletin_root: str,
    local_checkpoint_dir: str | None,
    delta_update_mode: str,
    volume_name: str,
    commit_mode: str,
    flush_cache_on_commit: bool = False,
    startup_timeout: int,
) -> None:
    """Start sglang + the versioned-proxy sidecar on a Server replica (from ``@modal.enter``).
    SGLang starts directly from the immutable base. The sidecar initializes the
    selected update destination in the background after serving is available."""
    from autoinference_utils.endpoint import (
        SGLangEndpoint,
        start_heartbeat_thread,
        warmup_chat_completions,
    )

    if delta_update_mode not in {"disk", "cpu"}:
        raise ValueError(f"unsupported delta update mode: {delta_update_mode!r}")
    cpu_cache_enabled = "--enable-cpu-weight-cache" in sglang_args
    if cpu_cache_enabled != (delta_update_mode == "cpu"):
        raise ValueError(
            "SGLANG_DELTA_UPDATE_MODE must be 'cpu' exactly when "
            "--enable-cpu-weight-cache is present in SGLANG_SERVER_ARGS"
        )

    replica.endpoint = SGLangEndpoint(
        model_path=model_name,
        worker_port=SGLANG_PORT,
        tp=tp,
        extra_server_args=sglang_args,
        health_timeout=startup_timeout,
        health_poll_interval=10.0,
        log_requests_level=-1,
    )
    replica.endpoint.start()
    warmup = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    warmup_chat_completions(
        port=SGLANG_PORT,
        payload=warmup,
        successful_requests=2,
        request_timeout=120.0,
        max_attempts_per_request=3,
    )
    replica.sidecar = process.start_sidecar(
        sidecar_port=SIDECAR_PORT,
        sglang_port=SGLANG_PORT,
        bulletin_root=bulletin_root,
        base_checkpoint_dir=model_name,
        local_checkpoint_dir=local_checkpoint_dir,
        delta_update_mode=delta_update_mode,
        disk_load_format=str(sglang_args.get("--load-format", "auto")),
        volume_name=volume_name,
        commit_mode=commit_mode,
        flush_cache_on_commit=flush_cache_on_commit,
    )
    # Modal admits the container to Flash routing when @enter returns and never re-polls /health, so
    # blocking here on /health (503 until the reconciler's first catch-up) is the only thing that keeps a
    # not-yet-synced replica out of rotation. Fresh boot (no pointer) clears at once; a mid-run joiner
    # waits until it has applied the live version, bounded by startup_timeout.
    process.wait_http(
        f"http://127.0.0.1:{SIDECAR_PORT}/health", replica.sidecar, startup_timeout
    )

    def engine_health() -> str | None:
        # Weight staging can make the engine health endpoint briefly stale.
        # Suppress that expected blip only while the reconciler reports work.
        error = replica.endpoint.health_check()
        if error is None:
            return None
        return (
            None
            if sync_in_progress(f"http://127.0.0.1:{SIDECAR_PORT}/server_info")
            else error
        )

    import modal.experimental

    start_heartbeat_thread(
        engine_health,
        on_failure=lambda: modal.experimental.stop_fetching_inputs(),
        max_consecutive_failures=12,  # ~1 min of sustained idle-state failures
    )
    print(f"Rollout server ready: model={model_name}, target_inputs={concurrency}")


def serve_stop(replica: Any) -> None:
    """Tear down the sidecar + sglang (from ``@modal.exit``)."""
    process.terminate_process(getattr(replica, "sidecar", None))
    endpoint = getattr(replica, "endpoint", None)
    if endpoint is not None:
        endpoint.stop()
