"""Shared rollout-Server logic: boot sglang + the stitch sidecar, and tear them down.

Modal requires an ``@app.cls`` class to live at module (global) scope, so each framework's
``app.py`` defines a thin ``Server`` class there; its ``@enter``/``@exit`` delegate to these
functions. The behavior is shared here; only the class skeleton + its per-app decorators
(image / gpu / volumes) stay local. The public container port is the sidecar (it fronts the
private sglang on SGLANG_PORT).
"""

from __future__ import annotations

import os
from typing import Any

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
    local_checkpoint_dir: str,
    volume_name: str,
    commit_mode: str,
    flush_cache_on_commit: bool = False,
    startup_timeout: int,
    sglang_env: dict[str, str] | None = None,
) -> None:
    """Start sglang + the versioned-proxy sidecar on a Server replica (from ``@modal.enter``).
    The engine serves ``model_name`` and materializes each version into
    ``local_checkpoint_dir`` itself via /pull_weights; the sidecar drives the sync.
    ``sglang_env`` is a per-config override of the sglang process env (over the image's
    baked defaults) — set before launch so the engine subprocess inherits it."""
    from autoinference_utils.endpoint import (
        SGLangEndpoint,
        start_heartbeat_thread,
        warmup_chat_completions,
    )

    if sglang_env:
        os.environ.update(sglang_env)
    replica.endpoint = SGLangEndpoint(
        model_path=model_name, worker_port=SGLANG_PORT, tp=tp,
        extra_server_args=sglang_args, health_timeout=startup_timeout, health_poll_interval=10.0,
    )
    replica.endpoint.start()
    warmup = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
    }
    warmup_chat_completions(port=SGLANG_PORT, payload=warmup, successful_requests=2,
                            request_timeout=120.0, max_attempts_per_request=3)
    replica.sidecar = process.start_sidecar(
        sidecar_port=SIDECAR_PORT, sglang_port=SGLANG_PORT, bulletin_root=bulletin_root,
        local_checkpoint_dir=local_checkpoint_dir, volume_name=volume_name, commit_mode=commit_mode,
        flush_cache_on_commit=flush_cache_on_commit,
    )
    # /server_info, not /health: /health stays 503 until catch-up, which would spin here and time out.
    process.wait_http(f"http://127.0.0.1:{SIDECAR_PORT}/server_info", replica.sidecar, startup_timeout)

    def engine_health() -> str | None:
        # The base seed (engine.prefetch) and every delta apply drive the fork's /pull_weights,
        # which starves sglang's event loop enough that its detokenizer heartbeat goes stale and
        # /health 503s. That stall is EXPECTED while the sidecar is seeding or mid-sync — report
        # failures only when it is genuinely idle, or replicas crash-cycle through every sync (the
        # base seed runs with sync_state=IDLE, so it needs the prefetch_done check, not just sync_state).
        # A dead engine process still raises once the seed/sync is done (or errored).
        error = replica.endpoint.health_check()
        if error is None:
            return None
        try:
            import json
            import urllib.request

            with urllib.request.urlopen(
                f"http://127.0.0.1:{SIDECAR_PORT}/server_info", timeout=5
            ) as response:
                info = json.loads(response.read())
                seeding = not info.get("prefetch_done", True) and not info.get("prefetch_error")
                if seeding or info.get("sync_state") in (
                    "QUEUED", "PREFETCHING", "PREPARING", "COMMITTING",
                ):
                    return None
        except Exception:  # noqa: BLE001 — sidecar unreachable: report the engine error
            pass
        return error

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
