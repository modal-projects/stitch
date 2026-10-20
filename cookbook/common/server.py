"""Shared rollout-Server logic: boot sglang + the stitch sidecar, and tear them down.

Modal requires an ``@app.cls`` class to live at module (global) scope, so each framework's
``app.py`` defines a thin ``Server`` class there; its ``@enter``/``@exit`` delegate to these
functions. The behavior is shared here; only the class skeleton + its per-app decorators
(image / gpu / volumes) stay local. The public container port is the sidecar (it fronts the
private sglang on SGLANG_PORT).
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

from . import process
from .constants import SGLANG_PORT, SIDECAR_PORT


def serve_startup(
    replica: Any,
    *,
    model_name: str,
    boot_version: int = 0,
    sglang_args: dict,
    concurrency: int,
    bulletin_root: str,
    local_checkpoint_dir: str | None,
    delta_update_mode: str,
    store_backend: str,
    volume_name: str,
    s3_root: str | None,
    s3_endpoint_url: str | None,
    run_id: str,
    commit_mode: str,
    flush_cache_on_commit: bool = False,
    startup_timeout: int,
    engine_health_timeout: float = 5.0,
    proxy_max_connections: int = 100,
    proxy_max_keepalive_connections: int = 20,
    warmup_images: int = 0,
) -> None:
    """Start sglang + the versioned-proxy sidecar on a Server replica (from ``@modal.enter``).
    SGLang starts directly from the immutable boot checkpoint. The sidecar enters
    rotation after its update destination is initialized and it has caught up.
    ``warmup_images`` > 0 also sends one request with that many images, so the
    vision path compiles before the replica takes traffic."""
    from autoinference_utils.endpoint import (
        SGLangEndpoint,
        start_heartbeat_thread,
        warmup_chat_completions,
    )

    if delta_update_mode not in {"disk", "cpu"}:
        raise ValueError(f"unsupported delta update mode: {delta_update_mode!r}")
    if delta_update_mode == "disk" and not local_checkpoint_dir:
        raise ValueError("disk delta update mode requires local_checkpoint_dir")
    sglang_args = {
        **sglang_args,
        "--weight-update-staging": delta_update_mode,
        "--weight-version": str(boot_version),
    }
    sglang_args.pop("--weight-update-local-checkpoint-dir", None)
    if local_checkpoint_dir is not None:
        sglang_args["--weight-update-local-checkpoint-dir"] = local_checkpoint_dir

    replica.endpoint = SGLangEndpoint(
        model_path=model_name,
        worker_port=SGLANG_PORT,
        extra_server_args=sglang_args,
        health_timeout=startup_timeout,
        health_poll_interval=10.0,
        log_requests_level=-1,
    )
    replica.endpoint.start()
    served_model_name = str(sglang_args.get("--served-model-name", model_name))
    warmup = {
        "model": served_model_name,
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
    if warmup_images:
        warmup_chat_completions(
            port=SGLANG_PORT,
            payload=image_warmup_payload(served_model_name, warmup_images),
            successful_requests=1,
            request_timeout=600.0,
            max_attempts_per_request=3,
        )
    replica.sidecar = process.start_sidecar(
        sidecar_port=SIDECAR_PORT,
        sglang_port=SGLANG_PORT,
        bulletin_root=bulletin_root,
        local_checkpoint_dir=local_checkpoint_dir,
        delta_update_mode=delta_update_mode,
        store_backend=store_backend,
        volume_name=volume_name,
        s3_root=s3_root,
        s3_endpoint_url=s3_endpoint_url,
        run_id=run_id,
        boot_version=boot_version,
        commit_mode=commit_mode,
        flush_cache_on_commit=flush_cache_on_commit,
        engine_health_timeout=engine_health_timeout,
        proxy_max_connections=proxy_max_connections,
        proxy_max_keepalive_connections=proxy_max_keepalive_connections,
    )
    # Modal admits the container when @enter returns. The sidecar owns readiness:
    # /health stays 503 through destination initialization and first catch-up.
    process.wait_http(
        f"http://127.0.0.1:{SIDECAR_PORT}/health", replica.sidecar, startup_timeout
    )

    def sidecar_http_health() -> str | None:
        # Stitch owns SGLang functional health. This independent parent probe
        # covers a sidecar that is alive but no longer serving HTTP.
        return process.http_health_error(
            f"http://127.0.0.1:{SIDECAR_PORT}/server_info",
            replica.sidecar,
        )

    def terminate_unresponsive_sidecar() -> None:
        # Closing the exposed port makes Modal replace this failed Server. A
        # graceful stop_fetching_inputs() would incorrectly report it as Done.
        print(
            "[heartbeat] sidecar remained unhealthy; terminating it for replacement",
            flush=True,
        )
        process.terminate_process(replica.sidecar)

    start_heartbeat_thread(
        sidecar_http_health,
        on_failure=terminate_unresponsive_sidecar,
    )

    print(
        f"Rollout server ready: model={served_model_name}, "
        f"boot_version={boot_version}, target_inputs={concurrency}"
    )


def image_warmup_payload(model: str, images: int) -> dict:
    """A chat request with distinct 720p images, so no cache can answer it."""
    content: list[dict] = [
        {"type": "image_url", "image_url": {"url": _png_data_url(1280, 720, shade)}}
        for shade in range(images)
    ]
    content.append({"type": "text", "text": "Reply with exactly OK."})
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _png_data_url(width: int, height: int, shade: int) -> str:
    """A solid-gray RGB PNG, built without an imaging library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = (b"\x00" + bytes([shade % 256]) * (3 * width)) * height
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def serve_stop(replica: Any) -> None:
    """Tear down the sidecar + sglang (from ``@modal.exit``)."""
    process.terminate_process(getattr(replica, "sidecar", None))
    endpoint = getattr(replica, "endpoint", None)
    if endpoint is not None:
        endpoint.stop()
