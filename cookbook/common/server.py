"""The shared rollout ``Server``: one sglang engine plus the stitch versioned-proxy
sidecar. Identical across frameworks, so it is defined once here and each framework's
app registers it with its own image / GPU / volumes / config via ``register_server``.

The public container port is the sidecar (it fronts the private sglang on SGLANG_PORT).
"""

from __future__ import annotations

from typing import Any

import modal
import modal.experimental

from . import process

SIDECAR_PORT = 8000
SGLANG_PORT = 8001
MINUTES = 60


def register_server(
    app: Any,
    *,
    image: Any,
    gpu: str,
    volumes: dict,
    cloud: str | None,
    region: str | None,
    model_name: str,
    sglang_args: dict,
    tp: int,
    concurrency: int,
    bulletin_root: str,
    local_checkpoint_dir: str,
    volume_name: str,
    commit_mode: str,
    min_containers: int,
    max_containers: int | None,
    proxy_regions: list[str],
    ephemeral_disk_mib: int | None,
    memory_mib: int | None,
    startup_timeout: int,
) -> Any:
    """Build + register the Server class on ``app`` and return it. Called at module load
    in each framework's ``app.py``; re-run on container import, so the closure config is
    reconstructed from the run config on both sides."""

    @app.cls(
        image=image, gpu=gpu, cloud=cloud, region=region, volumes=volumes,
        min_containers=min_containers, max_containers=max_containers,
        timeout=40 * MINUTES, scaledown_window=15 * MINUTES,
        ephemeral_disk=ephemeral_disk_mib, memory=memory_mib, include_source=False,
    )
    @modal.experimental.http_server(
        port=SIDECAR_PORT, proxy_regions=proxy_regions,
        exit_grace_period=25, startup_timeout=startup_timeout,
    )
    @modal.concurrent(target_inputs=concurrency)
    class Server:
        @modal.enter()
        def startup(self) -> None:
            from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions

            self.endpoint = SGLangEndpoint(
                model_path=model_name, worker_port=SGLANG_PORT, tp=tp,
                extra_server_args=sglang_args, health_timeout=startup_timeout, health_poll_interval=10.0,
            )
            self.endpoint.start()
            warmup = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            }
            warmup_chat_completions(port=SGLANG_PORT, payload=warmup, successful_requests=2,
                                    request_timeout=120.0, max_attempts_per_request=3)
            # The engine serves model_name and materializes each version into
            # local_checkpoint_dir itself via /pull_weights; the sidecar drives the sync.
            self.sidecar = process.start_sidecar(
                sidecar_port=SIDECAR_PORT, sglang_port=SGLANG_PORT,
                bulletin_root=bulletin_root, local_checkpoint_dir=local_checkpoint_dir,
                volume_name=volume_name, commit_mode=commit_mode,
            )
            process.wait_http(f"http://127.0.0.1:{SIDECAR_PORT}/health", self.sidecar, startup_timeout)
            print(f"Rollout server ready: model={model_name}, target_inputs={concurrency}")

        @modal.exit()
        def stop(self) -> None:
            process.terminate_process(getattr(self, "sidecar", None))
            if hasattr(self, "endpoint"):
                self.endpoint.stop()

    return Server
