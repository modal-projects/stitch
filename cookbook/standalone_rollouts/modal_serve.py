"""Standalone Modal rollout provider that implements the hot-load API shim.

Run commands from the repo root, for example:

    uv run --extra modal modal deploy -m cookbook.standalone_rollouts.modal_serve
"""

import importlib
import os
import subprocess
from pathlib import Path

import modal
import modal.experimental

from cookbook.slime_disagg import helpers
from stitch.providers.modal import resolve_flash_gateway_url


PROVIDER_CONFIG = os.environ.get("PROVIDER_CONFIG", "qwen3_4b_hot_load")
exp = importlib.import_module(f"cookbook.standalone_rollouts.configs.{PROVIDER_CONFIG}")

APP_NAME = exp.APP_NAME
MODEL_NAME = exp.MODEL_NAME
ROLLOUT_CONCURRENCY = exp.ROLLOUT_CONCURRENCY
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
MINUTES = 60
SERVER_STARTUP_TIMEOUT = 35 * MINUTES
S3_TRANSPORT_BUCKET_NAME = os.environ.get(
    "STITCH_SHIM_S3_BUCKET_NAME", exp.S3_TRANSPORT_BUCKET_NAME
)
S3_TRANSPORT_KEY_PREFIX = os.environ.get(
    "STITCH_SHIM_S3_KEY_PREFIX", exp.S3_TRANSPORT_KEY_PREFIX
)
S3_TRANSPORT_MOUNT_PATH = exp.S3_TRANSPORT_MOUNT_PATH
S3_TRANSPORT_REGION = os.environ.get("STITCH_SHIM_S3_REGION", exp.S3_TRANSPORT_REGION)
S3_TRANSPORT_OIDC_AUTH_ROLE_ARN = os.environ.get(
    "STITCH_SHIM_S3_OIDC_AUTH_ROLE_ARN", exp.S3_TRANSPORT_OIDC_AUTH_ROLE_ARN
)

SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"

image = (
    modal.Image.from_registry(SLIME_IMAGE_TAG)
    .entrypoint([])
    .run_commands(f"rm -rf {exp.HF_CACHE_PATH}")
    .pip_install(
        "autoinference-utils==0.2.0",
        "boto3",
        "fastapi",
        "httpx",
        "uvicorn",
        "zstandard",
    )
    .env(
        {
            "EXPERIMENT_CONFIG": PROVIDER_CONFIG,
            "PROVIDER_CONFIG": PROVIDER_CONFIG,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "STITCH_SHIM_MODAL_APP_NAME": APP_NAME,
            "STITCH_SHIM_MODAL_CLS_NAME": "Server",
            "STITCH_SHIM_TRANSPORT_ROOT": str(S3_TRANSPORT_MOUNT_PATH),
        }
    )
    .add_local_python_source("stitch")
    .add_local_dir(
        Path(__file__).parents[1],
        remote_path="/root/cookbook",
        ignore=["**/__pycache__"],
    )
)

with image.imports():
    from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions


def _key_prefix_for_mount(prefix: str) -> str | None:
    prefix = prefix.strip("/")
    if not prefix:
        return None
    return f"{prefix}/"


hf_cache_volume = modal.Volume.from_name(
    exp.HF_CACHE_VOLUME_NAME, create_if_missing=True
)
s3_transport_mount = modal.CloudBucketMount(
    bucket_name=S3_TRANSPORT_BUCKET_NAME,
    key_prefix=_key_prefix_for_mount(S3_TRANSPORT_KEY_PREFIX),
    secret=modal.Secret.from_dict({"AWS_REGION": S3_TRANSPORT_REGION})
    if S3_TRANSPORT_REGION
    else None,
    oidc_auth_role_arn=S3_TRANSPORT_OIDC_AUTH_ROLE_ARN,
)
state_dict = modal.Dict.from_name(exp.STATE_DICT_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": MODEL_NAME,
    "--dtype": "bfloat16",
    "--cuda-graph-max-bs": str(ROLLOUT_CONCURRENCY),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    "--update-weight-delta-chunk-bytes": str(
        exp.SGLANG_UPDATE_WEIGHT_DELTA_CHUNK_BYTES
    ),
    "--update-weight-delta-read-workers": str(
        exp.SGLANG_UPDATE_WEIGHT_DELTA_READ_WORKERS
    ),
    **exp.SGLANG_SERVER_ARGS,
}

WARMUP_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}


@app.cls(
    image=image,
    gpu=f"{exp.GPU}:{exp.ROLLOUT_NUM_GPUS_PER_ENGINE}",
    cloud=exp.CLOUD,
    region=exp.REGION,
    volumes={
        str(exp.HF_CACHE_PATH): hf_cache_volume,
        str(S3_TRANSPORT_MOUNT_PATH): s3_transport_mount,
    },
    secrets=[modal.Secret.from_name(exp.SHIM_SECRET_NAME)],
    min_containers=exp.ROLLOUT_MIN_CONTAINERS,
    timeout=40 * MINUTES,
    scaledown_window=15 * MINUTES,
    include_source=False,
)
@modal.experimental.http_server(
    port=SIDECAR_PORT,
    proxy_regions=exp.PROXY_REGIONS,
    exit_grace_period=25,
    startup_timeout=SERVER_STARTUP_TIMEOUT,
)
@modal.concurrent(target_inputs=ROLLOUT_CONCURRENCY)
class Server:
    """One SGLang server plus a provider hot-load API shim."""

    @modal.enter()
    def startup(self) -> None:
        self.endpoint = SGLangEndpoint(
            model_path=MODEL_NAME,
            worker_port=SGLANG_PORT,
            tp=exp.ROLLOUT_NUM_GPUS_PER_ENGINE,
            extra_server_args=SGLANG_SERVER_ARGS,
            health_timeout=SERVER_STARTUP_TIMEOUT,
            health_poll_interval=10.0,
        )
        self.endpoint.start()
        warmup_chat_completions(
            port=SGLANG_PORT,
            payload=WARMUP_PAYLOAD,
            successful_requests=2,
            request_timeout=120.0,
            max_attempts_per_request=3,
        )
        self.sidecar = _start_provider_sidecar()
        helpers.wait_http(
            f"http://127.0.0.1:{SIDECAR_PORT}/health",
            self.sidecar,
            SERVER_STARTUP_TIMEOUT,
        )
        print(
            f"API-shim rollout server ready: model={MODEL_NAME}, target_inputs={ROLLOUT_CONCURRENCY}"
        )

    @modal.exit()
    def stop(self) -> None:
        helpers.terminate_process(getattr(self, "sidecar", None))
        if hasattr(self, "endpoint"):
            self.endpoint.stop()


@app.function(
    image=image,
    volumes={str(exp.HF_CACHE_PATH): hf_cache_volume},
    timeout=2 * 60 * MINUTES,
    secrets=[modal.Secret.from_name(exp.HF_SECRET_NAME)],
    include_source=False,
)
def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_NAME)
    hf_cache_volume.commit()


@app.local_entrypoint()
def print_url() -> None:
    print(frontdoor_url())


@app.local_entrypoint()
def print_secret_template() -> None:
    print(
        "\n".join(
            [
                f"modal secret create {exp.SHIM_SECRET_NAME} \\",
                "  STITCH_SHIM_API_KEY=... \\",
                "  STITCH_SHIM_PROVIDER_MODEL=qwen3-4b \\",
                "  STITCH_SHIM_PROVIDER_DEPLOYMENT=rollout-prod \\",
                "  STITCH_SHIM_BASE_SNAPSHOT_IDENTITY=base",
            ]
        )
    )


@app.local_entrypoint()
def smoke(
    timeout_seconds: int = 30 * MINUTES,
    api_key: str = "",
    provider_model: str = "",
    provider_deployment: str = "",
) -> None:
    """Check the deployed gateway can report pool state and serve a completion."""
    import json
    import time
    import urllib.request

    gateway = frontdoor_url()
    deadline = time.time() + timeout_seconds
    headers = _shim_headers(
        api_key=api_key,
        provider_model=provider_model,
        provider_deployment=provider_deployment,
    )
    while True:
        try:
            req = urllib.request.Request(
                f"{gateway}/hot_load/v1/models/hot_load", headers=headers
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                pool = json.load(resp)
            if len(pool.get("replicas", [])) >= exp.ROLLOUT_MIN_CONTAINERS:
                break
            last_error = f"expected {exp.ROLLOUT_MIN_CONTAINERS} replicas, got {pool}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if time.time() > deadline:
            raise TimeoutError(f"Provider smoke failed: {last_error}")
        print(f"Waiting for provider pool: {last_error}")
        time.sleep(10)

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{gateway}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        print(json.dumps(json.load(resp), indent=2)[:2000])


def provider_gateway_url() -> str:
    urls = Server._experimental_get_flash_urls()
    if not urls:
        return resolve_flash_gateway_url(APP_NAME, Server.__name__)
    return str(urls[0]).rstrip("/")


def _frontdoor_headers(raw: dict[str, str]) -> dict[str, str]:
    """Map the protocol-neutral ``x-session-affinity`` onto Modal's gateway
    session header so Flash co-locates related requests at routing time. This
    must run *before* the gateway (i.e. outside the Server Cls), since the
    gateway picks the container from the header."""
    headers: dict[str, str] = {}
    affinity: str | None = None
    for key, value in raw.items():
        lower = key.lower()
        if lower in {"host", "content-length"}:
            continue
        if lower == "x-session-affinity":
            affinity = value
            continue
        headers[key] = value
    if affinity:
        headers["Modal-Session-ID"] = affinity
    return headers


@app.function(image=image, min_containers=1, scaledown_window=15 * MINUTES, include_source=False)
@modal.concurrent(max_inputs=ROLLOUT_CONCURRENCY)
@modal.asgi_app()
def frontdoor():
    """Public front door for external/opaque clients (the api-shim contract).

    External trainers keep sending the neutral ``x-session-affinity`` header;
    this proxy rewrites it to ``Modal-Session-ID`` and forwards everything to the
    internal Server Flash gateway, so affinity is honored by Modal's native
    gateway routing (one hop, request gated by the per-container sidecar) instead
    of being re-routed inside a container.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
    import httpx

    # No `from __future__ import annotations` in this module, so proxy()'s
    # `request: Request` / `-> Response` annotations evaluate eagerly against
    # these local imports — FastAPI resolves them without a globals() injection.
    api = FastAPI()
    gateway: dict[str, str | None] = {"url": None}

    @api.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(path: str, request: Request) -> Response:
        if gateway["url"] is None:
            gateway["url"] = provider_gateway_url()
        headers = _frontdoor_headers(dict(request.headers))
        body = await request.body()
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            resp = await client.request(
                request.method,
                f"{gateway['url']}/{path}",
                headers=headers,
                content=body,
                params=request.query_params,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type") or None,
        )

    return api


def frontdoor_url() -> str:
    """Advertised provider URL: external clients hit the front door, not the
    Server Flash gateway directly, so the affinity relabel happens pre-gateway."""
    return frontdoor.get_web_url().rstrip("/")


def _start_provider_sidecar() -> subprocess.Popen:
    cmd = [
        "python3",
        "-m",
        "cookbook.standalone_rollouts.provider",
        "--host",
        "0.0.0.0",
        "--port",
        str(SIDECAR_PORT),
        "--upstream-url",
        f"http://127.0.0.1:{SGLANG_PORT}",
        "--state-dict-name",
        exp.STATE_DICT_NAME,
        "--snapshot-root",
        exp.SNAPSHOT_ROOT,
        "--transport-root",
        str(S3_TRANSPORT_MOUNT_PATH),
        "--base-snapshot-identity",
        os.environ.get(
            "STITCH_SHIM_BASE_SNAPSHOT_IDENTITY", exp.BASE_SNAPSHOT_IDENTITY
        ),
    ]
    print("Starting provider shim:", " ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def _shim_headers(
    *, api_key: str = "", provider_model: str = "", provider_deployment: str = ""
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if api_key := (api_key or os.environ.get("STITCH_SHIM_API_KEY")):
        headers["Authorization"] = f"Bearer {api_key}"
    if provider_model := (
        provider_model or os.environ.get("STITCH_SHIM_PROVIDER_MODEL")
    ):
        headers["Provider-Model"] = provider_model
    if provider_deployment := (
        provider_deployment or os.environ.get("STITCH_SHIM_PROVIDER_DEPLOYMENT")
    ):
        headers["Provider-Deployment"] = provider_deployment
    return headers
