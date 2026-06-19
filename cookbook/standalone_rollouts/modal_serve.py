"""Standalone Modal rollout provider implementing the customer hot-load API.

Log-as-truth design: the front door is a singleton that owns the monotonic
``latest`` pointer in the S3 transport and implements the customer's
``POST/GET /hot_load`` API. An elastic Flash pool of
SGLang servers + stitch sidecars reconciles to ``latest`` on its own and serves
inference. There is no central desired-state mailbox: the pool pulls, and the
front door derives readiness by enumerating the live containers.

Run commands from the repo root, for example:

    uv run --extra modal modal deploy -m cookbook.standalone_rollouts.modal_serve
"""

import asyncio
import importlib
import os
import subprocess
from pathlib import Path

import modal
import modal.experimental

from cookbook.slime_disagg import helpers
from cookbook.standalone_rollouts import frontdoor as frontdoor_mod
from stitch.bulletin import FilesystemBulletinBoard
from stitch.providers.modal import (
    discover_flash_targets,
    resolve_flash_gateway_url,
    wake_targets,
)


PROVIDER_CONFIG = os.environ.get("PROVIDER_CONFIG", "qwen3_4b_hot_load")
exp = importlib.import_module(f"cookbook.standalone_rollouts.configs.{PROVIDER_CONFIG}")

APP_NAME = exp.APP_NAME
MODEL_NAME = exp.MODEL_NAME
ROLLOUT_CONCURRENCY = exp.ROLLOUT_CONCURRENCY
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
MINUTES = 60
SERVER_STARTUP_TIMEOUT = 35 * MINUTES
LOCAL_CHECKPOINT_PATH = exp.LOCAL_CHECKPOINT_PATH
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
SLIME_ROOT = "/root/slime"
SLIME_REPO_URL = "https://github.com/modal-projects/slime.git"
# PR #5 head (disaggregated-rollout, stacked on disk-delta-weight-sync). The
# provider sidecar applies disk deltas host-side via slime.utils.disk_delta, so
# the image must carry that branch's slime plus its checksum/compression deps.
# Pin a SHA, not the branch tip: the clone is a cached image layer.
SLIME_REPO_REF = "570cd0b3bc28141abfbf054333d129d41fe50f19"

image = (
    modal.Image.from_registry(SLIME_IMAGE_TAG)
    .entrypoint([])
    .run_commands(f"rm -rf {exp.HF_CACHE_PATH}")
    # Replace the bundled slime with the disk-delta branch so the sidecar can
    # import slime.utils.disk_delta (host-side apply).
    .run_commands(
        f"rm -rf {SLIME_ROOT}"
        f" && git clone --depth 1 {SLIME_REPO_URL} {SLIME_ROOT}"
        f" && cd {SLIME_ROOT}"
        f" && git fetch --depth 1 origin {SLIME_REPO_REF}"
        f" && git checkout FETCH_HEAD"
        f" && python3 -m pip install --no-deps -e {SLIME_ROOT}"
    )
    .pip_install(
        "autoinference-utils==0.2.0",  # SGLang server lifecycle for the rollout pool
        "boto3",
        "fastapi",
        "httpx",
        "uvicorn",
        # slime.utils.disk_delta host-side apply: zstd decompress + xxhash
        # (xxh3-128 default) / blake3 checksums. slime is installed --no-deps.
        "zstandard",
        "xxhash",
        "blake3",
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
            "STITCH_LOCAL_CHECKPOINT_DIR": LOCAL_CHECKPOINT_PATH,
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
# read_only=False: the front door writes the `latest` pointer here.
s3_transport_mount = modal.CloudBucketMount(
    bucket_name=S3_TRANSPORT_BUCKET_NAME,
    key_prefix=_key_prefix_for_mount(S3_TRANSPORT_KEY_PREFIX),
    secret=modal.Secret.from_dict({"AWS_REGION": S3_TRANSPORT_REGION})
    if S3_TRANSPORT_REGION
    else None,
    oidc_auth_role_arn=S3_TRANSPORT_OIDC_AUTH_ROLE_ARN,
    read_only=False,
)
app = modal.App(APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": MODEL_NAME,
    "--dtype": "bfloat16",
    "--cuda-graph-max-bs": str(ROLLOUT_CONCURRENCY),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
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
    """One SGLang server plus the stitch weight-sync sidecar, reconciling to the
    `latest` pointer the front door advances."""

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
        # Deltas are applied host-side onto a copy of the base checkpoint; the
        # base resolves to the same HF cache snapshot the SGLang server loaded.
        from huggingface_hub import snapshot_download

        base_checkpoint_dir = snapshot_download(MODEL_NAME, local_files_only=True)
        self.sidecar = _start_provider_sidecar(base_checkpoint_dir=base_checkpoint_dir)
        helpers.wait_http(
            f"http://127.0.0.1:{SIDECAR_PORT}/health",
            self.sidecar,
            SERVER_STARTUP_TIMEOUT,
        )
        print(
            f"Rollout server ready: model={MODEL_NAME}, target_inputs={ROLLOUT_CONCURRENCY}"
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


@app.function(
    image=image,
    secrets=[modal.Secret.from_name(exp.SHIM_SECRET_NAME)],
    timeout=35 * MINUTES,
    include_source=False,
)
def check(timeout_seconds: int = 20 * MINUTES) -> None:
    """Authenticated smoke from inside Modal: reads the provider secret (so the
    API key never leaves Modal), polls GET /hot_load readiness through the front
    door, then serves a base completion through it. Run with:

        uv run --extra modal modal run -m cookbook.standalone_rollouts.modal_serve::check
    """
    import json
    import time
    import urllib.request

    gateway = modal.Server.from_name(APP_NAME, "FrontDoor").get_url().rstrip("/")
    headers = _shim_headers()
    deadline = time.time() + timeout_seconds
    last = ""
    while True:
        try:
            req = urllib.request.Request(
                f"{gateway}/hot_load/v1/models/hot_load", headers=headers
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                pool = json.load(resp)
            ready = [r for r in pool.get("replicas", []) if r.get("readiness")]
            print(
                f"pool: {len(pool.get('replicas', []))} replicas, {len(ready)} ready :: {pool}"
            )
            if len(ready) >= exp.ROLLOUT_MIN_CONTAINERS:
                break
            last = f"only {len(ready)} ready"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        if time.time() > deadline:
            raise TimeoutError(f"front-door readiness timed out: {last}")
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
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        print("completion:", json.dumps(json.load(resp))[:1200])


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
                "  STITCH_SHIM_PROVIDER_DEPLOYMENT=rollout-prod",
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
    last_error = ""
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
        headers={"Content-Type": "application/json", **headers},
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


def _auth_error(headers):
    """Validate the customer auth headers against the provider secret. Returns a
    JSONResponse to reject, or None to allow."""
    from fastapi.responses import JSONResponse

    api_key = os.environ.get("STITCH_SHIM_API_KEY")
    if api_key and headers.get("authorization") != f"Bearer {api_key}":
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    provider_model = os.environ.get("STITCH_SHIM_PROVIDER_MODEL")
    if provider_model and headers.get("provider-model") != provider_model:
        return JSONResponse(
            {"error": "Provider-Model header does not match this deployment"},
            status_code=400,
        )
    provider_deployment = os.environ.get("STITCH_SHIM_PROVIDER_DEPLOYMENT")
    if (
        provider_deployment
        and headers.get("provider-deployment") != provider_deployment
    ):
        return JSONResponse(
            {"error": "Provider-Deployment header does not match this deployment"},
            status_code=400,
        )
    return None


FRONTDOOR_PORT = 8000


@app.server(
    image=image,
    volumes={str(S3_TRANSPORT_MOUNT_PATH): s3_transport_mount},
    secrets=[modal.Secret.from_name(exp.SHIM_SECRET_NAME)],
    min_containers=1,
    max_containers=2,  # singleton: exactly one writer of the `latest` pointer
    nonpreemptible=True,  # keep the sole writer up; a preemption blips the API
    scaledown_window=1,
    region=exp.REGION,  # co-locate the front door with the rollout pool (us)
    routing_region=exp.ROUTING_REGION,  # share the pool's Flash proxy region
    port=FRONTDOOR_PORT,
    unauthenticated=True,  # public customer endpoint; auth is enforced in-app
    target_concurrency=1000,  # one container, many concurrent inputs
    startup_timeout=5 * MINUTES,
    exit_grace_period=25,
    include_source=False,
)
class FrontDoor:
    """Public front door: the single writer of `latest` and the customer
    hot-load API, plus an affinity-relabeling proxy to the rollout gateway.

    Serves the front-door FastAPI app under uvicorn (App.server). No
    `from __future__ import annotations` interplay here — frontdoor_mod's
    create_frontdoor_app resolves its handler annotations against its own
    eager fastapi import.
    """

    @modal.enter()
    def start(self) -> None:
        import threading

        import httpx
        import uvicorn
        from fastapi.responses import Response

        board = FilesystemBulletinBoard(str(S3_TRANSPORT_MOUNT_PATH), layout="slime")
        gateway: dict[str, str | None] = {"url": None}
        clients: dict[str, httpx.AsyncClient] = {}

        def _proxy_client() -> httpx.AsyncClient:
            client = clients.get("client")
            if client is None:
                client = httpx.AsyncClient(timeout=None, trust_env=False)
                clients["client"] = client
            return client

        async def read_current_version() -> int:
            return board.read_latest()

        async def advance_to(version: int) -> None:
            # Singleton writer: a single small write is one atomic S3 PutObject,
            # so no rename dance is needed. Direct write avoids FUSE rename
            # semantics.
            (S3_TRANSPORT_MOUNT_PATH / "latest").write_text(
                f"{int(version):06d}", encoding="utf-8"
            )

        async def list_server_infos() -> list[dict]:
            targets = await asyncio.to_thread(
                discover_flash_targets, APP_NAME, Server.__name__
            )
            infos: list[dict] = []
            async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
                for target in targets:
                    try:
                        resp = await client.get(f"{target}/server_info")
                        infos.append(resp.json())
                    except Exception:  # noqa: BLE001 — unreachable replica reported as not-ready
                        infos.append(
                            {"sync_state": None, "last_sync_error": "unreachable"}
                        )
            return infos

        async def wake(version: int) -> None:
            targets = await asyncio.to_thread(
                discover_flash_targets, APP_NAME, Server.__name__
            )
            await asyncio.to_thread(wake_targets, targets, version)

        async def proxy(request, path: str) -> Response:
            if gateway["url"] is None:
                gateway["url"] = provider_gateway_url()
            headers = _frontdoor_headers(dict(request.headers))
            body = await request.body()
            resp = await _proxy_client().request(
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

        asgi_app = frontdoor_mod.create_frontdoor_app(
            read_current_version=read_current_version,
            advance_to=advance_to,
            list_server_infos=list_server_infos,
            proxy=proxy,
            authorize=_auth_error,
            wake=wake,
        )
        config = uvicorn.Config(
            asgi_app, host="0.0.0.0", port=FRONTDOOR_PORT, log_level="info"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    @modal.exit()
    def stop(self) -> None:
        server = getattr(self, "_server", None)
        if server is not None:
            server.should_exit = True
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=25)


def frontdoor_url() -> str:
    """Advertised provider URL: external clients hit the front door, which owns
    `latest` and relabels affinity pre-gateway."""
    return FrontDoor.get_url().rstrip("/")


def _start_provider_sidecar(*, base_checkpoint_dir: str) -> subprocess.Popen:
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
        "--transport-root",
        str(S3_TRANSPORT_MOUNT_PATH),
        "--local-checkpoint-dir",
        LOCAL_CHECKPOINT_PATH,
        "--base-checkpoint-dir",
        base_checkpoint_dir,
        "--commit-mode",
        exp.COMMIT_MODE,
    ]
    print("Starting provider sidecar:", " ".join(cmd))
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
