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
from cookbook.standalone_rollouts.auth import authorize
from cookbook.standalone_rollouts.base_checkpoint import (
    is_hf_repo_id,
    resolve_base_checkpoint,
)
from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import plain_write_text
from stitch.providers.modal import (
    discover_flash_targets,
    resolve_flash_gateway_url,
    wake_targets,
)


PROVIDER_CONFIG = os.environ.get("PROVIDER_CONFIG", "moonlight_hot_load")
exp = importlib.import_module(f"cookbook.standalone_rollouts.configs.{PROVIDER_CONFIG}")

APP_NAME = exp.APP_NAME
MODEL_NAME = exp.MODEL_NAME
# Required: the checkpoint the engine boots from and every delta seeds onto.
# HF repo id (resolved from the prewarmed cache) or an absolute S3/prep dir.
BASE_CHECKPOINT = exp.BASE_CHECKPOINT
ROLLOUT_CONCURRENCY = exp.ROLLOUT_CONCURRENCY
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
MINUTES = 60
SERVER_STARTUP_TIMEOUT = 35 * MINUTES
LOCAL_CHECKPOINT_PATH = exp.LOCAL_CHECKPOINT_PATH
# Host-local dir holding the weight_vN symlink view of the transport's
# opaque-identity dirs (rebuilt from the front-door ledger before each sync).
DELTA_VIEW_PATH = "/local-delta-view"
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
# The provider sidecar applies disk deltas host-side via slime.utils.disk_delta,
# so the image must carry the fork's slime plus its checksum/compression deps.
# Pin a SHA, not the branch tip: the clone is a cached image layer (see
# cookbook/slime_disagg/modal_train.py).
SLIME_REPO_REF = "11bb0fa48aa37d5c54fe297143c6bc1d40f311bf"

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

# Dev iteration: SLIME_LOCAL_DIR overlays a local slime checkout onto the image's
# cloned fork (installed editable at /root/slime), so fork edits take effect on
# container start with no image rebuild. Unset by default.
if slime_local := os.environ.get("SLIME_LOCAL_DIR"):
    image = image.add_local_dir(
        slime_local,
        remote_path=SLIME_ROOT,
        ignore=[".git", "**/__pycache__", "**/*.pyc"],
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
        from huggingface_hub import snapshot_download

        # Boot the engine from, and seed every delta onto, the one base
        # checkpoint the config names. The engine's initial served weights are
        # the customer's base (not a stock stand-in); the sidecar patches a copy
        # of the same dir per delta and reloads. --served-model-name stays
        # MODEL_NAME, so the wire label is unchanged whatever the load path is.
        base_checkpoint_dir = resolve_base_checkpoint(
            BASE_CHECKPOINT, snapshot_download=snapshot_download
        )
        self.endpoint = SGLangEndpoint(
            model_path=base_checkpoint_dir,
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

    # Prewarm the base into the HF cache volume so the serving path resolves it
    # with local_files_only=True. A path-based base (S3 mount / prep volume) is
    # provisioned out of band, so there is nothing to download.
    if is_hf_repo_id(BASE_CHECKPOINT):
        snapshot_download(repo_id=BASE_CHECKPOINT)
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
                "  STITCH_SHIM_PROVIDER_MODEL=moonlight \\",
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
    JSONResponse to reject, or None to allow. Fail-closed when the API key is
    unset (see :func:`cookbook.standalone_rollouts.auth.authorize`)."""
    from fastapi.responses import JSONResponse

    rejection = authorize(
        headers,
        api_key=os.environ.get("STITCH_SHIM_API_KEY"),
        provider_model=os.environ.get("STITCH_SHIM_PROVIDER_MODEL"),
        provider_deployment=os.environ.get("STITCH_SHIM_PROVIDER_DEPLOYMENT"),
    )
    if rejection is None:
        return None
    status, message = rejection
    return JSONResponse({"error": message}, status_code=status)


FRONTDOOR_PORT = 8000


@app.server(
    image=image,
    volumes={str(S3_TRANSPORT_MOUNT_PATH): s3_transport_mount},
    secrets=[modal.Secret.from_name(exp.SHIM_SECRET_NAME)],
    min_containers=1,
    max_containers=1,  # singleton: exactly one writer of the `latest` pointer
    nonpreemptible=True,  # keep the sole writer up; a preemption blips the API
    scaledown_window=2,
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

        # Fail closed: this in-app check is the only auth gate (the Modal server
        # is unauthenticated=True), so refuse to start open rather than serve a
        # public endpoint with a missing/mis-named secret key.
        if not os.environ.get("STITCH_SHIM_API_KEY"):
            raise RuntimeError(
                "STITCH_SHIM_API_KEY is not set; the front door is the only auth "
                f"gate and will not start open. Set it in the {exp.SHIM_SECRET_NAME} secret."
            )

        import json

        board = FilesystemBulletinBoard(str(S3_TRANSPORT_MOUNT_PATH), layout="slime")
        transport_root = Path(str(S3_TRANSPORT_MOUNT_PATH))
        ledger_path = transport_root / "identities.json"
        gateway: dict[str, str | None] = {"url": None}
        clients: dict[str, httpx.AsyncClient] = {}

        def _proxy_client() -> httpx.AsyncClient:
            client = clients.get("client")
            if client is None:
                client = httpx.AsyncClient(timeout=None, trust_env=False)
                clients["client"] = client
            return client

        async def load_ledger() -> dict:
            def _read() -> dict:
                try:
                    return json.loads(ledger_path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    return {}

            return await asyncio.to_thread(_read)

        async def save_ledger(data: dict) -> None:
            # Rename-free write on the S3 mount (see plain_write_text); the front
            # door is the singleton writer, serialized under the app's advance lock.
            await asyncio.to_thread(
                plain_write_text, ledger_path, json.dumps(data, sort_keys=True) + "\n"
            )

        async def normalize_index(identity: str, metadata: dict) -> None:
            # Merge the disk-delta metadata block into the customer's uploaded
            # index so the decoder can apply it, without asking the customer to
            # produce a slime-shaped index. Raises FileNotFoundError if the upload
            # has not landed (the front door turns that into a 409).
            index_path = transport_root / identity / "model.safetensors.index.json"

            def _rewrite() -> None:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                index.setdefault("metadata", {}).update(metadata)
                plain_write_text(index_path, json.dumps(index))

            await asyncio.to_thread(_rewrite)

        async def advance_to(version: int) -> None:
            # Run-less pointer: write the bare weight_vN identity the pool pulls.
            # One plain PutObject-style overwrite (see plain_write_text).
            board.write_latest(None, version)

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
            load_ledger=load_ledger,
            save_ledger=save_ledger,
            normalize_index=normalize_index,
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
        # Customer uploads to opaque-identity dirs; resolve weight_vN through a
        # host-local symlink view into the mount (see provider.build_manager).
        "--delta-view-dir",
        DELTA_VIEW_PATH,
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
