"""Modal Flash sparse-delta SLIME example.

Deploys a Modal Flash SGLang pool with a weight-version sidecar, then runs a
SLIME actor-only Ray training job that publishes sparse deltas through a v2
Modal Volume bulletin board.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import modal
import modal.experimental


EXAMPLE_ROOT = Path(__file__).resolve().parent
COOKBOOK_ROOT = EXAMPLE_ROOT.parent
VENDOR_ROOT = EXAMPLE_ROOT / "vendor"
STITCH_PACKAGE = Path(os.getenv("STITCH_REPO_PATH", COOKBOOK_ROOT.parent))
STITCH_SRC = STITCH_PACKAGE / "src"
for local_path in (COOKBOOK_ROOT, VENDOR_ROOT, STITCH_SRC):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

from configs.base import CHECKPOINTS_PATH, DATA_PATH, HF_CACHE_PATH  # noqa: E402
from slime_disagg.modal_helpers import (  # noqa: E402
    ensure_pythonpath,
    redact_command_for_log,
    reset_bulletin_board as reset_bulletin_board_impl,
    run_flash_pool_smoke as run_flash_pool_smoke_impl,
    spawn_train_from_deployed_or_ephemeral,
    start_sglang_sidecar,
    terminate_process,
    training_nodes,
    wait_http,
)
from stitch.providers.modal import (  # noqa: E402
    discover_flash_targets,
    resolve_flash_gateway_url,
    resolve_flash_gateway_url_aio,
)


SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"
SLIME_ROOT = "/root/slime"
SLIME_REPO_URL = os.getenv("SLIME_REPO_URL", "https://github.com/modal-projects/slime.git")
SLIME_REPO_REF = os.getenv("SLIME_REPO_REF", "jvmncs/rollout-endpoint")
DEFAULT_EXPERIMENT = os.getenv("EXPERIMENT_CONFIG", "qwen3_4b_delta_flash")
DEFAULT_APP_NAME = "slime-qwen3-4b-delta-flash"
SERVER_CLS_NAME = "Server"

MINUTES = 60
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
RAY_PORT = 6379


def _load_experiment(name: str) -> ModuleType:
    if not name:
        name = DEFAULT_EXPERIMENT
    module_name = name if "." in name else f"slime_disagg.configs.{name}"
    return importlib.import_module(module_name)


def _experiment_app_name(exp: ModuleType) -> str:
    return os.getenv("SLIME_DELTA_APP_NAME", getattr(exp, "APP_NAME", DEFAULT_APP_NAME))


exp_mod = _load_experiment(DEFAULT_EXPERIMENT)
modal_cfg = exp_mod.modal
slime_cfg = exp_mod.slime

APP_NAME = _experiment_app_name(exp_mod)
MODEL_NAME = slime_cfg.hf_checkpoint
DELTA_VOLUME_NAME = getattr(exp_mod, "DELTA_VOLUME_NAME", "slime-delta-bulletin")
DELTA_BULLETIN_ROOT = getattr(exp_mod, "DELTA_BULLETIN_ROOT", "/delta-bulletin")
DELTA_VERSION_DIR = getattr(exp_mod, "DELTA_VERSION_DIR", f"{DELTA_BULLETIN_ROOT}/versions")

ROLLOUT_N_GPUS = int(os.getenv("ROLLOUT_N_GPUS", str(getattr(slime_cfg, "rollout_num_gpus_per_engine", 1))))
ROLLOUT_GPU_TYPE = os.getenv("ROLLOUT_GPU", modal_cfg.gpu)
ROLLOUT_GPU = f"{ROLLOUT_GPU_TYPE}:{ROLLOUT_N_GPUS}" if ROLLOUT_N_GPUS > 1 else ROLLOUT_GPU_TYPE
MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "2"))
TARGET_INPUTS = int(os.getenv("TARGET_INPUTS", str(getattr(slime_cfg, "sglang_server_concurrency", 64))))
SCALEDOWN_WINDOW = int(os.getenv("SCALEDOWN_WINDOW_SECONDS", str(15 * MINUTES)))
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT_SECONDS", str(35 * MINUTES)))
PROXY_REGIONS = [x for x in os.getenv("PROXY_REGIONS", "us-east").split(",") if x]

HF_IMAGE_ENV = {
    "EXPERIMENT_CONFIG": DEFAULT_EXPERIMENT,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "PYTHONPATH": "/root:/root/Megatron-LM/",
}
if "SLIME_DELTA_APP_NAME" in os.environ:
    HF_IMAGE_ENV["SLIME_DELTA_APP_NAME"] = APP_NAME
# in_place needs an sglang build with the overlap-drain fix; default quiesce.
if "SIDECAR_COMMIT_MODE" in os.environ:
    HF_IMAGE_ENV["SIDECAR_COMMIT_MODE"] = os.environ["SIDECAR_COMMIT_MODE"]

SERVER_ARGS = {
    "--served-model-name": MODEL_NAME,
    "--dtype": "bfloat16",
    "--context-length": os.getenv("SGLANG_CONTEXT_LENGTH", "16384"),
    "--mem-fraction-static": os.getenv("SGLANG_MEM_FRACTION_STATIC", "0.84"),
    "--chunked-prefill-size": os.getenv("SGLANG_CHUNKED_PREFILL_SIZE", "4096"),
    "--max-prefill-tokens": os.getenv("SGLANG_MAX_PREFILL_TOKENS", "4096"),
    "--cuda-graph-max-bs": str(TARGET_INPUTS),
    "--max-running-requests": str(TARGET_INPUTS),
    "--trust-remote-code": "",
    "--update-weight-delta-chunk-bytes": str(getattr(slime_cfg, "sglang_update_weight_delta_chunk_bytes", 1024**3)),
    "--update-weight-delta-read-workers": str(getattr(slime_cfg, "sglang_update_weight_delta_read_workers", 8)),
}
SERVER_ARGS.update(getattr(exp_mod, "SGLANG_SERVER_ARGS", {}))

WARMUP_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}

image = (
    modal.Image.from_registry(SLIME_IMAGE_TAG)
    .entrypoint([])
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    .run_commands(
        " && ".join(
            [
                f"rm -rf {shlex.quote(SLIME_ROOT)}",
                f"git clone --depth 1 {shlex.quote(SLIME_REPO_URL)} {shlex.quote(SLIME_ROOT)}",
                f"cd {shlex.quote(SLIME_ROOT)}",
                f"git fetch --depth 1 origin {shlex.quote(SLIME_REPO_REF)}",
                "git checkout FETCH_HEAD",
                f"python3 -m pip install --no-deps -e {shlex.quote(SLIME_ROOT)}",
            ]
        )
    )
    .pip_install(
        "autoinference-utils==0.2.0",
        "editables",
        "fastapi",
        "hatchling",
        "httpx",
        "modal==1.4.1",
        "uvicorn",
        "zstandard",
    )
    .add_local_dir(
        STITCH_PACKAGE,
        remote_path="/root/packages/stitch",
        copy=True,
        ignore=[
            "**/__pycache__",
            "**/*.pyc",
            "**/*_test.py",
            "docs",
            "cookbook",
            ".git",
            ".jj",
            ".venv",
            ".pytest_cache",
        ],
    )
    .run_commands("python3 -m pip install --no-build-isolation --no-deps -e /root/packages/stitch")
    .env(HF_IMAGE_ENV)
    .add_local_dir(
        VENDOR_ROOT / "configs",
        remote_path="/root/configs",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        VENDOR_ROOT / "modal_helpers",
        remote_path="/root/modal_helpers",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        EXAMPLE_ROOT,
        remote_path="/root/slime_disagg",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc", "vendor"],
    )
    .add_local_file(
        EXAMPLE_ROOT / "modal_app.py",
        remote_path="/root/modal_app.py",
        copy=True,
    )
)

with image.imports():
    from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions
    from modal_helpers.utils import (
        build_train_cmd,
        get_modal_cluster_context,
        prepare_slime_config,
        start_ray_head,
    )


hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=True)
delta_volume = modal.Volume.from_name(DELTA_VOLUME_NAME, create_if_missing=True, version=2)

train_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
    DELTA_BULLETIN_ROOT: delta_volume,
}

app = modal.App(APP_NAME)


@app.cls(
    include_source=False,
    image=image,
    gpu=ROLLOUT_GPU,
    cloud=modal_cfg.cloud if modal_cfg.cloud else None,
    region=modal_cfg.region if modal_cfg.region else None,
    volumes={str(HF_CACHE_PATH): hf_cache_volume, DELTA_BULLETIN_ROOT: delta_volume},
    min_containers=MIN_CONTAINERS,
    timeout=40 * MINUTES,
    scaledown_window=SCALEDOWN_WINDOW,
)
@modal.experimental.http_server(
    port=SIDECAR_PORT,
    proxy_regions=PROXY_REGIONS,
    exit_grace_period=25,
    startup_timeout=STARTUP_TIMEOUT,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class Server:
    @modal.enter()
    def startup(self) -> None:
        self.endpoint = SGLangEndpoint(
            model_path=MODEL_NAME,
            worker_port=SGLANG_PORT,
            tp=ROLLOUT_N_GPUS,
            extra_server_args=SERVER_ARGS,
            health_timeout=STARTUP_TIMEOUT,
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
        self.sidecar = start_sglang_sidecar(
            sidecar_port=SIDECAR_PORT,
            sglang_port=SGLANG_PORT,
            bulletin_root=DELTA_BULLETIN_ROOT,
            volume_name=DELTA_VOLUME_NAME,
        )
        wait_http(f"http://127.0.0.1:{SIDECAR_PORT}/health", self.sidecar, STARTUP_TIMEOUT)
        print(
            f"Modal Flash sparse-delta Server ready: model={MODEL_NAME}, "
            f"current_weight_version=0, target_inputs={TARGET_INPUTS}"
        )

    @modal.exit()
    def stop(self) -> None:
        terminate_process(getattr(self, "sidecar", None))
        if hasattr(self, "endpoint"):
            self.endpoint.stop()


@app.local_entrypoint()
def smoke_flash_pool(weight_version: int = 0, expect_min_containers: int = MIN_CONTAINERS) -> None:
    """Hit the Flash gateway and direct container URLs."""
    run_flash_pool_smoke(weight_version=weight_version, expect_min_containers=expect_min_containers)


def run_flash_pool_smoke(
    weight_version: int = 0,
    expect_min_containers: int = MIN_CONTAINERS,
    timeout_seconds: int = STARTUP_TIMEOUT,
) -> None:
    """Hit the deployed Flash gateway and direct container URLs."""
    run_flash_pool_smoke_impl(
        gateway_resolver=_resolve_flash_gateway_url,
        target_discoverer=_discover_flash_targets,
        model_name=MODEL_NAME,
        weight_version=weight_version,
        expect_min_containers=expect_min_containers,
        timeout_seconds=timeout_seconds,
    )


@app.local_entrypoint()
def launch_train(experiment: str = DEFAULT_EXPERIMENT, app_name: str = "") -> None:
    """Spawn training on the deployed app, falling back to this ephemeral app."""
    exp = _load_experiment(experiment)
    target_app_name = app_name or _experiment_app_name(exp)
    spawn_train_from_deployed_or_ephemeral(
        deployed_app_name=target_app_name,
        experiment=experiment,
        fallback_function=train,
    )


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model(experiment: str = DEFAULT_EXPERIMENT) -> None:
    from huggingface_hub import snapshot_download

    cfg = _load_experiment(experiment).slime
    snapshot_download(repo_id=cfg.hf_checkpoint)
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def prepare_dataset(experiment: str = DEFAULT_EXPERIMENT) -> None:
    cfg = _load_experiment(experiment).slime
    data_volume.reload()
    cfg.prepare_data()
    data_volume.commit()


@app.function(
    image=image,
    volumes={DELTA_BULLETIN_ROOT: delta_volume},
    timeout=20 * MINUTES,
)
def reset_bulletin_board(confirm: bool = False) -> None:
    reset_bulletin_board_impl(DELTA_BULLETIN_ROOT, delta_volume, confirm=confirm)
    print(f"Reset {DELTA_VOLUME_NAME} bulletin board to version 0.")


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.memory if modal_cfg.memory else None,
    cloud=modal_cfg.cloud if modal_cfg.cloud else None,
    region=modal_cfg.region if modal_cfg.region else None,
    volumes=train_volumes,
    timeout=24 * 60 * 60,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(training_nodes(slime_cfg), rdma=True)
async def train(experiment: str = DEFAULT_EXPERIMENT) -> None:
    await asyncio.gather(
        hf_cache_volume.reload.aio(),
        data_volume.reload.aio(),
        checkpoints_volume.reload.aio(),
        delta_volume.reload.aio(),
    )

    exp = _load_experiment(experiment)
    cfg = exp.slime
    mcfg = exp.modal
    app_name = _experiment_app_name(exp)
    cfg.rollout_http_endpoint_url = await _resolve_flash_gateway_url_aio(app_name)
    cfg.update_weight_delta_dir = getattr(exp, "DELTA_VERSION_DIR", DELTA_VERSION_DIR)
    cfg.update_weight_delta_root = getattr(exp, "DELTA_BULLETIN_ROOT", DELTA_BULLETIN_ROOT)
    cfg.environment["DELTA_VOLUME_NAME"] = getattr(exp, "DELTA_VOLUME_NAME", DELTA_VOLUME_NAME)
    cfg.environment["SLIME_DELTA_APP_NAME"] = app_name
    cfg.environment["SLIME_DELTA_SERVER_CLS_NAME"] = SERVER_CLS_NAME
    ensure_pythonpath(cfg, "/root", "/root/Megatron-LM/")

    n_nodes = training_nodes(cfg)
    rank, master_addr, my_ip, _ = get_modal_cluster_context(n_nodes)

    os.environ["SLIME_HOST_IP"] = my_ip
    os.environ["SGLANG_HOST_IP"] = my_ip
    os.environ["HOST_IP"] = my_ip
    ray_env_vars = {
        "RAY_ADDRESS": f"{master_addr}:{RAY_PORT}",
        "no_proxy": f"127.0.0.1,{master_addr},{my_ip}",
        "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
        "MASTER_ADDR": master_addr,
        **cfg.environment,
    }
    os.environ.update(ray_env_vars)

    if rank != 0:
        subprocess.run(
            [
                "ray",
                "start",
                f"--node-ip-address={my_ip}",
                "--address",
                f"{master_addr}:{RAY_PORT}",
                "--disable-usage-stats",
            ],
            check=True,
            timeout=int(os.getenv("RAY_WORKER_START_TIMEOUT_SECONDS", "240")),
        )
        while True:
            await asyncio.sleep(10)

    start_ray_head(my_ip, n_nodes, ray_port=RAY_PORT, include_dashboard=False)
    prepare_slime_config(cfg, tempfile.mkdtemp())

    cmd = build_train_cmd(cfg, SLIME_ROOT)

    print(
        f"Training {experiment}: nodes={n_nodes}, gpu={mcfg.gpu}:{cfg.actor_num_gpus_per_node}, "
        f"rollout_endpoint={cfg.rollout_http_endpoint_url}"
    )
    print(f"Command: {redact_command_for_log(cmd)}")

    env = os.environ.copy()
    result = subprocess.run(["bash", "-lc", cmd], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Training command failed with exit code {result.returncode}")


def _resolve_flash_gateway_url(app_name: str = APP_NAME) -> str:
    if url := os.getenv("ROLLOUT_GATEWAY_URL"):
        return url.rstrip("/")
    return resolve_flash_gateway_url(app_name, SERVER_CLS_NAME)


async def _resolve_flash_gateway_url_aio(app_name: str = APP_NAME) -> str:
    if url := os.getenv("ROLLOUT_GATEWAY_URL"):
        return url.rstrip("/")
    return await resolve_flash_gateway_url_aio(app_name, SERVER_CLS_NAME)


def _discover_flash_targets(app_name: str = APP_NAME) -> list[str]:
    return discover_flash_targets(app_name, SERVER_CLS_NAME)
