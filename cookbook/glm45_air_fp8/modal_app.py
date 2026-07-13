"""Disaggregated GLM-4.5-Air training on Modal, assembled on the stitch core.

A Modal Flash pool of sglang servers handles rollouts; a clustered Trainer runs miles
on Ray and publishes XOR weight deltas through a Modal Volume the pool syncs from. The
stitch pieces are small and live elsewhere: the Server runs ``stitch.service.serve``
over a ModalVolumeStore + SGLangEngine (sidecar.py), and the Trainer wires miles' three
plug points to the core (hooks.py). Everything here is the Modal/miles/Ray deployment.

    uv run --extra modal modal deploy -m cookbook.glm45_air_fp8.modal_app
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import modal
import modal.experimental

from stitch.pools.modal_flash import ModalFlashPool

from . import config, infra
from .config import CHECKPOINTS_PATH, DATA_PATH, HF_CACHE_PATH, PREP_PATH, MilesConfig

modal_cfg = config.modal
miles_cfg = config.miles

# The pool warm-boots min_containers the moment the app is materialized (deploy OR any
# `modal run`); those replicas serve the prepared FP8 base, so prepare_checkpoints must
# run with the pool down: POOL_MIN_CONTAINERS=0.
POOL_MIN_CONTAINERS = int(os.environ.get("POOL_MIN_CONTAINERS", modal_cfg.rollout_min_containers))

APP_NAME = config.APP_NAME
MODEL_NAME = miles_cfg.hf_checkpoint
ROLLOUT_CONCURRENCY = modal_cfg.rollout_target_inputs or miles_cfg.sglang_server_concurrency
N_TRAIN_NODES = infra.training_nodes(miles_cfg)

MINUTES = 60
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
RAY_PORT = 6379
SERVER_STARTUP_TIMEOUT = 35 * MINUTES

MILES_IMAGE_TAG = "radixark/miles:dev-202607090055"  # dated tag, never `latest` (Modal caches per tag string)
MILES_ROOT = "/root/miles"
MEGATRON_PATH = "/root/Megatron-LM"  # source-only megatron.training must be on PYTHONPATH
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "bdaf8d04bbc26981ff4bc95b8cfe679c3cd29013"

TORCH_DIST_CONVERT_WRAPPER = "/root/convert_hf_to_torch_dist_modal.py"
RECIPE_DIR = "/root/cookbook/glm45_air_fp8"  # mount point; /root is on PYTHONPATH

# Pinned weight-sync sglang: the base supplies kernels/CUDA; the fork carries
# /pull_weights, the hardened local_checkpoint receiver, and the quantized-reload
# restore protocol (reload == init). No trainer package is installed on this image.
SGLANG_IMAGE_TAG = "lmsysorg/sglang:v0.5.14"
SGLANG_FORK_REPO = "https://github.com/modal-projects/sglang.git"
SGLANG_FORK_BRANCH = "weight-sync-miles"
SGLANG_FORK_COMMIT = "2347c32817479e0521ac578230604fa4bbdc6cea"


def _mount_example(image: modal.Image) -> modal.Image:
    """Mount stitch + this example so the trainer, Ray actors, and the sidecar
    subprocess (`python -m cookbook.glm45_air_fp8.sidecar`) all resolve their imports.
    Mounted at container start, so code edits never rebuild the image."""
    return image.add_local_python_source("stitch").add_local_dir(
        Path(__file__).parent, remote_path=RECIPE_DIR, ignore=["**/__pycache__"]
    )


# ── Trainer image (miles + Megatron + TE) ────────────────────────────────────
image = (
    modal.Image.from_registry(MILES_IMAGE_TAG)
    .entrypoint([])
    # RDMA/EFA userspace so multi-node NCCL binds EFA under rdma=True instead of TCP.
    .apt_install("libibverbs-dev", "libibverbs1", "libhwloc-dev", "libnl-route-3-200")
    .run_commands(f"rm -rf {HF_CACHE_PATH}")  # baked HF cache must not shadow the mounted volume
    .run_commands(
        f"rm -rf {MILES_ROOT}"
        f" && git clone {MILES_REPO_URL} {MILES_ROOT}"
        f" && cd {MILES_ROOT} && git fetch origin {MILES_REPO_REF} && git checkout FETCH_HEAD"
        f" && python3 -m pip install --no-deps -e {MILES_ROOT}"
    )
    # The trainer-side delta ENCODER (miles delta.py) needs the codecs even under --no-deps.
    .pip_install("fastapi", "httpx", "uvicorn", "zstandard", "xxhash", "blake3")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(str(Path(__file__).parent / "convert_hf_to_torch_dist_modal.py"), TORCH_DIST_CONVERT_WRAPPER, copy=True)
)
image = _mount_example(image)
# Dev overlay: MILES_LOCAL_DIR replaces the cloned fork so miles edits take effect on
# container start with no image rebuild.
if miles_local := os.environ.get("MILES_LOCAL_DIR"):
    image = image.add_local_dir(miles_local, remote_path=MILES_ROOT, ignore=[".git", "**/__pycache__", "**/*.pyc"])


# ── Server (rollout) image (weight-sync sglang, no trainer package) ──────────
server_image = (
    modal.Image.from_registry(SGLANG_IMAGE_TAG)
    .run_commands(
        f"cd /sgl-workspace/sglang && git remote add modal-fork {SGLANG_FORK_REPO}"
        f" && git fetch modal-fork {SGLANG_FORK_BRANCH} && git checkout {SGLANG_FORK_COMMIT} -- python/"
    )
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    .pip_install(
        "autoinference-utils==0.2.0",  # sglang server lifecycle
        "fastapi", "httpx", "uvicorn",  # the stitch sidecar
        "zstandard", "xxhash", "blake3",  # engine-side /pull_weights receiver's codecs
    )
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_DISABLE_CUDNN_CHECK": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_TIMEOUT_KEEP_ALIVE": "300",
        "SGLANG_ENABLE_RELOAD_LOAD_PLAN": "0",  # record path hangs the GLM fused-MoE loader; keep off
        # Read by the engine's /pull_weights pre-read hook and the sidecar's Store refresh.
        "DELTA_VOLUME_NAME": config.DELTA_VOLUME_NAME,
    })
    # The kernel-cache volume mounts at /root/.cache/sglang, which can't mount over a
    # non-empty path — clear it as the final filesystem step (repopulated on first boot).
    .run_commands("rm -rf /root/.cache/sglang")
)
server_image = _mount_example(server_image)
if miles_local:
    server_image = server_image.add_local_dir(miles_local, remote_path=MILES_ROOT, ignore=[".git", "**/__pycache__", "**/*.pyc"])

with server_image.imports():
    from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions


# ── Volumes ──────────────────────────────────────────────────────────────────
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("miles-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("miles-checkpoints", create_if_missing=True)
prep_volume = modal.Volume.from_name("miles-prep-checkpoints", create_if_missing=True)
sglang_cache_volume = modal.Volume.from_name("miles-sglang-cache", create_if_missing=True)  # survives cold starts
delta_volume = modal.Volume.from_name(config.DELTA_VOLUME_NAME, create_if_missing=True, version=2)

SGLANG_CACHE_PATH = "/root/.cache/sglang"
train_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
    str(PREP_PATH): prep_volume,
    config.DELTA_BULLETIN_ROOT: delta_volume,
}

app = modal.App(APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": MODEL_NAME,
    "--cuda-graph-max-bs": str(ROLLOUT_CONCURRENCY),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    # The engine reads published versions off the Modal Volume; this hook reloads it
    # first (object stores lack cross-host read-after-write consistency).
    "--custom-pull-weights-pre-read-hook": "stitch.stores.modal_volume.pull_weights_pre_read_hook",
    **config.SGLANG_SERVER_ARGS,
}

WARMUP_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}


@app.cls(
    image=server_image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.rollout_num_gpus_per_engine}",
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(PREP_PATH): prep_volume,
        SGLANG_CACHE_PATH: sglang_cache_volume,
        config.DELTA_BULLETIN_ROOT: delta_volume,
    },
    min_containers=POOL_MIN_CONTAINERS,
    max_containers=modal_cfg.rollout_max_containers,
    timeout=40 * MINUTES,
    scaledown_window=15 * MINUTES,
    ephemeral_disk=modal_cfg.rollout_ephemeral_disk_mib,
    memory=modal_cfg.rollout_memory_mib,
    include_source=False,
)
@modal.experimental.http_server(
    port=SIDECAR_PORT, proxy_regions=modal_cfg.proxy_regions,
    exit_grace_period=25, startup_timeout=SERVER_STARTUP_TIMEOUT,
)
@modal.concurrent(target_inputs=ROLLOUT_CONCURRENCY)
class Server:
    """One sglang rollout server plus the stitch versioned-proxy sidecar. The public
    container port is the sidecar (it fronts the private sglang on SGLANG_PORT)."""

    @modal.enter()
    def startup(self) -> None:
        self.endpoint = SGLangEndpoint(
            model_path=MODEL_NAME,
            worker_port=SGLANG_PORT,
            tp=miles_cfg.rollout_num_gpus_per_engine,
            extra_server_args=SGLANG_SERVER_ARGS,
            health_timeout=SERVER_STARTUP_TIMEOUT,
            health_poll_interval=10.0,
        )
        self.endpoint.start()
        warmup_chat_completions(port=SGLANG_PORT, payload=WARMUP_PAYLOAD, successful_requests=2,
                                request_timeout=120.0, max_attempts_per_request=3)
        # The engine serves MODEL_NAME and materializes each version into
        # LOCAL_CHECKPOINT_PATH itself via /pull_weights; the sidecar drives the sync.
        self.sidecar = infra.start_sidecar(
            sidecar_port=SIDECAR_PORT, sglang_port=SGLANG_PORT,
            bulletin_root=config.DELTA_BULLETIN_ROOT,
            local_checkpoint_dir=config.LOCAL_CHECKPOINT_PATH,
            volume_name=config.DELTA_VOLUME_NAME,
            commit_mode=config.SIDECAR_COMMIT_MODE,
        )
        infra.wait_http(f"http://127.0.0.1:{SIDECAR_PORT}/health", self.sidecar, SERVER_STARTUP_TIMEOUT)
        print(f"Rollout server ready: model={MODEL_NAME}, target_inputs={ROLLOUT_CONCURRENCY}")

    @modal.exit()
    def stop(self) -> None:
        infra.terminate_process(getattr(self, "sidecar", None))
        if hasattr(self, "endpoint"):
            self.endpoint.stop()


# ── Trainer (miles on Ray) ────────────────────────────────────────────────────
# Modal requires clustered B200 functions to use all 8 GPUs/node, so a single-node
# trainer runs as a plain cls; clustering (RDMA + EFA) is applied only when >1 node.
_TRAINER_KWARGS = dict(
    image=image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.memory,
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes=train_volumes,
    ephemeral_disk=modal_cfg.trainer_ephemeral_disk_mib,
    timeout=24 * 60 * MINUTES,
    startup_timeout=20 * MINUTES,
    scaledown_window=30 * MINUTES,
    include_source=False,
)
if N_TRAIN_NODES > 1:
    _TRAINER_KWARGS["experimental_options"] = {"efa_enabled": True}


class Trainer:
    """miles actor cluster. Ray comes up once per container in enter(), so back-to-back
    runs reuse it."""

    @modal.enter()
    def start_ray(self) -> None:
        rank, master_addr, my_ip = infra.get_modal_cluster_context(N_TRAIN_NODES)
        # Megatron is editable-installed from the image checkout; patch on-disk before Ray.
        infra.apply_git_patches(config.MEGATRON_RUNTIME_PATCHES, MEGATRON_PATH, "Megatron patch")
        self.rank = rank
        infra.start_host_mem_monitor()  # per-node host-RAM trace (publish gather is the OOM peak)
        os.environ.update({
            "MILES_HOST_IP": my_ip,
            "SGLANG_HOST_IP": my_ip,
            "HOST_IP": my_ip,
            "MASTER_ADDR": master_addr,
            "RAY_ADDRESS": f"{master_addr}:{RAY_PORT}",
            "no_proxy": f"127.0.0.1,{master_addr},{my_ip}",
            "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
            "PYTHONPATH": f"{MEGATRON_PATH}:{os.environ.get('PYTHONPATH', '')}",  # source-only megatron.training
            **miles_cfg.environment,
        })
        if rank == 0:
            infra.start_ray_head(my_ip, N_TRAIN_NODES, ray_port=RAY_PORT)
        else:
            infra.start_ray_worker(my_ip, master_addr, ray_port=RAY_PORT)

    @modal.method()
    def train(self, payload: dict) -> None:
        """Run one training job from a MilesConfig payload (see MilesConfig.to_payload)."""
        for volume in train_volumes.values():
            volume.reload()

        cfg = MilesConfig.from_payload(payload)
        # te_precision_config_file is re-read on every Ray actor, so it must exist at an
        # identical local path on all nodes. Materialize before the rank-0 gate (SPMD).
        infra.materialize_node_local_yaml(cfg, "te_precision_config_file")
        if self.rank != 0:
            return

        cfg.rollout_endpoint_url = ModalFlashPool(APP_NAME, Server.__name__).gateway_url()
        # Fresh run id per launch: this run's chain lives under <root>/<run_id>/, while the
        # canonical pointer at <root>/latest is self-identifying, so a new run is a forward
        # move, never a colliding rewind.
        run_id = uuid.uuid4().hex[:12]
        cfg.update_weight_disk_dir = f"{config.DELTA_BULLETIN_ROOT}/{run_id}"
        if getattr(cfg, "save_interval", None) is None:
            cfg.load = cfg.save = cfg.save_hf = None
        else:
            cfg.load = cfg.save = f"{CHECKPOINTS_PATH}/{run_id}/checkpoints"
        # Merge the run's bulletin identity into custom_config_path (already carrying the
        # request-gating knobs); miles setattr's every key onto args for the hooks.
        custom_config = {
            **dict(getattr(cfg, "custom_config_path", {}) or {}),
            "update_weight_delta_volume_name": config.DELTA_VOLUME_NAME,
            "rollout_modal_flash_app_name": APP_NAME,
            "rollout_modal_flash_server_cls_name": Server.__name__,
            "run_id": run_id,
        }
        cfg.custom_config_path = custom_config
        infra.prepare_config(cfg, tempfile.mkdtemp(), config.YAML_CONFIG_FIELDS)
        cmd = infra.build_train_cmd(cfg, MILES_ROOT)

        # Claim the pool before miles publishes: reset every replica to base for this run.
        from . import hooks

        hooks.claim_pool(SimpleNamespace(update_weight_disk_dir=cfg.update_weight_disk_dir, **custom_config))

        print(f"Training: nodes={N_TRAIN_NODES}, rollout_endpoint={cfg.rollout_endpoint_url}")
        print(f"Command: {cmd}")
        # Tee to a committed Volume log so failures survive the app-logs buffer.
        log_path = f"{CHECKPOINTS_PATH}/{run_id}/train.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        teed = f"set -o pipefail; ({cmd}) 2>&1 | tee {log_path}"
        try:
            subprocess.run(["bash", "-lc", teed], check=True)
        finally:
            try:
                checkpoints_volume.commit()
                print(f"Train log committed to miles-checkpoints at {run_id}/train.log")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not commit train log: {exc}")


if N_TRAIN_NODES > 1:
    Trainer = modal.experimental.clustered(N_TRAIN_NODES, rdma=True)(Trainer)
Trainer = app.cls(**_TRAINER_KWARGS)(Trainer)


# ── Preparation ────────────────────────────────────────────────────────────────
@app.function(
    image=image, gpu=f"{modal_cfg.gpu}:1",
    volumes={str(HF_CACHE_PATH): hf_cache_volume, str(PREP_PATH): prep_volume},
    timeout=6 * 60 * MINUTES, secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
def prepare_checkpoints() -> None:
    """Build the bf16 masters (trainer arch source) + the served FP8 base.

    masters = the bf16 SOURCE_MODEL (dereferenced, stale quant config stripped);
    served base = the published ROLLOUT_SOURCE_MODEL (native HF FP8)."""
    os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)  # the standard downloader is the reliable path for this model
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    from huggingface_hub import snapshot_download

    prep_volume.reload()
    tag = config.MODEL_TAG
    bf16_dir, fp8_dir = f"{PREP_PATH}/{tag}/bf16", f"{PREP_PATH}/{tag}/fp8"

    def _build_bf16(out: str) -> None:
        # bf16 source IS the masters; -L dereferences so the prep dir holds real files.
        subprocess.run(f"cp -aL {snapshot_download(config.SOURCE_MODEL)}/. {out}/", shell=True, check=True)
        _strip_stale_quant_config(os.path.join(out, "config.json"))

    _staged(bf16_dir, _build_bf16)
    _staged(fp8_dir, lambda out: subprocess.run(
        f"cp -aL {snapshot_download(config.ROLLOUT_SOURCE_MODEL)}/. {out}/", shell=True, check=True))
    prep_volume.commit()
    print(f"Prepared masters={bf16_dir} served_base={fp8_dir}")


def _staged(final_dir: str, build) -> None:
    """Build into a .partial sibling and atomically rename, so an interrupted step
    never leaves a half-built dir the reuse check mistakes for complete."""
    if os.path.isdir(final_dir) and os.listdir(final_dir):
        print(f"reusing existing {final_dir}")
        return
    partial = f"{final_dir}.partial"
    subprocess.run(["rm", "-rf", partial], check=True)
    os.makedirs(partial, exist_ok=True)
    build(partial)
    os.rename(partial, final_dir)


def _strip_stale_quant_config(config_path: str) -> None:
    """Drop any quantization_config (top-level and text_config-nested) from an HF config,
    so the bf16 masters don't claim the source's quant scheme."""
    import json

    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        cfg = json.load(f)
    removed = bool(cfg.pop("quantization_config", None))
    if isinstance(cfg.get("text_config"), dict):
        removed = bool(cfg["text_config"].pop("quantization_config", None)) or removed
    if removed:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"stripped stale quantization_config from {config_path}")


TORCH_DIST_PREP_NODES = modal_cfg.torch_dist_prep_nodes


def prepare_torch_dist() -> None:
    """Build {tag}/torch_dist (the raw-mode ref_load) from the {tag}/bf16 masters via a
    clustered torchrun conversion (the model won't fit an 8-way split)."""
    rank, master_addr, _my_ip = infra.get_modal_cluster_context(TORCH_DIST_PREP_NODES)
    prep_volume.reload()
    tag = config.MODEL_TAG
    bf16_dir, torch_dist_dir = f"{PREP_PATH}/{tag}/bf16", f"{PREP_PATH}/{tag}/torch_dist"
    if os.path.exists(os.path.join(torch_dist_dir, "latest_checkpointed_iteration.txt")):
        print(f"reusing existing torch_dist {torch_dist_dir}")
        return
    inner = (
        f"source {MILES_ROOT}/{miles_cfg.miles_model_script} && "
        f"PYTHONPATH={MEGATRON_PATH} torchrun"
        f" --nnodes {TORCH_DIST_PREP_NODES} --node-rank {rank}"
        f" --master-addr {master_addr} --master-port 29500"
        f" --nproc-per-node {modal_cfg.torch_dist_prep_gpus_per_node}"
        f" {TORCH_DIST_CONVERT_WRAPPER} ${{MODEL_ARGS[@]}}"
        f" --hf-checkpoint {bf16_dir} --save {torch_dist_dir} --megatron-to-hf-mode raw"
        f" {modal_cfg.torch_dist_convert_extra_args}"
    )
    env = {**os.environ, "SKIP_RELEASE_RENAME": "1"}
    subprocess.run(["bash", "-c", inner], check=True, env=env)
    # Every node commits its own distcp shards (disjoint files merge on the Volume);
    # a rank-0-only commit would drop the other nodes' shards.
    prep_volume.commit()
    if rank == 0:
        print(f"Prepared torch_dist={torch_dist_dir}")


_torch_dist_kwargs = dict(
    image=image, gpu=f"{modal_cfg.gpu}:{modal_cfg.torch_dist_prep_gpus_per_node}",
    volumes={str(HF_CACHE_PATH): hf_cache_volume, str(PREP_PATH): prep_volume},
    timeout=6 * 60 * MINUTES,
    ephemeral_disk=(modal_cfg.torch_dist_prep_ephemeral_disk_mib or modal_cfg.rollout_ephemeral_disk_mib),
    secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
if TORCH_DIST_PREP_NODES > 1:
    prepare_torch_dist = modal.experimental.clustered(TORCH_DIST_PREP_NODES, rdma=True)(prepare_torch_dist)
    _torch_dist_kwargs["experimental_options"] = {"efa_enabled": True}
prepare_torch_dist = app.function(**_torch_dist_kwargs)(prepare_torch_dist)


@app.function(
    image=image, volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * MINUTES, secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
def prepare_dataset() -> None:
    data_volume.reload()
    miles_cfg.prepare_data()
    data_volume.commit()


# ── Entrypoints ──────────────────────────────────────────────────────────────
@app.local_entrypoint()
def launch_train() -> None:
    """Spawn training on the deployed app. Config ships as data, so edits run without a
    redeploy; infrastructure changes still require one."""
    from modal.exception import NotFoundError

    try:
        trainer = modal.Cls.from_name(APP_NAME, Trainer.__name__)()
        call = trainer.train.spawn(miles_cfg.to_payload())
    except NotFoundError:
        raise SystemExit(
            f"App {APP_NAME!r} is not deployed. Run:\n"
            f"  uv run --extra modal modal deploy -m cookbook.glm45_air_fp8.modal_app"
        )
    print(f"Spawned train on {APP_NAME}: {call.object_id}")


@app.local_entrypoint()
def smoke_flash_pool(weight_version: int = 0, timeout_seconds: int = 30 * MINUTES) -> None:
    """Check the deployed Flash pool serves completions at the expected weight version."""
    infra.smoke_flash_pool(
        app_name=APP_NAME, cls_name=Server.__name__, model_name=MODEL_NAME,
        weight_version=weight_version, timeout_seconds=timeout_seconds,
    )
