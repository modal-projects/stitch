"""Disaggregated slime training on Modal, assembled on the stitch core.

``EXPERIMENT_CONFIG`` selects a config module under ``cookbook.slime_disagg.configs``. The
Server (sglang + stitch sidecar) is the shared common one; the Trainer runs slime on Ray
and publishes XOR deltas through a Modal Volume the pool syncs from.

    EXPERIMENT_CONFIG=kimi_k2_6_int4 uv run --extra modal modal deploy -m cookbook.slime_disagg.app

Config access is uniform: the experiment module ``exp`` is the single source of truth —
its ``exp.modal`` (infra), ``exp.slime`` (training), and ``exp.<CONST>`` are read directly;
shared deployment constants come from ``common.constants``. ``ROLLOUT_CONCURRENCY`` is the
one resolved value (the experiment's Flash target, else the engine's concurrency).
"""

from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
import uuid
from types import SimpleNamespace

import modal
import modal.experimental

from stitch.pools.modal_flash import ModalFlashPool

from cookbook.common import launch, ray_cluster, serving_image, server, smoke
from cookbook.common.constants import (
    CHECKPOINTS_PATH, DATA_PATH, HF_CACHE_PATH, MINUTES, RAY_PORT, SERVER_STARTUP_TIMEOUT, SIDECAR_PORT,
)
from cookbook.slime_disagg import trainer_image
from cookbook.slime_disagg.config import SlimeConfig, YAML_CONFIG_FIELDS
from cookbook.slime_disagg.trainer_image import SLIME_ROOT

# Deploy-time environment (selection + dev overlay, NOT experiment config).
EXPERIMENT = os.environ.get("EXPERIMENT_CONFIG", "kimi_k2_6_int4")
SLIME_LOCAL_DIR = os.environ.get("SLIME_LOCAL_DIR")  # dev overlay of a local slime checkout

exp = importlib.import_module(f"cookbook.slime_disagg.configs.{EXPERIMENT}")
modal_cfg = exp.modal
slime_cfg = exp.slime

# The Flash autoscaler target (and sglang concurrency cap): the experiment's explicit
# target_inputs, else the engine's configured concurrency.
ROLLOUT_CONCURRENCY = modal_cfg.rollout_target_inputs or slime_cfg.sglang_server_concurrency

image = trainer_image.build_trainer_image(hf_cache_path=str(HF_CACHE_PATH), slime_local=SLIME_LOCAL_DIR)
server_image = serving_image.build_serving_image(hf_cache_path=str(HF_CACHE_PATH), delta_volume_name=exp.DELTA_VOLUME_NAME)
if SLIME_LOCAL_DIR:
    server_image = server_image.add_local_dir(SLIME_LOCAL_DIR, remote_path=SLIME_ROOT, ignore=[".git", "**/__pycache__", "**/*.pyc"])

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True, version=2)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True, version=2)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=True, version=2)
delta_volume = modal.Volume.from_name(exp.DELTA_VOLUME_NAME, create_if_missing=True, version=2)

train_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
    exp.DELTA_BULLETIN_ROOT: delta_volume,
}

app = modal.App(exp.APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": slime_cfg.hf_checkpoint,
    "--dtype": "bfloat16",
    "--cuda-graph-max-bs": str(ROLLOUT_CONCURRENCY),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    "--custom-pull-weights-pre-read-hook": "stitch.stores.modal_volume.pull_weights_pre_read_hook",
    **exp.SGLANG_SERVER_ARGS,
}


# The rollout Server: a thin module-level class (Modal requires @app.cls at global scope)
# whose enter/exit delegate to the shared common.server logic. sglang + the stitch sidecar.
@app.cls(
    image=server_image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.rollout_num_gpus_per_engine}",
    cloud=modal_cfg.cloud, region=modal_cfg.region,
    volumes={str(HF_CACHE_PATH): hf_cache_volume, exp.DELTA_BULLETIN_ROOT: delta_volume},
    min_containers=modal_cfg.rollout_min_containers, max_containers=modal_cfg.rollout_max_containers,
    timeout=40 * MINUTES, scaledown_window=15 * MINUTES,
    ephemeral_disk=modal_cfg.rollout_ephemeral_disk_mib, memory=modal_cfg.rollout_memory_mib,
    include_source=False,
)
@modal.experimental.http_server(
    port=SIDECAR_PORT, proxy_regions=modal_cfg.proxy_regions,
    exit_grace_period=25, startup_timeout=SERVER_STARTUP_TIMEOUT,
)
@modal.concurrent(target_inputs=ROLLOUT_CONCURRENCY)
class Server:
    @modal.enter()
    def startup(self) -> None:
        server.serve_startup(
            self, model_name=slime_cfg.hf_checkpoint, sglang_args=SGLANG_SERVER_ARGS,
            tp=slime_cfg.rollout_num_gpus_per_engine, concurrency=ROLLOUT_CONCURRENCY,
            bulletin_root=exp.DELTA_BULLETIN_ROOT, local_checkpoint_dir=exp.LOCAL_CHECKPOINT_PATH,
            volume_name=exp.DELTA_VOLUME_NAME, commit_mode=exp.SIDECAR_COMMIT_MODE,
            startup_timeout=SERVER_STARTUP_TIMEOUT, sglang_env=getattr(exp, "SGLANG_ENV", {}),
        )

    @modal.exit()
    def stop(self) -> None:
        server.serve_stop(self)


# ── Trainer (slime on Ray) ────────────────────────────────────────────────────
_TRAINER_KWARGS = dict(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.memory,
    cloud=modal_cfg.cloud, region=modal_cfg.region,
    volumes=train_volumes,
    ephemeral_disk=modal_cfg.trainer_ephemeral_disk_mib,
    timeout=24 * 60 * MINUTES, startup_timeout=20 * MINUTES, scaledown_window=30 * MINUTES,
    include_source=False,
)
if slime_cfg.n_train_nodes > 1:
    _TRAINER_KWARGS["experimental_options"] = {"efa_enabled": True}


class Trainer:
    """slime actor cluster. Ray comes up once per container in enter(), so back-to-back
    runs reuse it."""

    @modal.enter()
    def start_ray(self) -> None:
        from cookbook.common import process

        rank, master_addr, my_ip = ray_cluster.get_modal_cluster_context(slime_cfg.n_train_nodes)
        self.rank = rank
        process.start_host_mem_monitor()  # per-node host-RAM trace (publish gather is the OOM peak)
        os.environ.update({
            "SLIME_HOST_IP": my_ip, "SGLANG_HOST_IP": my_ip, "HOST_IP": my_ip,
            "MASTER_ADDR": master_addr, "RAY_ADDRESS": f"{master_addr}:{RAY_PORT}",
            "no_proxy": f"127.0.0.1,{master_addr},{my_ip}", "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
            **slime_cfg.environment,
        })
        if rank == 0:
            ray_cluster.start_ray_head(my_ip, slime_cfg.n_train_nodes, ray_port=RAY_PORT)
        else:
            ray_cluster.start_ray_worker(my_ip, master_addr, ray_port=RAY_PORT)

    @modal.method()
    def train(self, payload: dict) -> None:
        """Run one training job from a SlimeConfig payload (see SlimeConfig.to_payload)."""
        for volume in train_volumes.values():
            volume.reload()
        if self.rank != 0:  # rank 0 drives; other ranks only need their Ray workers (started in enter)
            return

        cfg = SlimeConfig.from_payload(payload)
        cfg.rollout_endpoint_url = ModalFlashPool(exp.APP_NAME, "Server").gateway_url()
        run_id = uuid.uuid4().hex[:12]  # per-launch fence token; forks a fresh chain
        cfg.update_weight_disk_dir = f"{exp.DELTA_BULLETIN_ROOT}/{run_id}"
        hook_knobs = {
            "update_weight_delta_volume_name": exp.DELTA_VOLUME_NAME,
            "rollout_modal_flash_app_name": exp.APP_NAME,
            "rollout_modal_flash_server_cls_name": "Server",
            "run_id": run_id,
        }
        cfg.custom_config_path = hook_knobs  # materialized to a YAML path below; keep the mapping for claim
        launch.resolve_config(cfg, tempfile.mkdtemp(), YAML_CONFIG_FIELDS)
        cmd = launch.build_train_cmd(cfg, SLIME_ROOT, "slime_model_script")

        # Claim the pool before slime publishes: reset every replica to base for this run.
        from cookbook.common import hooks

        hooks.claim_pool(SimpleNamespace(update_weight_disk_dir=cfg.update_weight_disk_dir, **hook_knobs))

        print(f"Training {EXPERIMENT}: nodes={slime_cfg.n_train_nodes}, rollout_endpoint={cfg.rollout_endpoint_url}")
        print(f"Command: {cmd}")
        subprocess.run(["bash", "-lc", cmd], check=True)


if slime_cfg.n_train_nodes > 1:
    Trainer = modal.experimental.clustered(slime_cfg.n_train_nodes, rdma=True)(Trainer)
Trainer = app.cls(**_TRAINER_KWARGS)(Trainer)


# ── Preparation + entrypoints ──────────────────────────────────────────────────
@app.function(
    image=image, volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=2 * 60 * MINUTES, secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
def download_model() -> None:
    """Snapshot the served model into the HF cache (sglang serves it by repo id)."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=slime_cfg.hf_checkpoint)
    hf_cache_volume.commit()


@app.function(
    image=image, volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * MINUTES, secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
def prepare_dataset() -> None:
    data_volume.reload()
    slime_cfg.prepare_data()
    data_volume.commit()


@app.local_entrypoint()
def launch_train() -> None:
    """Spawn training on the deployed app. Config ships as data, so edits run without a
    redeploy; infrastructure changes still require one."""
    from modal.exception import NotFoundError

    try:
        trainer = modal.Cls.from_name(exp.APP_NAME, "Trainer")()
        call = trainer.train.spawn(slime_cfg.to_payload())
    except NotFoundError:
        raise SystemExit(
            f"App {exp.APP_NAME!r} is not deployed. Run:\n"
            f"  EXPERIMENT_CONFIG={EXPERIMENT} uv run --extra modal modal deploy -m cookbook.slime_disagg.app"
        )
    print(f"Spawned train on {exp.APP_NAME}: {call.object_id}")


@app.local_entrypoint()
def smoke_flash_pool(weight_version: int = 0, timeout_seconds: int = 30 * MINUTES) -> None:
    """Check the deployed Flash pool serves completions at the expected weight version."""
    smoke.smoke_flash_pool(
        app_name=exp.APP_NAME, cls_name="Server", model_name=slime_cfg.hf_checkpoint,
        weight_version=weight_version, timeout_seconds=timeout_seconds,
    )
