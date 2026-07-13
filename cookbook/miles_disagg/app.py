"""Disaggregated miles training on Modal, assembled on the stitch core.

``EXPERIMENT_CONFIG`` selects a config module under ``cookbook.miles_disagg.configs``. The
Server (sglang + stitch sidecar) is the shared common one; the Trainer runs miles on Ray
and publishes XOR deltas through a Modal Volume the pool syncs from.

    EXPERIMENT_CONFIG=glm45_air_fp8 uv run --extra modal modal deploy -m cookbook.miles_disagg.app

Config access is uniform: the experiment module ``exp`` is the single source of truth —
its ``exp.modal`` (infra), ``exp.miles`` (training), and ``exp.<CONST>`` are read directly;
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
from typing import Any

import modal
import modal.experimental

from stitch.pools.modal_flash import ModalFlashPool

from cookbook.common import launch, ray_cluster, serving_image, server, smoke
from cookbook.common.constants import (
    CHECKPOINTS_PATH, DATA_PATH, HF_CACHE_PATH, MINUTES, PREP_PATH, RAY_PORT,
    SERVER_STARTUP_TIMEOUT, SGLANG_CACHE_PATH, SIDECAR_PORT,
)
from cookbook.miles_disagg import prep, trainer_image
from cookbook.miles_disagg.config import MilesConfig, YAML_CONFIG_FIELDS
from cookbook.miles_disagg.trainer_image import MEGATRON_PATH, MILES_ROOT

# Deploy-time environment (selection + dev overlay, NOT experiment config).
EXPERIMENT = os.environ.get("EXPERIMENT_CONFIG", "glm45_air_fp8")
MILES_LOCAL_DIR = os.environ.get("MILES_LOCAL_DIR")  # dev overlay of a local miles checkout

exp = importlib.import_module(f"cookbook.miles_disagg.configs.{EXPERIMENT}")
modal_cfg = exp.modal
miles_cfg = exp.miles

# The Flash autoscaler target (and sglang concurrency cap): the experiment's explicit
# target_inputs, else the engine's configured concurrency.
ROLLOUT_CONCURRENCY = modal_cfg.rollout_target_inputs or miles_cfg.sglang_server_concurrency

image = trainer_image.build_trainer_image(hf_cache_path=str(HF_CACHE_PATH), miles_local=MILES_LOCAL_DIR)
server_image = serving_image.build_serving_image(hf_cache_path=str(HF_CACHE_PATH), delta_volume_name=exp.DELTA_VOLUME_NAME)
if MILES_LOCAL_DIR:
    server_image = server_image.add_local_dir(MILES_LOCAL_DIR, remote_path=MILES_ROOT, ignore=[".git", "**/__pycache__", "**/*.pyc"])

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True, version=2)
data_volume = modal.Volume.from_name("miles-data", create_if_missing=True, version=2)
checkpoints_volume = modal.Volume.from_name("miles-checkpoints", create_if_missing=True, version=2)
prep_volume = modal.Volume.from_name("miles-prep-checkpoints", create_if_missing=True, version=2)
sglang_cache_volume = modal.Volume.from_name("miles-sglang-cache", create_if_missing=True, version=2)  # survives cold starts
delta_volume = modal.Volume.from_name(exp.DELTA_VOLUME_NAME, create_if_missing=True, version=2)

train_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
    str(PREP_PATH): prep_volume,
    exp.DELTA_BULLETIN_ROOT: delta_volume,
}

app = modal.App(exp.APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": miles_cfg.hf_checkpoint,
    "--cuda-graph-max-bs": str(ROLLOUT_CONCURRENCY),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    # The engine reads published versions off the Modal Volume; this hook reloads it
    # first (object stores lack cross-host read-after-write consistency).
    "--custom-pull-weights-pre-read-hook": "stitch.stores.modal_volume.pull_weights_pre_read_hook",
    **exp.SGLANG_SERVER_ARGS,
}


# The rollout Server: a thin module-level class (Modal requires @app.cls at global scope)
# whose enter/exit delegate to the shared common.server logic. sglang + the stitch sidecar.
@app.cls(
    image=server_image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.rollout_num_gpus_per_engine}",
    cloud=modal_cfg.cloud, region=modal_cfg.region,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(PREP_PATH): prep_volume,
        SGLANG_CACHE_PATH: sglang_cache_volume,
        exp.DELTA_BULLETIN_ROOT: delta_volume,
    },
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
            self, model_name=miles_cfg.hf_checkpoint, sglang_args=SGLANG_SERVER_ARGS,
            tp=miles_cfg.rollout_num_gpus_per_engine, concurrency=ROLLOUT_CONCURRENCY,
            bulletin_root=exp.DELTA_BULLETIN_ROOT, local_checkpoint_dir=exp.LOCAL_CHECKPOINT_PATH,
            volume_name=exp.DELTA_VOLUME_NAME, commit_mode=exp.SIDECAR_COMMIT_MODE,
            startup_timeout=SERVER_STARTUP_TIMEOUT,
        )

    @modal.exit()
    def stop(self) -> None:
        server.serve_stop(self)


# ── Trainer (miles on Ray) ────────────────────────────────────────────────────
_TRAINER_KWARGS = dict(
    image=image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.memory,
    cloud=modal_cfg.cloud, region=modal_cfg.region,
    volumes=train_volumes,
    ephemeral_disk=modal_cfg.trainer_ephemeral_disk_mib,
    timeout=24 * 60 * MINUTES, startup_timeout=20 * MINUTES, scaledown_window=30 * MINUTES,
    include_source=False,
)
if miles_cfg.n_train_nodes > 1:
    _TRAINER_KWARGS["experimental_options"] = {"efa_enabled": True}


class Trainer:
    """miles actor cluster. Ray comes up once per container in enter(), so back-to-back
    runs reuse it."""

    @modal.enter()
    def start_ray(self) -> None:
        from cookbook.common import process

        rank, master_addr, my_ip = ray_cluster.get_modal_cluster_context(miles_cfg.n_train_nodes)
        process.apply_git_patches(list(getattr(exp, "MEGATRON_RUNTIME_PATCHES", [])), MEGATRON_PATH, "Megatron patch")
        self.rank = rank
        process.start_host_mem_monitor()  # per-node host-RAM trace (publish gather is the OOM peak)
        os.environ.update({
            "MILES_HOST_IP": my_ip, "SGLANG_HOST_IP": my_ip, "HOST_IP": my_ip,
            "MASTER_ADDR": master_addr, "RAY_ADDRESS": f"{master_addr}:{RAY_PORT}",
            "no_proxy": f"127.0.0.1,{master_addr},{my_ip}", "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
            "PYTHONPATH": f"{MEGATRON_PATH}:{os.environ.get('PYTHONPATH', '')}",  # source-only megatron.training
            **miles_cfg.environment,
        })
        if rank == 0:
            ray_cluster.start_ray_head(my_ip, miles_cfg.n_train_nodes, ray_port=RAY_PORT)
        else:
            ray_cluster.start_ray_worker(my_ip, master_addr, ray_port=RAY_PORT)

    @modal.method()
    def train(self, payload: dict) -> None:
        """Run one training job from a MilesConfig payload (see MilesConfig.to_payload)."""
        for volume in train_volumes.values():
            volume.reload()

        cfg = MilesConfig.from_payload(payload)
        _materialize_node_local_yaml(cfg, "te_precision_config_file")  # re-read per Ray actor; needs identical local path
        if self.rank != 0:
            return

        cfg.rollout_endpoint_url = ModalFlashPool(exp.APP_NAME, "Server").gateway_url()
        run_id = uuid.uuid4().hex[:12]  # per-launch fence token; forks a fresh chain
        cfg.update_weight_disk_dir = f"{exp.DELTA_BULLETIN_ROOT}/{run_id}"
        if getattr(cfg, "save_interval", None) is None:
            cfg.load = cfg.save = cfg.save_hf = None
        else:
            cfg.load = cfg.save = f"{CHECKPOINTS_PATH}/{run_id}/checkpoints"
        # Merge the run's bulletin identity into custom_config_path (already carrying the
        # request-gating knobs); miles setattr's every key onto args for the hooks.
        custom_config = {
            **dict(getattr(cfg, "custom_config_path", {}) or {}),
            "update_weight_delta_volume_name": exp.DELTA_VOLUME_NAME,
            "rollout_modal_flash_app_name": exp.APP_NAME,
            "rollout_modal_flash_server_cls_name": "Server",
            "run_id": run_id,
        }
        cfg.custom_config_path = custom_config
        launch.resolve_config(cfg, tempfile.mkdtemp(), YAML_CONFIG_FIELDS)
        cmd = launch.build_train_cmd(cfg, MILES_ROOT, "miles_model_script")

        # Claim the pool before miles publishes: reset every replica to base for this run.
        from cookbook.common import hooks

        hooks.claim_pool(SimpleNamespace(update_weight_disk_dir=cfg.update_weight_disk_dir, **custom_config))

        print(f"Training {EXPERIMENT}: nodes={miles_cfg.n_train_nodes}, rollout_endpoint={cfg.rollout_endpoint_url}")
        print(f"Command: {cmd}")
        log_path = f"{CHECKPOINTS_PATH}/{run_id}/train.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        teed = f"set -o pipefail; ({cmd}) 2>&1 | tee {log_path}"  # tee to a committed log; survives the app-logs buffer
        try:
            subprocess.run(["bash", "-lc", teed], check=True)
        finally:
            try:
                checkpoints_volume.commit()
                print(f"Train log committed to miles-checkpoints at {run_id}/train.log")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not commit train log: {exc}")


if miles_cfg.n_train_nodes > 1:
    Trainer = modal.experimental.clustered(miles_cfg.n_train_nodes, rdma=True)(Trainer)
Trainer = app.cls(**_TRAINER_KWARGS)(Trainer)


# ── miles launch helper (te_precision_config_file is miles-only) ──────────────────
def _materialize_node_local_yaml(cfg: Any, field: str, dest_dir: str = "/root/.miles_node_yaml") -> None:
    """Write an inline YAML config to a deterministic node-local path on EVERY node.
    Fields like te_precision_config_file are re-read on each Ray actor, so they must
    resolve to identical content at an identical path on all nodes."""
    import yaml

    if isinstance(val := getattr(cfg, field, None), dict):
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{field}.yaml")
        with open(path, "w") as f:
            yaml.dump(val, f)
        setattr(cfg, field, path)


# ── Preparation + entrypoints ──────────────────────────────────────────────────
@app.function(
    image=image, gpu=f"{modal_cfg.gpu}:1",
    volumes={str(HF_CACHE_PATH): hf_cache_volume, str(PREP_PATH): prep_volume},
    timeout=6 * 60 * MINUTES, secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
def prepare_checkpoints() -> None:
    prep.prepare_checkpoints(exp, prep_volume)


# torch_dist conversion is clustered when >1 node; wrap the same name in place so Modal
# registers a single function (a leftover unregistered PartialFunction confuses the runner).
def prepare_torch_dist() -> None:
    rank, master_addr, _ = ray_cluster.get_modal_cluster_context(modal_cfg.torch_dist_prep_nodes)
    prep.prepare_torch_dist(exp, prep_volume, rank=rank, master_addr=master_addr)


_torch_dist_kwargs = dict(
    image=image, gpu=f"{modal_cfg.gpu}:{modal_cfg.torch_dist_prep_gpus_per_node}",
    volumes={str(HF_CACHE_PATH): hf_cache_volume, str(PREP_PATH): prep_volume},
    timeout=6 * 60 * MINUTES,
    ephemeral_disk=(modal_cfg.torch_dist_prep_ephemeral_disk_mib or modal_cfg.rollout_ephemeral_disk_mib),
    secrets=[modal.Secret.from_name("huggingface-secret")], include_source=False,
)
if modal_cfg.torch_dist_prep_nodes > 1:
    prepare_torch_dist = modal.experimental.clustered(modal_cfg.torch_dist_prep_nodes, rdma=True)(prepare_torch_dist)
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


@app.local_entrypoint()
def launch_train() -> None:
    """Spawn training on the deployed app. Config ships as data, so edits run without a
    redeploy; infrastructure changes still require one."""
    from modal.exception import NotFoundError

    try:
        trainer = modal.Cls.from_name(exp.APP_NAME, "Trainer")()
        call = trainer.train.spawn(miles_cfg.to_payload())
    except NotFoundError:
        raise SystemExit(
            f"App {exp.APP_NAME!r} is not deployed. Run:\n"
            f"  EXPERIMENT_CONFIG={EXPERIMENT} uv run --extra modal modal deploy -m cookbook.miles_disagg.app"
        )
    print(f"Spawned train on {exp.APP_NAME}: {call.object_id}")


@app.local_entrypoint()
def smoke_flash_pool(weight_version: int = 0, timeout_seconds: int = 30 * MINUTES) -> None:
    """Check the deployed Flash pool serves completions at the expected weight version."""
    smoke.smoke_flash_pool(
        app_name=exp.APP_NAME, cls_name="Server", model_name=miles_cfg.hf_checkpoint,
        weight_version=weight_version, timeout_seconds=timeout_seconds,
    )
