"""Disaggregated slime training on Modal, assembled on the stitch core.

``EXPERIMENT_CONFIG`` selects a config module under ``cookbook.slime_disagg.configs``. The
Server (sglang + stitch sidecar) is the shared common one; the Trainer runs slime on Ray
and publishes XOR deltas through a Modal Volume the pool syncs from.

Prepare the model + dataset once first (a separate app, so prep never spins up the rollout
Server floor — see ``cookbook.slime_disagg.prep_app``), then launch a run with one command — it
mints a unique run id, stands up that run's pool, and starts training. Each launch is its own run,
isolated even from an identical-config relaunch (see ``cookbook.slime_disagg.launch``):

    EXPERIMENT_CONFIG=kimi_k2_6_int4 uv run --extra modal python -m cookbook.slime_disagg.launch

Config access is uniform: the experiment module ``exp`` is the single source of truth —
its ``exp.modal`` (infra), ``exp.slime`` (training), and ``exp.<CONST>`` are read directly;
shared deployment constants come from ``common.constants``. ``ROLLOUT_CONCURRENCY`` is the
one resolved value (the experiment's Flash target, else the engine's concurrency).
"""

from __future__ import annotations

import importlib
import os
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any

import modal
import modal.experimental

from cookbook.common import launch, ray_cluster, router, server, serving_image
from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    DRAFT_PATH,
    HF_CACHE_PATH,
    MINUTES,
    RAY_PORT,
    SERVER_STARTUP_TIMEOUT,
    SGLANG_CACHE_PATH,
    SIDECAR_PORT,
    STITCH_PATH,
)
from cookbook.slime_disagg import trainer_image
from cookbook.slime_disagg.config import YAML_CONFIG_FIELDS, SlimeConfig
from cookbook.slime_disagg.trainer_image import SLIME_ROOT
from stitch.pools.modal_flash_lb_temp import ModalFlashLBPool

EXPERIMENT = os.environ[
    "EXPERIMENT_CONFIG"
]  # required; a default would silently serve the wrong experiment
SLIME_LOCAL_DIR = os.environ.get(
    "SLIME_LOCAL_DIR"
)  # optional dev overlay of a local slime checkout

exp = importlib.import_module(f"cookbook.slime_disagg.configs.{EXPERIMENT}")
modal_cfg = exp.modal
slime_cfg = exp.slime

# Per-run id, minted fresh by cookbook.slime_disagg.launch. The same identity
# scopes the pool, Stitch pointer, publications, checkpoints, and logs.
RUN_ID = os.environ["RUN_ID"]
APP_NAME = f"{exp.APP_NAME}-{RUN_ID}"
RUN_DIR = STITCH_PATH / RUN_ID
UPDATES_DIR = RUN_DIR / "updates"

# Flash autoscaler target / sglang concurrency cap: explicit target_inputs, else engine concurrency.
ROLLOUT_CONCURRENCY = (
    modal_cfg.rollout_target_inputs or slime_cfg.sglang_server_concurrency
)

# EXPERIMENT_CONFIG + RUN_ID are baked into both images so a container's re-import rebuilds the same
# app name and transport paths as the deploy, not the defaults.
image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    run_id=RUN_ID,
    slime_local=SLIME_LOCAL_DIR,
)
server_image = serving_image.build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    run_id=RUN_ID,
    runtime=getattr(exp, "SGLANG_RUNTIME", serving_image.DEFAULT_SGLANG_RUNTIME),
)
if SLIME_LOCAL_DIR:
    server_image = server_image.add_local_dir(
        SLIME_LOCAL_DIR,
        remote_path=SLIME_ROOT,
        ignore=[".git", "**/__pycache__", "**/*.pyc"],
    )

hf_cache_volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True, version=2)
checkpoint_volume = modal.Volume.from_name(
    "slime-checkpoints",
    create_if_missing=True,
    version=2,
)
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
)
run_volume = modal.Volume.from_name(
    exp.EXPERIMENT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
draft_volume = (
    modal.Volume.from_name(
        modal_cfg.draft_volume,
        environment_name=modal_cfg.draft_volume_env,
        version=2,
    )
    if modal_cfg.draft_volume
    else None
)

train_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(CHECKPOINTS_PATH): checkpoint_volume,
    str(DATA_PATH): data_volume,
    str(STITCH_PATH): run_volume,
}

app = modal.App(APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": slime_cfg.hf_checkpoint,
    "--dtype": "bfloat16",
    "--cuda-graph-max-bs-decode": str(ROLLOUT_CONCURRENCY),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    **exp.SGLANG_SERVER_ARGS,
}


# The rollout Server is a thin module-level class whose lifecycle delegates to the
# shared common.server logic: sglang plus the stitch sidecar.
@app.server(
    image=server_image,
    gpu=modal_cfg.rollout_gpus(slime_cfg.rollout_num_gpus_per_engine),
    cloud=modal_cfg.cloud,
    compute_region=modal_cfg.region,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(CHECKPOINTS_PATH): checkpoint_volume,
        str(STITCH_PATH): run_volume,
        SGLANG_CACHE_PATH: sglang_cache_volume,
        **({str(DRAFT_PATH): draft_volume} if draft_volume is not None else {}),
    },
    min_containers=modal_cfg.rollout_min_containers,
    max_containers=modal_cfg.rollout_max_containers,
    target_concurrency=ROLLOUT_CONCURRENCY,
    scaledown_window=15 * MINUTES,
    ephemeral_disk=modal_cfg.rollout_ephemeral_disk_mib,
    memory=modal_cfg.rollout_memory_mib,
    include_source=False,
    port=SIDECAR_PORT,
    routing_region=modal_cfg.routing_region,
    unauthenticated=True,
    exit_grace_period=60 * MINUTES,
    startup_timeout=SERVER_STARTUP_TIMEOUT,
)
class Server:
    @modal.enter()
    def startup(self) -> None:
        server.serve_startup(
            self,
            model_name=slime_cfg.hf_checkpoint,
            sglang_args=SGLANG_SERVER_ARGS,
            tp=slime_cfg.rollout_num_gpus_per_engine,
            concurrency=ROLLOUT_CONCURRENCY,
            bulletin_root=str(RUN_DIR),
            local_checkpoint_dir=exp.LOCAL_CHECKPOINT_PATH,
            delta_update_mode=exp.SGLANG_DELTA_UPDATE_MODE,
            volume_name=exp.EXPERIMENT_VOLUME_NAME,
            commit_mode=exp.SIDECAR_COMMIT_MODE,
            run_id=RUN_ID,
            flush_cache_on_commit=exp.SIDECAR_FLUSH_CACHE_ON_COMMIT,
            startup_timeout=SERVER_STARTUP_TIMEOUT,
        )

    @modal.exit()
    def stop(self) -> None:
        server.serve_stop(self)


# ── Session-routing LB (cookbook/common/router.py) ─────────────────────────────
# The router is two CPU Flash classes in this same app, so it deploys and dies with
# the pool; the GPU class keeps ``Server``, rollout traffic enters through ``Router``.
router_image = router.build_router_image(EXPERIMENT, RUN_ID)
session_routes = router.session_routes_dict(APP_NAME)


@app.server(
    image=router_image,
    cpu=2,
    memory=1024,
    min_containers=modal_cfg.router_registry_min_containers,
    routing_region=modal_cfg.routing_region,
    include_source=False,
    port=8000,
    unauthenticated=True,
)
class RouterRegistry:
    """Polls Server replicas' live queue depth; serves the snapshot at /loads."""

    @modal.enter()
    def enter(self) -> None:
        router.serve_registry(self, app_name=APP_NAME, upstream_cls="Server")

    @modal.exit()
    def exit(self) -> None:
        router.stop_server(self)


@app.server(
    image=router_image,
    cpu=4,
    memory=2048,
    min_containers=modal_cfg.router_min_containers,
    target_concurrency=modal_cfg.router_target_concurrency,
    routing_region=modal_cfg.routing_region,
    include_source=False,
    port=8000,
    unauthenticated=True,
    exit_grace_period=30 * MINUTES,
)
class Router:
    """Front door for rollout traffic: session-affinity routing across Server replicas,
    with 503 eviction + retry, so a saturated replica sheds sessions instead of
    attracting them."""

    @modal.enter()
    def enter(self) -> None:
        router.serve_router(
            self,
            registry_url=RouterRegistry.get_url(),
            upstream_url=Server.get_url(),
            session_routes=session_routes,
            overload_threshold=ROLLOUT_CONCURRENCY,
        )

    @modal.exit()
    def exit(self) -> None:
        router.stop_server(self)


# ── Trainer (slime on Ray) ────────────────────────────────────────────────────
# Multi-node needs an RDMA gang (clustered) over the EFA fabric; single-node takes
# neither. Both are inline on the decorator so there's one declaration, not a rebind.
_MULTINODE = slime_cfg.n_train_nodes > 1


@app.cls(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.trainer_memory_mib,
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes=train_volumes,
    ephemeral_disk=modal_cfg.trainer_ephemeral_disk_mib,
    timeout=24 * 60 * MINUTES,
    startup_timeout=20 * MINUTES,
    scaledown_window=30 * MINUTES,
    include_source=False,
    **({"experimental_options": {"efa_enabled": True}} if _MULTINODE else {}),
)
@(
    modal.experimental.clustered(slime_cfg.n_train_nodes, rdma=True)
    if _MULTINODE
    else lambda c: c
)
class Trainer:
    """slime actor cluster. Ray comes up once per container in enter(), so back-to-back
    runs reuse it."""

    @modal.enter()
    def start_ray(self) -> None:
        from cookbook.common import process

        rank, master_addr, my_ip = ray_cluster.get_modal_cluster_context(
            slime_cfg.n_train_nodes
        )
        self.rank = rank
        process.start_host_mem_monitor()  # per-node host-RAM trace
        ray_cluster.start_ray_node(
            rank,
            master_addr,
            my_ip,
            n_nodes=slime_cfg.n_train_nodes,
            ray_port=RAY_PORT,
            extra_env={"SLIME_HOST_IP": my_ip, **slime_cfg.environment},
        )

    @modal.method()
    def train(self, payload: dict) -> None:
        """Run one training job from a SlimeConfig payload (see SlimeConfig.to_payload)."""
        for volume in train_volumes.values():
            volume.reload()
        if self.rank != 0:
            return

        cfg = SlimeConfig.from_payload(payload)
        cfg.rollout_endpoint_url = ModalFlashLBPool(APP_NAME, "Server").gateway_url()
        # Slime requires this CLI argument; the deployment owns its run-scoped value.
        cfg.update_weight_disk_dir = str(UPDATES_DIR)
        hook_knobs = {
            "experiment_volume_name": exp.EXPERIMENT_VOLUME_NAME,
            "rollout_modal_flash_app_name": APP_NAME,
            "rollout_modal_flash_server_cls_name": "Server",
            "run_id": RUN_ID,
        }
        cfg.custom_config_path = (
            hook_knobs  # materialized to a YAML path below; keep the mapping for claim
        )
        launch.resolve_config(
            cfg,
            tempfile.mkdtemp(),
            checkpoint_fields=("hf_checkpoint", "load", "ref_load", "critic_load"),
            yaml_fields=YAML_CONFIG_FIELDS,
        )
        cmd = _build_train_cmd(cfg)

        # Claim the pool before slime publishes: reset every replica to base for this run.
        from cookbook.common import hooks

        hooks.claim_pool(
            SimpleNamespace(
                update_weight_disk_dir=cfg.update_weight_disk_dir, **hook_knobs
            )
        )

        print(
            f"Training {EXPERIMENT}: nodes={slime_cfg.n_train_nodes}, rollout_endpoint={cfg.rollout_endpoint_url}"
        )
        print(f"Command: {cmd}")
        subprocess.run(["bash", "-lc", cmd], check=True)


def _build_train_cmd(cfg: SlimeConfig) -> str:
    train_script = f"{SLIME_ROOT}/{'train_async.py' if cfg.async_mode else 'train.py'}"
    model_script = cfg.slime_model_script
    if model_script:
        inner = (
            f"source {SLIME_ROOT}/{model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(cfg.cli_args())}"


# ── Entrypoints (preparation lives in a separate app: cookbook.slime_disagg.prep_app) ──
def spawn_train() -> Any:
    """Spawn the trainer on this run's already-deployed pool (config ships as data, so config
    edits run without a redeploy; infra changes still require one)."""
    trainer = modal.Cls.from_name(APP_NAME, "Trainer")()
    call = trainer.train.spawn(slime_cfg.to_payload())
    print(f"Spawned train on {APP_NAME}: {call.object_id}")
    return call


@app.local_entrypoint()
def launch_train() -> None:
    """Spawn training on a pool that's already up for this RUN. ``cookbook.slime_disagg.launch``
    deploys + spawns in one command; use this only to re-spawn against a running pool."""
    from modal.exception import NotFoundError

    try:
        spawn_train()
    except NotFoundError:
        raise SystemExit(
            f"App {APP_NAME!r} is not deployed. Launch a fresh run with:\n"
            f"  EXPERIMENT_CONFIG={EXPERIMENT} uv run --extra modal python -m cookbook.slime_disagg.launch"
        ) from None
    print(f"stop this run when done: modal app stop {APP_NAME}")
