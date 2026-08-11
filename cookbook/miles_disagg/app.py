"""Disaggregated miles training on Modal, assembled on the stitch core.

``EXPERIMENT_CONFIG`` selects a config module under ``cookbook.miles_disagg.configs``. The
Server (sglang + stitch sidecar) is the shared common one; the Trainer runs miles on Ray
and publishes XOR deltas through the configured checkpoint store.

Prepare the checkpoints once first (a separate app, so prep never spins up the rollout Server
floor — see ``cookbook.miles_disagg.prep_app``), then launch a run with one command — it mints a
unique run id, stands up that run's pool, and starts training. Each launch is its own run,
isolated even from an identical-config relaunch (see ``cookbook.miles_disagg.launch``):

    EXPERIMENT_CONFIG=glm45_air_fp8 uv run --extra modal python -m cookbook.miles_disagg.launch

Config access is uniform: the experiment module ``exp`` is the single source of truth —
its ``exp.modal`` (infra), ``exp.miles`` (training), and ``exp.<CONST>`` are read directly;
shared deployment constants come from ``common.constants``. ``ROLLOUT_CONCURRENCY`` is the
one resolved value (the experiment's Flash target, else the engine's concurrency).
"""

from __future__ import annotations

import importlib
import os
import shlex
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from typing import Any

import modal
import modal.experimental

from cookbook.common import launch, ray_cluster, router, server, serving_image, storage
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
from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.config import YAML_CONFIG_FIELDS, MilesConfig
from cookbook.miles_disagg.resume import (
    RESUME_POINT_ENV,
    ResumePoint,
)
from cookbook.miles_disagg.trainer_image import MEGATRON_PATH, MILES_ROOT
from stitch.pools.modal_flash_lb_temp import ModalFlashLBPool

EXPERIMENT = os.environ[
    "EXPERIMENT_CONFIG"
]  # required; a default would silently serve the wrong experiment
MILES_LOCAL_DIR = os.environ.get(
    "MILES_LOCAL_DIR"
)  # optional dev overlay of a local miles checkout

exp = importlib.import_module(f"cookbook.miles_disagg.configs.{EXPERIMENT}")
modal_cfg = exp.modal
miles_cfg = exp.miles


def _resume_point() -> ResumePoint | None:
    if payload := os.environ.get(RESUME_POINT_ENV):
        return ResumePoint.from_json(payload)
    return None


def _rollout_boot_checkpoint() -> str:
    point = _resume_point()
    return point.rollout_checkpoint if point is not None else miles_cfg.hf_checkpoint


def _rollout_boot_version() -> int:
    point = _resume_point()
    return point.version if point is not None else 0


# Per-run id, minted fresh by cookbook.miles_disagg.launch. The same identity
# scopes the pool, Stitch pointer, publications, checkpoints, and logs.
RUN_ID = os.environ["RUN_ID"]
APP_NAME = f"{exp.APP_NAME}-{RUN_ID}"
RUN_DIR = STITCH_PATH / RUN_ID
STORE_DEPLOYMENT = storage.StoreDeployment.from_environment()
UPDATES_DIR = STORE_DEPLOYMENT.updates_dir(RUN_DIR)
STORE_SECRETS = STORE_DEPLOYMENT.modal_secrets()

# Flash autoscaler target / sglang concurrency cap: explicit target_inputs, else engine concurrency.
ROLLOUT_CONCURRENCY = (
    modal_cfg.rollout_target_inputs or miles_cfg.sglang_server_concurrency
)

# EXPERIMENT_CONFIG + RUN are baked into both images so a container's re-import rebuilds the same
# app name and transport paths as the deploy, not the defaults.
image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    run_id=RUN_ID,
    miles_repo_ref=getattr(exp, "MILES_REPO_REF", trainer_image.MILES_REPO_REF),
    miles_local=MILES_LOCAL_DIR,
    extra_pip_packages=(
        *getattr(exp, "TRAINER_EXTRA_PIP_PACKAGES", ()),
        *STORE_DEPLOYMENT.extra_packages,
    ),
    image_run_commands=getattr(exp, "TRAINER_IMAGE_RUN_COMMANDS", ()),
    extra_env=STORE_DEPLOYMENT.image_environment,
)
# Server containers re-import this module, so persist this attempt's resume point.
server_image = serving_image.build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    run_id=RUN_ID,
    extra_packages=STORE_DEPLOYMENT.extra_packages,
    extra_env={
        **(getattr(exp, "SGLANG_SERVER_ENV", None) or {}),
        RESUME_POINT_ENV: os.environ.get(RESUME_POINT_ENV, ""),
        **STORE_DEPLOYMENT.image_environment,
    },
    runtime=getattr(exp, "SGLANG_RUNTIME", serving_image.DEFAULT_SGLANG_RUNTIME),
)
if MILES_LOCAL_DIR:
    server_image = server_image.add_local_dir(
        MILES_LOCAL_DIR,
        remote_path=MILES_ROOT,
        ignore=[".git", "**/__pycache__", "**/*.pyc"],
    )

hf_cache_volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
data_volume = modal.Volume.from_name("miles-data", create_if_missing=True, version=2)
checkpoint_volume = modal.Volume.from_name(
    "miles-checkpoints",
    create_if_missing=True,
    version=2,
)
run_volume = modal.Volume.from_name(
    exp.EXPERIMENT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
)  # survives cold starts
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
    "--served-model-name": _rollout_boot_checkpoint(),
    **(
        {}
        if "--cuda-graph-config" in exp.SGLANG_SERVER_ARGS
        else {"--cuda-graph-max-bs-decode": str(ROLLOUT_CONCURRENCY)}
    ),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    **exp.SGLANG_SERVER_ARGS,
}


# The rollout Server is a thin module-level class whose lifecycle delegates to the
# shared common.server logic: sglang plus the stitch sidecar.
@app.server(
    image=server_image,
    gpu=modal_cfg.rollout_gpus(miles_cfg.rollout_num_gpus_per_engine),
    cloud=modal_cfg.cloud,
    compute_region=modal_cfg.region,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(CHECKPOINTS_PATH): checkpoint_volume,
        **(
            {str(STITCH_PATH): run_volume}
            if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME
            else {}
        ),
        SGLANG_CACHE_PATH: sglang_cache_volume,
        **(
            {str(DRAFT_PATH): draft_volume.read_only()}
            if draft_volume is not None
            else {}
        ),
    },
    min_containers=modal_cfg.rollout_min_containers,
    max_containers=modal_cfg.rollout_max_containers,
    target_concurrency=ROLLOUT_CONCURRENCY,
    scaledown_window=15 * MINUTES,
    ephemeral_disk=modal_cfg.rollout_ephemeral_disk_mib,
    memory=modal_cfg.rollout_memory_mib,
    secrets=STORE_SECRETS,
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
        STORE_DEPLOYMENT.bootstrap_credentials()
        store_config = STORE_DEPLOYMENT.hook_config(APP_NAME)
        server.serve_startup(
            self,
            model_name=_rollout_boot_checkpoint(),
            boot_version=_rollout_boot_version(),
            sglang_args=SGLANG_SERVER_ARGS,
            tp=miles_cfg.rollout_num_gpus_per_engine,
            concurrency=ROLLOUT_CONCURRENCY,
            bulletin_root=str(RUN_DIR),
            local_checkpoint_dir=exp.LOCAL_CHECKPOINT_PATH,
            delta_update_mode=exp.SGLANG_DELTA_UPDATE_MODE,
            store_backend=store_config["stitch_store_backend"],
            volume_name=exp.EXPERIMENT_VOLUME_NAME,
            s3_root=store_config.get("stitch_s3_root"),
            s3_endpoint_url=store_config.get("stitch_s3_endpoint_url"),
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
router_image = router.build_router_image(
    EXPERIMENT,
    RUN_ID,
    extra_env=STORE_DEPLOYMENT.image_environment,
)
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


# ── Trainer (miles on Ray) ────────────────────────────────────────────────────
# Multi-node needs an RDMA gang (clustered) over the EFA fabric; single-node takes
# neither. Both are inline on the decorator so there's one declaration, not a rebind.
_MULTINODE = miles_cfg.n_train_nodes > 1


@app.cls(
    image=image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.trainer_memory_mib,
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes=train_volumes,
    secrets=[
        *STORE_SECRETS,
        *(
            [modal.Secret.from_name("wandb-secret")]
            if getattr(miles_cfg, "use_wandb", False)
            else []
        ),
    ],
    ephemeral_disk=modal_cfg.trainer_ephemeral_disk_mib,
    timeout=24 * 60 * MINUTES,
    startup_timeout=20 * MINUTES,
    scaledown_window=30 * MINUTES,
    include_source=False,
    **({"experimental_options": {"efa_enabled": True}} if _MULTINODE else {}),
)
@(
    modal.experimental.clustered(miles_cfg.n_train_nodes, rdma=True)
    if _MULTINODE
    else lambda c: c
)
class Trainer:
    """miles actor cluster. Ray comes up once per container in enter(), so back-to-back
    runs reuse it."""

    @modal.enter()
    def start_ray(self) -> None:
        from cookbook.common import process

        STORE_DEPLOYMENT.bootstrap_credentials()
        rank, master_addr, my_ip = ray_cluster.get_modal_cluster_context(
            miles_cfg.n_train_nodes
        )
        process.apply_git_patches(
            list(getattr(exp, "MEGATRON_RUNTIME_PATCHES", [])),
            MEGATRON_PATH,
            "Megatron patch",
        )
        self.rank = rank
        process.start_host_mem_monitor()  # per-node host-RAM trace
        ray_cluster.start_ray_node(
            rank,
            master_addr,
            my_ip,
            n_nodes=miles_cfg.n_train_nodes,
            ray_port=RAY_PORT,
            extra_env={
                "MILES_HOST_IP": my_ip,
                "PYTHONPATH": f"{MEGATRON_PATH}:{os.environ.get('PYTHONPATH', '')}",  # source-only megatron.training
                **miles_cfg.environment,
            },
        )

    @modal.method()
    def train(self, payload: dict, resume_payload: str | None = None) -> None:
        """Run one training job from a MilesConfig payload (see MilesConfig.to_payload)."""
        for volume in train_volumes.values():
            volume.reload()

        resume_point = (
            ResumePoint.from_json(resume_payload)
            if resume_payload is not None
            else None
        )
        cfg = MilesConfig.from_payload(payload)
        launch.materialize_node_local_yaml(cfg, "te_precision_config_file")
        if self.rank != 0:
            return

        cfg.rollout_endpoint_url = ModalFlashLBPool(APP_NAME, "Server").gateway_url()
        if resume_point is not None:
            cfg.load = resume_point.trainer_checkpoint
            cfg.hf_checkpoint = resume_point.rollout_checkpoint
            cfg.exit_on_missing_checkpoint = True
        # Miles requires this CLI argument; the deployment owns its run-scoped value.
        cfg.update_weight_disk_dir = str(UPDATES_DIR)
        if getattr(cfg, "save_interval", None) is None:
            cfg.save = cfg.save_hf = None
        else:
            cfg.save = str(RUN_DIR / "checkpoints")
            if save_hf := getattr(cfg, "save_hf", None):
                cfg.save_hf = str(RUN_DIR / save_hf)
        # miles setattr's every key onto args for the hooks.
        custom_config = {
            **(cfg.custom_config_path or {}),
            **STORE_DEPLOYMENT.hook_config(APP_NAME),
            "experiment_volume_name": exp.EXPERIMENT_VOLUME_NAME,
            "rollout_modal_flash_app_name": APP_NAME,
            "rollout_modal_flash_server_cls_name": "Server",
            "run_id": RUN_ID,
        }
        cfg.custom_config_path = custom_config
        launch.resolve_config(
            cfg,
            tempfile.mkdtemp(),
            checkpoint_fields=("hf_checkpoint", "load", "ref_load", "critic_load"),
            yaml_fields=YAML_CONFIG_FIELDS,
        )
        cmd = _build_train_cmd(cfg)

        # Claim the version already served by the pool before Miles publishes.
        from cookbook.common import hooks

        hooks.claim_pool(
            SimpleNamespace(
                update_weight_disk_dir=cfg.update_weight_disk_dir, **custom_config
            ),
            boot_version=resume_point.version if resume_point is not None else 0,
        )

        resume_log = (
            f", checkpoint_version={resume_point.version}, "
            f"next_version={resume_point.version + 1}, "
            f"source_run_id={resume_point.source_run_id}, "
            f"stitch_boot={RUN_ID}/weight_v{resume_point.version:06d}"
            if resume_point is not None
            else ""
        )
        print(
            f"Training {EXPERIMENT}: run={RUN_ID}{resume_log}, "
            f"nodes={miles_cfg.n_train_nodes}, rollout_endpoint={cfg.rollout_endpoint_url}"
        )
        print(f"Command: {cmd}")
        log_path = str(RUN_DIR / "train.log")
        with tempfile.TemporaryDirectory() as local_log_dir:
            local_log_path = os.path.join(local_log_dir, "train.log")
            teed = f"set -o pipefail; ({cmd}) 2>&1 | tee {local_log_path}"
            try:
                subprocess.run(["bash", "-lc", teed], check=True)
            finally:
                try:
                    os.makedirs(os.path.dirname(log_path), exist_ok=True)
                    shutil.copyfile(local_log_path, log_path)
                    run_volume.commit()
                    print(
                        f"Train log committed to {exp.EXPERIMENT_VOLUME_NAME} at {log_path}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: could not commit train log: {exc}")


def _build_train_cmd(cfg: MilesConfig) -> str:
    train_script = f"{MILES_ROOT}/{'train_async.py' if cfg.async_mode else 'train.py'}"
    args = cfg.cli_args()
    if cfg.megatron_model_type:
        from miles.utils.external_utils.model_args_utils import load_model_args

        args = [
            *shlex.split(load_model_args(cfg.megatron_model_type)),
            *args,
        ]
    return shlex.join(["python3", train_script, *args])


# ── Entrypoints (preparation lives in a separate app: cookbook.miles_disagg.prep_app) ──
def spawn_train() -> Any:
    """Spawn the trainer on this run's already-deployed pool (config ships as data, so config
    edits run without a redeploy; infra changes still require one)."""
    trainer = modal.Cls.from_name(APP_NAME, "Trainer")()
    call = trainer.train.spawn(
        miles_cfg.to_payload(),
        os.environ.get(RESUME_POINT_ENV),
    )
    print(f"Spawned train on {APP_NAME}: {call.object_id}")
    return call


@app.local_entrypoint()
def launch_train() -> None:
    """Spawn training on a pool that's already up for this RUN. ``cookbook.miles_disagg.launch``
    deploys + spawns in one command; use this only to re-spawn against a running pool."""
    from modal.exception import NotFoundError

    try:
        spawn_train()
    except NotFoundError:
        raise SystemExit(
            f"App {APP_NAME!r} is not deployed. Launch a fresh run with:\n"
            f"  EXPERIMENT_CONFIG={EXPERIMENT} uv run --extra modal python -m cookbook.miles_disagg.launch"
        ) from None
    print(f"stop this run when done: modal app stop {APP_NAME}")
