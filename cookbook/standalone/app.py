"""Standalone rollout serving on the stitch core — the pool without a trainer.

``EXPERIMENT_CONFIG`` selects a config module under ``cookbook.standalone``. The Server
(sglang + stitch sidecar) and the session-routing LB are the complete deployment: weight
publications arrive from an external trainer or harness through the configured checkpoint
store, and rollout traffic enters through ``Router``.

Prepare the served base once first (a separate app, so prep never spins up the rollout
Server floor — see ``cookbook.standalone.prep_app``), then launch a pool with one command —
it mints a unique run id and stands up that run's pool (see ``cookbook.standalone.launch``):

    EXPERIMENT_CONFIG=glm5_2_fp8 uv run --extra modal python -m cookbook.standalone.launch
"""

from __future__ import annotations

import importlib
import os

import modal

from cookbook.common import router, serving_image, storage
from cookbook.common import server as common_server
from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    DRAFT_PATH,
    HF_CACHE_PATH,
    MINUTES,
    SERVER_STARTUP_TIMEOUT,
    SGLANG_CACHE_PATH,
    SIDECAR_PORT,
    STITCH_PATH,
)
from cookbook.standalone import checkpoint

EXPERIMENT = os.environ[
    "EXPERIMENT_CONFIG"
]  # required; a default would silently serve the wrong experiment
exp = importlib.import_module(f"cookbook.standalone.configs.{EXPERIMENT}")
modal_cfg = exp.modal

# Minted once for a run. The same identity scopes the pool, Stitch pointer, and
# publications, so an external publisher targets exactly this pool's transport.
RUN_ID = os.environ["RUN_ID"]
APP_NAME = f"{exp.APP_NAME}-{RUN_ID}"
RUN_DIR = STITCH_PATH / RUN_ID
STORE_DEPLOYMENT = storage.StoreDeployment.from_environment()
STORE_SECRETS = STORE_DEPLOYMENT.modal_secrets()

# Flash autoscaler target; the engine's own admission cap is the experiment's to set.
ROLLOUT_CONCURRENCY = modal_cfg.rollout_target_inputs
if ROLLOUT_CONCURRENCY is None:
    raise ValueError("rollout_target_inputs is required for a standalone pool")

# EXPERIMENT_CONFIG + RUN_ID are baked into the images so a container's re-import rebuilds
# the same app name and transport paths as the deploy, not the defaults.
server_image = serving_image.build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    run_id=RUN_ID,
    extra_packages=STORE_DEPLOYMENT.extra_packages,
    extra_env={
        **(getattr(exp, "SGLANG_SERVER_ENV", None) or {}),
        **STORE_DEPLOYMENT.image_environment,
    },
    runtime=getattr(exp, "SGLANG_RUNTIME", serving_image.DEFAULT_SGLANG_RUNTIME),
)

hf_cache_volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
checkpoint_volume = modal.Volume.from_name(
    exp.CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
run_volume = modal.Volume.from_name(
    exp.EXPERIMENT_VOLUME_NAME, create_if_missing=True, version=2
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

app = modal.App(APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": str(exp.BASE_CHECKPOINT_PATH),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    **exp.SGLANG_SERVER_ARGS,
}


# The rollout Server is a thin module-level class whose lifecycle delegates to the
# shared common.server logic: sglang plus the stitch sidecar.
@app.server(
    image=server_image,
    gpu=modal_cfg.rollout_gpus(exp.ROLLOUT_GPUS_PER_ENGINE),
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
        checkpoint.require_checkpoint(exp.BASE_CHECKPOINT_PATH)
        store = storage.create_store(
            store_config["stitch_store_backend"],
            local_root=RUN_DIR,
            run_id=RUN_ID,
            volume_name=exp.EXPERIMENT_VOLUME_NAME,
            s3_root=store_config.get("stitch_s3_root"),
            s3_endpoint_url=store_config.get("stitch_s3_endpoint_url"),
        )
        checkpoint.wait_for_boot_pointer(store, RUN_ID)
        common_server.serve_startup(
            self,
            model_name=str(exp.BASE_CHECKPOINT_PATH),
            sglang_args=SGLANG_SERVER_ARGS,
            concurrency=ROLLOUT_CONCURRENCY,
            bulletin_root=str(RUN_DIR),
            local_checkpoint_dir=exp.LOCAL_CHECKPOINT_PATH,
            delta_update_mode=exp.SGLANG_DELTA_UPDATE_MODE,
            store_backend=store_config["stitch_store_backend"],
            volume_name=exp.EXPERIMENT_VOLUME_NAME,
            s3_root=store_config.get("stitch_s3_root"),
            s3_endpoint_url=store_config.get("stitch_s3_endpoint_url"),
            run_id=RUN_ID,
            commit_mode=exp.SIDECAR_COMMIT_MODE,
            flush_cache_on_commit=exp.SIDECAR_FLUSH_CACHE_ON_COMMIT,
            # Base configs omit SIDECAR_RECONCILE_INTERVAL: 5.0s is stock behavior.
            reconcile_interval=getattr(exp, "SIDECAR_RECONCILE_INTERVAL", 5.0),
            startup_timeout=SERVER_STARTUP_TIMEOUT,
        )

    @modal.exit()
    def stop(self) -> None:
        common_server.serve_stop(self)


# The launcher invokes this one-shot function after deploy. Keeping replicas
# read-only avoids a multi-replica claim race, while the short wait in startup
# keeps them out of rotation until the claim is visible.
@app.function(
    image=server_image,
    volumes=(
        {str(STITCH_PATH): run_volume}
        if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME
        else {}
    ),
    secrets=STORE_SECRETS,
    include_source=False,
    timeout=10 * MINUTES,
)
def claim_boot_pointer() -> None:
    """Claim this run's v0 pointer through the deployed checkpoint store."""

    STORE_DEPLOYMENT.bootstrap_credentials()
    store_config = STORE_DEPLOYMENT.hook_config(APP_NAME)
    store = storage.create_store(
        store_config["stitch_store_backend"],
        local_root=RUN_DIR,
        run_id=RUN_ID,
        volume_name=exp.EXPERIMENT_VOLUME_NAME,
        s3_root=store_config.get("stitch_s3_root"),
        s3_endpoint_url=store_config.get("stitch_s3_endpoint_url"),
    )
    checkpoint.claim_boot_pointer(store, RUN_ID)


# ── Session-routing LB ─────────────────────────────────────────────────────────
# The router is two CPU Flash classes in this same app, so it deploys and dies with
# the pool; the GPU class keeps ``Server``, rollout traffic enters through ``Router``.
# OFFLINE_EVALS on the experiment config opts into the offline-evals router
# (cookbook/standalone/offline_evals/): version-pinned placement plus the
# registry-driven rollout control loop. Unset, the wiring is the stock
# common-router pair below.
router_image = router.build_router_image(
    EXPERIMENT,
    RUN_ID,
    extra_env=STORE_DEPLOYMENT.image_environment,
)
session_routes = router.session_routes_dict(APP_NAME)

if getattr(exp, "OFFLINE_EVALS", False):
    from cookbook.standalone.offline_evals import eval_router, rollout_control

    if STORE_DEPLOYMENT.extra_packages:
        # The registry container reads the checkpoint store (rollout control loop).
        router_image = router_image.pip_install(*STORE_DEPLOYMENT.extra_packages)
    rollout_marks = rollout_control.rollout_control_dict(APP_NAME)

    @app.server(
        image=router_image,
        cpu=2,
        memory=1024,
        # Single-writer: the rollout control loop is correct only with exactly one
        # registry container, so the registry is pinned to 1 regardless of config.
        min_containers=1,
        max_containers=1,
        routing_region=modal_cfg.routing_region,
        volumes=(
            {str(STITCH_PATH): run_volume}
            if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME
            else {}
        ),
        secrets=STORE_SECRETS,
        include_source=False,
        port=8000,
        unauthenticated=True,
    )
    class RouterRegistry:
        """Polls Server replicas' server_info + queue depth; serves the snapshot at /loads.

        Also drives the rollout control loop (session liveness, pending-reconcile
        marks, consolidation) — correct only with a single registry container."""

        @modal.enter()
        def enter(self) -> None:
            STORE_DEPLOYMENT.bootstrap_credentials()
            store_config = STORE_DEPLOYMENT.hook_config(APP_NAME)
            store = storage.create_store(
                store_config["stitch_store_backend"],
                local_root=RUN_DIR,
                run_id=RUN_ID,
                volume_name=exp.EXPERIMENT_VOLUME_NAME,
                s3_root=store_config.get("stitch_s3_root"),
                s3_endpoint_url=store_config.get("stitch_s3_endpoint_url"),
            )
            eval_router.serve_eval_registry(
                self,
                app_name=APP_NAME,
                upstream_cls="Server",
                session_routes=session_routes,
                control=rollout_marks,
                store=store,
                rollout_concurrency=ROLLOUT_CONCURRENCY,
                session_ttl=getattr(
                    exp, "SESSION_TTL_SECONDS", eval_router.DEFAULT_SESSION_TTL_SECONDS
                ),
            )

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
        """Front door for rollout traffic: version-pinned, session-affinity routing
        across Server replicas, with 503 eviction + retry, so a saturated replica
        sheds sessions instead of attracting them."""

        @modal.enter()
        def enter(self) -> None:
            eval_router.serve_eval_router(
                self,
                registry_url=RouterRegistry.get_url(),
                upstream_url=Server.get_url(),
                session_routes=session_routes,
                overload_threshold=ROLLOUT_CONCURRENCY,
            )

        @modal.exit()
        def exit(self) -> None:
            router.stop_server(self)

else:

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
