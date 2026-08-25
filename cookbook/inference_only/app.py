"""Inference-only Stitch pool on Modal, assembled on the stitch core.

``EXPERIMENT_CONFIG`` selects a config module under ``cookbook.inference_only.configs``.
There is no trainer: the Server (sglang + stitch sidecar) is the shared common one,
and a later publisher can advance ``latest`` through the configured store.

Prepare the checkpoint once first (a separate app, so prep never spins up the
rollout Server floor — see ``cookbook.inference_only.prep_app``), then launch:

    EXPERIMENT_CONFIG=glm5_2_fp8 uv run --extra modal python -m cookbook.inference_only.launch
"""

from __future__ import annotations

import importlib
import os

import modal

from cookbook.common import router, server, serving_image, storage
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
from cookbook.inference_only.checkpoint import (
    claim_boot_pointer as _guarded_claim_boot_pointer,
    ensure_boot_pointer,
    require_checkpoint,
)

EXPERIMENT = os.environ[
    "EXPERIMENT_CONFIG"
]  # required; a default would silently serve the wrong experiment

exp = importlib.import_module(f"cookbook.inference_only.configs.{EXPERIMENT}")
modal_cfg = exp.modal
STARTUP_TIMEOUT = getattr(exp, "SERVER_STARTUP_TIMEOUT", SERVER_STARTUP_TIMEOUT)

# Per-run id, minted by cookbook.inference_only.launch. The same identity scopes
# the pool, Stitch pointer, publications, and logs.
RUN_ID = os.environ["RUN_ID"]
APP_NAME = f"{exp.APP_NAME}-{RUN_ID}"
RUN_DIR = STITCH_PATH / RUN_ID
STORE_DEPLOYMENT = storage.StoreDeployment.from_environment()
STORE_SECRETS = STORE_DEPLOYMENT.modal_secrets()

# Flash autoscaler target / sglang concurrency cap: explicit target_inputs required.
if modal_cfg.rollout_target_inputs is None:
    raise ValueError("rollout_target_inputs must be set in the experiment config")
ROLLOUT_CONCURRENCY = modal_cfg.rollout_target_inputs

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
    "huggingface-cache", create_if_missing=True, version=modal_cfg.hf_cache_volume_version
)
checkpoint_volume = modal.Volume.from_name(
    exp.CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
)
run_volume = modal.Volume.from_name(
    exp.EXPERIMENT_VOLUME_NAME, create_if_missing=True, version=2
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

app = modal.App(APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": getattr(
        exp, "SERVED_MODEL_NAME", str(exp.ROLLOUT_CHECKPOINT_PATH)
    ),
    **(
        {}
        if "--cuda-graph-config" in exp.SGLANG_SERVER_ARGS
        else {"--cuda-graph-max-bs-decode": str(ROLLOUT_CONCURRENCY)}
    ),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    **exp.SGLANG_SERVER_ARGS,
}


# The rollout Server is a thin module-level class whose lifecycle delegates to
# the shared common.server logic: sglang plus the stitch sidecar.
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
    startup_timeout=STARTUP_TIMEOUT,
)
class Server:
    @modal.enter()
    def startup(self) -> None:
        STORE_DEPLOYMENT.bootstrap_credentials()
        store_config = STORE_DEPLOYMENT.hook_config(APP_NAME)
        require_checkpoint(exp.ROLLOUT_CHECKPOINT_PATH)
        store = storage.create_store(
            store_config["stitch_store_backend"],
            local_root=RUN_DIR,
            run_id=RUN_ID,
            volume_name=exp.EXPERIMENT_VOLUME_NAME,
            s3_root=store_config.get("stitch_s3_root"),
            s3_endpoint_url=store_config.get("stitch_s3_endpoint_url"),
        )
        ensure_boot_pointer(store, RUN_ID)
        server.serve_startup(
            self,
            model_name=str(exp.ROLLOUT_CHECKPOINT_PATH),
            sglang_args=SGLANG_SERVER_ARGS,
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
            startup_timeout=STARTUP_TIMEOUT,
            version_lease_ttl=getattr(exp, "SIDECAR_VERSION_LEASE_TTL", None),
            lease_header=getattr(exp, "SIDECAR_LEASE_HEADER", None),
        )

    @modal.exit()
    def stop(self) -> None:
        server.serve_stop(self)


# One-shot boot-pointer claim. The claim writes through the store, and the
# default ModalVolumeStore is only reachable where the run volume is mounted —
# so it runs as a function in this app (invoked remotely by the launcher right
# after deploy), never in the local launch process. Replicas stay read-only on
# the pointer: they only refresh + wait in ``ensure_boot_pointer``.
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
    """Claim this run's boot pointer through the same store factory the sidecar uses."""
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
    _guarded_claim_boot_pointer(store, RUN_ID)


# The router is two CPU Flash classes in this same app, so it deploys and dies
# with the pool; the GPU class keeps ``Server``, inference traffic enters through
# ``Router``.
router_image = router.build_router_image(
    EXPERIMENT,
    RUN_ID,
    extra_env=STORE_DEPLOYMENT.image_environment,
)
session_routes = router.session_routes_dict(APP_NAME)
consolidation = router.consolidation_dict(APP_NAME)


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
        router.serve_registry(
            self,
            app_name=APP_NAME,
            upstream_cls="Server",
            rollout_concurrency=ROLLOUT_CONCURRENCY,
            consolidation=consolidation,
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
    """Front door for inference traffic: session-affinity routing across Server replicas,
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
