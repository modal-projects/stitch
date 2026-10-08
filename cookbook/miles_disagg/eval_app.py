"""One offline eval point as its own Modal app: a pool serving one saved checkpoint, and
a driver that runs the eval spec's dataset through the evaluated recipe's harness on it.

Selected by EXPERIMENT_CONFIG (the recipe), EVAL_CONFIG (the spec) and EVAL_RUN /
EVAL_VERSION / EVAL_VIEW (the point); ``cookbook.miles_disagg.eval_launch`` sets them.

The pool is the recipe's pool configuration for the view and boots the checkpoint
directly. Its sidecar watches a store of the point's own on the eval volume, which
nothing publishes to, so the pool serves that checkpoint for its whole life. Training's
volumes mount read-only.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
from typing import Any

import modal

from cookbook.common import server, serving_image, storage
from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    DRAFT_PATH,
    HF_CACHE_PATH,
    MINUTES,
    SERVER_STARTUP_TIMEOUT,
    SGLANG_CACHE_PATH,
    STITCH_PATH,
)
from cookbook.miles_disagg import evaluation, trainer_image
from cookbook.miles_disagg.config import validate_recipe

EXPERIMENT = os.environ["EXPERIMENT_CONFIG"]
EVAL_CONFIG = os.environ["EVAL_CONFIG"]
exp = importlib.import_module(f"cookbook.miles_disagg.configs.{EXPERIMENT}")
validate_recipe(exp)
spec = importlib.import_module(f"cookbook.miles_disagg.eval_configs.{EVAL_CONFIG}")
POINT = evaluation.EvalPoint(
    experiment=EXPERIMENT,
    run_id=os.environ.get("EVAL_RUN") or None,
    version=int(os.environ["EVAL_VERSION"]),
    view=os.environ["EVAL_VIEW"],
)
# The view's serving configuration; EVAL_ENGINES resizes its fixed fleet for speed.
# EVAL_SESSIONS_PER_ENGINE lowers each engine's load: at long contexts a full engine
# decodes slowly enough that ordinary turns reach the turn time limit.
POOL = spec.POOLS[POINT.view]
if engines := os.environ.get("EVAL_ENGINES"):
    POOL = dataclasses.replace(
        POOL, min_containers=int(engines), max_containers=int(engines)
    )
if sessions := os.environ.get("EVAL_SESSIONS_PER_ENGINE"):
    POOL = dataclasses.replace(POOL, target_inputs=int(sessions))
# EVAL_SOFT_WATCHDOG_TIMEOUT makes SGLang log its running batch and a py-spy stack of
# each stalled process after that many seconds without progress, without killing it,
# so an engine that hangs records why before the sidecar watchdog replaces it.
if soft_watchdog := os.environ.get("EVAL_SOFT_WATCHDOG_TIMEOUT"):
    POOL = dataclasses.replace(
        POOL,
        sglang_args={**POOL.sglang_args, "--soft-watchdog-timeout": soft_watchdog},
    )
APP_NAME = evaluation.app_name(spec.NAME, exp.APP_NAME, POINT)
POINT_DIR = STITCH_PATH / evaluation.results_path(spec, POINT)
if (dump_dir := os.environ.get("MODAL_SWE_TRAJECTORY_DUMP_DIR")) and not (
    os.path.normpath(dump_dir).startswith(f"{STITCH_PATH}/")
):
    raise ValueError(
        f"MODAL_SWE_TRAJECTORY_DUMP_DIR={dump_dir!r} must be under {STITCH_PATH}, "
        "the eval volume's mount in the driver"
    )
STORE = storage.StoreDeployment(backend=storage.MODAL_VOLUME, s3_secret_name=None)
# Baked into both images so a container's re-import selects the same point.
POINT_ENVIRONMENT = {
    "EXPERIMENT_CONFIG": EXPERIMENT,
    "EVAL_CONFIG": EVAL_CONFIG,
    "EVAL_RUN": POINT.run_id or "",
    "EVAL_VERSION": str(POINT.version),
    "EVAL_VIEW": POINT.view,
    "EVAL_ENGINES": os.environ.get("EVAL_ENGINES", ""),
    "EVAL_SESSIONS_PER_ENGINE": os.environ.get("EVAL_SESSIONS_PER_ENGINE", ""),
    "EVAL_SOFT_WATCHDOG_TIMEOUT": os.environ.get("EVAL_SOFT_WATCHDOG_TIMEOUT", ""),
    **STORE.image_environment,
    # Only a traced launch sets it, so other points' images are unchanged.
    **(
        {"STITCH_TRACE_REQUESTS": os.environ["STITCH_TRACE_REQUESTS"]}
        if os.environ.get("STITCH_TRACE_REQUESTS")
        else {}
    ),
    # Only a launch that records trajectories for the sampler benchmark sets it: a
    # directory on the eval volume (mounted at /stitch in the driver) where every
    # episode writes its trajectory. The agent's Ray workers inherit it.
    **(
        {"MODAL_SWE_TRAJECTORY_DUMP_DIR": os.environ["MODAL_SWE_TRAJECTORY_DUMP_DIR"]}
        if os.environ.get("MODAL_SWE_TRAJECTORY_DUMP_DIR")
        else {}
    ),
}

server_image = serving_image.build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    extra_env={**(getattr(exp, "SGLANG_SERVER_ENV", None) or {}), **POINT_ENVIRONMENT},
    runtime=getattr(exp, "SGLANG_RUNTIME", serving_image.DEFAULT_SGLANG_RUNTIME),
)
driver_image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    miles_repo_ref=getattr(exp, "MILES_REPO_REF", trainer_image.MILES_REPO_REF),
    extra_pip_packages=tuple(getattr(exp, "TRAINER_EXTRA_PIP_PACKAGES", ())),
    image_run_commands=getattr(exp, "TRAINER_IMAGE_RUN_COMMANDS", ()),
    # Ray reads RAY_DEDUP_LOGS once, at import: every worker's abort lines, not one
    # per message across the cluster.
    extra_env={**POINT_ENVIRONMENT, "RAY_DEDUP_LOGS": "0"},
)

eval_volume = modal.Volume.from_name(
    evaluation.EVAL_VOLUME_NAME, create_if_missing=True, version=2
)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", version=2).read_only()
checkpoint_volume = modal.Volume.from_name("miles-checkpoints", version=2).read_only()
data_volume = modal.Volume.from_name("miles-data", version=2).read_only()
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
)
pool_volumes: dict[str, Any] = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(CHECKPOINTS_PATH): checkpoint_volume,
    str(STITCH_PATH): eval_volume,
    SGLANG_CACHE_PATH: sglang_cache_volume,
}
if not POINT.is_base:
    pool_volumes[str(evaluation.SOURCE_RUN_PATH)] = modal.Volume.from_name(
        exp.EXPERIMENT_VOLUME_NAME, version=2
    ).read_only()
if exp.modal.draft_volume:
    pool_volumes[str(DRAFT_PATH)] = modal.Volume.from_name(
        exp.modal.draft_volume, environment_name=exp.modal.draft_volume_env
    ).read_only()

app = modal.App(APP_NAME)


class _EvalServer:
    @modal.enter()
    def startup(self) -> None:
        os.environ.update(POOL.environment)
        checkpoint = evaluation.checkpoint_dir(
            exp, POINT, source_run_root=evaluation.SOURCE_RUN_PATH
        )
        server.serve_startup(
            self,
            model_name=str(checkpoint),
            boot_version=POINT.version,
            sglang_args={
                "--served-model-name": exp.miles.hf_checkpoint,
                "--trust-remote-code": "",
                **POOL.sglang_args,
            },
            concurrency=POOL.target_inputs,
            bulletin_root=str(STITCH_PATH / APP_NAME),
            local_checkpoint_dir=exp.LOCAL_CHECKPOINT_PATH,
            delta_update_mode=exp.SGLANG_DELTA_UPDATE_MODE,
            store_backend=STORE.backend,
            volume_name=evaluation.EVAL_VOLUME_NAME,
            s3_root=None,
            s3_endpoint_url=None,
            commit_mode=exp.SIDECAR_COMMIT_MODE,
            flush_cache_on_commit=exp.SIDECAR_FLUSH_CACHE_ON_COMMIT,
            run_id=APP_NAME,
            startup_timeout=SERVER_STARTUP_TIMEOUT,
            engine_health_timeout=getattr(exp, "SIDECAR_ENGINE_HEALTH_TIMEOUT", 30.0),
            weight_view=POINT.view,
        )

    @modal.exit()
    def stop(self) -> None:
        server.serve_stop(self)


# Modal resolves a Server class by name at module scope, as for the training pools.
_server_cls = type(
    POOL.name,
    (_EvalServer,),
    {"__module__": __name__, "__qualname__": POOL.name},
)
globals()[POOL.name] = _server_cls
globals()[POOL.name] = app.server(
    **server.modal_server_options(exp.modal, POOL),
    image=server_image,
    volumes=pool_volumes,
)(_server_cls)


@app.function(
    image=driver_image,
    # Miles' argument parser imports Megatron, whose Transformer Engine loads the CUDA
    # driver library; the client computes nothing on the GPU. An H100 host leaves room
    # for the memory limit: the session servers grow with the tokens a point serves, and
    # long-response checkpoints outgrew 256 GiB.
    gpu="H100",
    cpu=(32.0, 64.0),
    memory=(65_536, 1_048_576),
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(CHECKPOINTS_PATH): checkpoint_volume,
        str(DATA_PATH): data_volume,
        str(STITCH_PATH): eval_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=24 * 60 * MINUTES,
    # A retry reruns the point from scratch; one covers a lost container.
    retries=modal.Retries(max_retries=1, initial_delay=10.0),
    include_source=False,
)
def drive(manifest: dict, smoke: list[int] | None = None) -> dict:
    from cookbook.miles_disagg import eval_driver
    from stitch.pools.modal_flash import ModalFlashPool

    return eval_driver.run(
        exp=exp,
        spec=spec,
        point=POINT,
        pool=POOL,
        pool_url=ModalFlashPool(APP_NAME, POOL.name).gateway_url(),
        point_dir=POINT_DIR,
        manifest=manifest,
        commit=eval_volume.commit,
        smoke=(smoke[0], smoke[1]) if smoke else None,
    )
