"""Single-writer publisher: ``/publish``, ``/job``, ``/status``.

Materialization is async (``.spawn`` a Modal function) because a synchronous
multi-hundred-GB copy inside the FastAPI handler dies at Modal ingress
(~18 min 502) and Flash then drains the container mid-copy. Separate app
from the inference pool so it scales independently.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cookbook.common import storage
from cookbook.common.constants import CHECKPOINTS_PATH, MINUTES, STITCH_PATH
from cookbook.inference_only.publish_materialize import (
    _ensure_safetensors_index,
    _prepare_version_dir,
    _updates_dir,
    _version_dir,
)
from stitch.stores.base import Store

logger = logging.getLogger(__name__)

EXPERIMENT = os.environ["EXPERIMENT_CONFIG"]
exp = importlib.import_module(f"cookbook.inference_only.configs.{EXPERIMENT}")
# Must match the pool's run (app.py requires it too): the publish wake targets the
# pool's Servers, and a default here would silently wake the wrong app.
RUN_ID = os.environ["RUN_ID"]
APP_NAME = f"{exp.APP_NAME}-publisher-{RUN_ID}"
POOL_APP_NAME = f"{exp.APP_NAME}-{RUN_ID}"
POOL_CLS_NAME = "Server"
RUN_DIR = STITCH_PATH / RUN_ID
STORE_DEPLOYMENT = storage.StoreDeployment.from_environment()
STORE_SECRETS = STORE_DEPLOYMENT.modal_secrets()
# Port the publisher's uvicorn binds; must match the @app.server(port=...) below —
# Modal Flash waits for this port and kills the container if nothing listens.
PUBLISHER_PORT = 8001

publisher_image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "httpx",
        # numpy is safe_open's header-read framework; the image has no torch.
        "numpy",
        "safetensors>=0.4.0",
        "uvicorn",
    )
    .env({"EXPERIMENT_CONFIG": EXPERIMENT, "RUN_ID": RUN_ID})
    .add_local_python_source("stitch")
    .add_local_python_source("cookbook")
    .add_local_dir(
        "tools",
        remote_path="/root/tools",
        ignore=["**/__pycache__", "**/*.pyc"],
    )
)

app = modal.App(APP_NAME)

run_volume = modal.Volume.from_name(
    exp.EXPERIMENT_VOLUME_NAME, create_if_missing=True, version=2
)
# Prep writes model snapshots here; the publisher only reads them.
checkpoint_volume = modal.Volume.from_name(
    exp.CHECKPOINT_VOLUME_NAME, create_if_missing=True, version=2
)


def _run_volume_mounts() -> dict[str, modal.Volume]:
    """The run store's volume mount — empty when the store backend isn't a Modal volume."""
    if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME:
        return {str(STITCH_PATH): run_volume}
    return {}


class PublishRequest(BaseModel):
    run_id: str
    version: int
    source: str  # local path to the model directory


class StatusResponse(BaseModel):
    latest_version: int | None
    staged_versions: list[int]


def _build_store(run_id: str) -> Store:
    """Build the run's store exactly the way the pool does (same root and namespace)."""
    STORE_DEPLOYMENT.bootstrap_credentials()
    # Namespace on the POOL app name so the S3 root matches the pool's store.
    store_config = STORE_DEPLOYMENT.hook_config(POOL_APP_NAME)
    return storage.create_store(
        store_config["stitch_store_backend"],
        local_root=RUN_DIR,
        run_id=run_id,
        volume_name=exp.EXPERIMENT_VOLUME_NAME,
        s3_root=store_config.get("stitch_s3_root"),
        s3_endpoint_url=store_config.get("stitch_s3_endpoint_url"),
    )


def _is_already_published(store: Store, version: int) -> bool:
    """No-op check keyed on the store pointer, refreshed first for cross-host visibility.

    Keyed on the pointer — NOT on version_dir existence — so a staged dir left
    behind by a failed job never masks a publish that still needs to run.
    """
    store.refresh()
    pointer = store.read_pointer()
    return pointer is not None and pointer.version >= version


def _function_call_from_id(job_id: str) -> modal.FunctionCall:
    """Look up a spawned job by id (module-level so tests can stub it)."""
    return modal.FunctionCall.from_id(job_id)


def _job_state(job_id: str) -> dict[str, Any]:
    """Non-blocking poll of a spawned materialization job.

    get(timeout=0) raises TimeoutError while the job is still running and
    re-raises the remote exception when the job failed.
    """
    try:
        call = _function_call_from_id(job_id)
        result = call.get(timeout=0)
    # modal 1.5.4 raises *builtins* TimeoutError from get(timeout=0) while the
    # spawn is still running.
    except TimeoutError:
        return {"job_id": job_id, "status": "pending"}
    except Exception as exc:  # noqa: BLE001 — remote failure surfaces here
        return {"job_id": job_id, "status": "failure", "error": repr(exc)}
    return {"job_id": job_id, "status": "success", "result": result}


@app.function(
    image=publisher_image,
    volumes=_run_volume_mounts(),
    secrets=STORE_SECRETS,
    cpu=2,
    memory=2048,
    timeout=30 * MINUTES,
    max_containers=1,
)
def publish_version(run_id: str, model_dir: Path) -> None:
    """Materialize a model directory into the versioned updates store.

    Called locally by the publisher after /publish request validation. The version
    and full/delta kind come from the directory's HF index (stitch derives them).
    """
    import stitch.publish
    from stitch.pools.modal_flash import ModalFlashPool

    store = _build_store(run_id)
    # Wake the POOL app's Servers, not this publisher app.
    pool = ModalFlashPool(app_name=POOL_APP_NAME, cls_name=POOL_CLS_NAME)

    stitch.publish.publish_version(store, pool, str(model_dir), run_id=run_id)


@app.function(
    image=publisher_image,
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
        **_run_volume_mounts(),
    },
    secrets=STORE_SECRETS,
    # The copy is page-cache/IO bound; cpu>=4 keeps ~1GB/s volume throughput and
    # the 4h timeout covers a ~756GB source with headroom. max_containers=1
    # serializes jobs even if the HTTP-side 409 guard is bypassed (recycle).
    cpu=4,
    memory=8192,
    timeout=4 * 60 * MINUTES,
    max_containers=1,
)
def materialize_version(run_id: str, version: int, source: str, publish: bool) -> dict[str, Any]:
    """Async materialization job: copy, index-stamp, optionally publish.

    Runs off the Flash server container so a multi-hundred-GB copy survives
    ingress timeouts and publisher-container drains.
    """
    source_path = Path(source)
    version_dir = _version_dir(RUN_DIR, version)
    if source_path.resolve() != version_dir.resolve():
        action = _prepare_version_dir(version_dir, source_path)
        logger.info(
            "materialized source for version %d: %s (%s)", version, version_dir, action
        )

    _ensure_safetensors_index(version_dir, version)

    if publish:
        publish_version.local(run_id=run_id, model_dir=version_dir)
        logger.info("published version %d via stitch", version)

    return {"version": version, "path": str(version_dir), "published": publish}


class PublisherServer:
    """FastAPI app plus uvicorn lifecycle; instantiable without a container."""

    def __init__(
        self,
        *,
        store: Store,
        run_dir: Path,
        port: int = PUBLISHER_PORT,
    ) -> None:
        self.store = store
        self.run_dir = run_dir
        self.updates_dir = _updates_dir(run_dir)
        self.port = port
        # In-memory single-writer guard. A Flash recycle resets it; the store
        # pointer's rewind guard is the durable backstop.
        self._current_job_id: str | None = None

    def _job_busy(self) -> bool:
        """True while a spawned job is still running."""
        if self._current_job_id is None:
            return False
        if _job_state(self._current_job_id)["status"] == "pending":
            return True
        self._current_job_id = None
        return False

    def _spawn_materialize(self, *, run_id: str, version: int, source: str, publish: bool) -> str:
        """Spawn the async materialization job; 409 while another job runs."""
        if self._job_busy():
            raise HTTPException(
                status_code=409,
                detail=f"materialization job {self._current_job_id} still running",
            )
        call = materialize_version.spawn(
            run_id=run_id, version=version, source=source, publish=publish
        )
        self._current_job_id = call.object_id
        logger.info("spawned materialization job %s (version=%d)", call.object_id, version)
        return call.object_id

    def build_app(self) -> Any:
        fastapi_app = FastAPI()

        @fastapi_app.post("/publish")
        async def publish(request: PublishRequest) -> Any:
            """Validate and spawn materialize+publish; 200 no-op if the pointer is already past ``version``."""
            version_dir = _version_dir(self.run_dir, request.version)

            # Pointer-keyed no-op: a fabricated dir must NOT mask publishing.
            if _is_already_published(self.store, request.version):
                logger.info(
                    "store pointer already at or past version %d; returning no-op",
                    request.version,
                )
                return {
                    "status": "no-op",
                    "reason": "store pointer already at or past version",
                    "version": request.version,
                    "path": str(version_dir),
                }

            source_path = Path(request.source)
            if not source_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"source directory not found: {source_path}",
                )

            job_id = self._spawn_materialize(
                run_id=request.run_id,
                version=request.version,
                source=request.source,
                publish=True,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "accepted",
                    "job_id": job_id,
                    "version": request.version,
                    "path": str(version_dir),
                },
            )

        @fastapi_app.get("/job/{job_id}")
        async def job(job_id: str) -> dict[str, Any]:
            """GET /job/{job_id} -> pending | success | failure (non-blocking)."""
            return _job_state(job_id)

        @fastapi_app.get("/status")
        async def status() -> StatusResponse:
            """GET /status

            latest_version is the store POINTER; the on-disk dir listing is
            exposed separately as staged_versions, so a partial dir left by a
            recycled mid-copy container can never masquerade as latest.
            """
            pointer = self.store.read_pointer()

            staged: list[int] = []
            if self.updates_dir.exists():
                for d in sorted(self.updates_dir.iterdir()):
                    if d.is_dir() and d.name.startswith("weight_v"):
                        try:
                            staged.append(int(d.name.replace("weight_v", "")))
                        except ValueError:
                            pass

            return StatusResponse(
                latest_version=pointer.version if pointer is not None else None,
                staged_versions=staged,
            )

        return fastapi_app

    def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self.build_app(), host="0.0.0.0", port=self.port, timeout_keep_alive=300
        )
        self.uvicorn_server = uvicorn.Server(config)
        self.server_thread = threading.Thread(
            target=self.uvicorn_server.run, daemon=True
        )
        self.server_thread.start()

    def stop(self) -> None:
        self.uvicorn_server.should_exit = True
        self.server_thread.join()


@app.server(
    image=publisher_image,
    cpu=2,
    memory=2048,
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
        **_run_volume_mounts(),
    },
    secrets=STORE_SECRETS,
    # Single-writer: exactly one publisher, kept warm so /publish never cold-starts.
    min_containers=1,
    max_containers=1,
    startup_timeout=10 * MINUTES,
    port=PUBLISHER_PORT,
    unauthenticated=True,
)
class Publisher:
    """Single-writer publisher server."""

    @modal.enter()
    def startup(self) -> None:
        STORE_DEPLOYMENT.bootstrap_credentials()
        _updates_dir(RUN_DIR).mkdir(parents=True, exist_ok=True)
        store = _build_store(RUN_ID)
        # Flash kills the container if nothing binds `port` during startup, so the
        # HTTP listener must come up here in @modal.enter, not lazily.
        self._server = PublisherServer(store=store, run_dir=RUN_DIR, port=PUBLISHER_PORT)
        self._server.start()
        logger.info(
            "Publisher serving on port %d, updates directory: %s",
            PUBLISHER_PORT,
            self._server.updates_dir,
        )

    @modal.exit()
    def shutdown(self) -> None:
        server = getattr(self, "_server", None)
        if server is not None:
            server.stop()
