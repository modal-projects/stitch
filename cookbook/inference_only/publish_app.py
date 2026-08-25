"""Single-writer publisher: ``/publish``, ``/job``, ``/status``.

Materialization is async (``.spawn`` a Modal function) because a synchronous
multi-hundred-GB copy inside the FastAPI handler dies at Modal ingress
(~18 min 502) and Flash then drains the container mid-copy. Separate app
from the inference pool so it scales independently.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from modal import exception as modal_exc
from pydantic import BaseModel

from cookbook.common import storage
from cookbook.common.constants import CHECKPOINTS_PATH, MINUTES, STITCH_PATH
from cookbook.inference_only.publish_materialize import (
    _delta_index_metadata,
    _ensure_safetensors_index,
    _prepare_version_dir,
    _target_is_valid,
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
    # Accepted for API compat; full-vs-delta is derived from the index's
    # `delta_encoding` key by stitch.publish, not from this flag.
    delta: bool = False
    # Skip the fleet-mixed gate (e.g. republish after the operator confirms the
    # mixed state is expected). The gate is a safety rail, not a lock.
    force: bool = False


class StatusResponse(BaseModel):
    latest_version: int | None
    staged_versions: list[int]


def _build_store(run_id: str) -> Store:
    """Build the run's store exactly the way the pool does (same root and namespace)."""
    if run_id != RUN_ID:
        # A store for a foreign run shares this publisher's local root and S3
        # namespace, so its pointer writes would land on the LIVE run — and a
        # cross-run pointer move bypasses the rewind guard as a "reset".
        raise ValueError(
            f"refusing to build a store for run_id {run_id!r}; "
            f"this publisher owns run {RUN_ID!r}"
        )
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


def _require_own_run_id(run_id: str) -> None:
    """Reject requests addressed at a foreign run (HTTP 400).

    Every job spawned here writes THIS run's volume and pointer; a body run_id
    that doesn't match the publisher's env RUN_ID would rewrite the live run's
    pointer from a foreign request (and bypass the rewind guard as a reset).
    """
    if run_id != RUN_ID:
        raise HTTPException(
            status_code=400,
            detail=f"run_id {run_id!r} does not match this publisher's run {RUN_ID!r}",
        )


def _is_already_published(store: Store, version: int) -> bool:
    """No-op check keyed on the store pointer, refreshed first for cross-host visibility.

    Keyed on the pointer — NOT on version_dir existence — so a staged dir left
    behind by a failed job never masks a publish that still needs to run.
    """
    store.refresh()
    pointer = store.read_pointer()
    return pointer is not None and pointer.version >= version


def _reject_full_publish_in_cpu_mode(version_dir: Path, version: int) -> None:
    """Refuse a FULL publish when the run is in cpu (delta-only) update mode.

    cpu mode applies XOR deltas in place; a FULL publish wedges the run
    (replicas ERROR-loop post-pointer). A dir is FULL when its index has no
    ``delta_encoding`` — including a MISSING index (the materialize job's
    ``_ensure_safetensors_index`` would generate a non-delta one). Shared by the
    /publish handler (pre-spawn, on the source) and the materialize job
    (post-index, pre-pointer) so the two checks cannot drift.
    """
    if getattr(exp, "SGLANG_DELTA_UPDATE_MODE", "disk") != "cpu":
        return
    if _delta_index_metadata(version_dir) is not None:
        return
    index_path = version_dir / "model.safetensors.index.json"
    raise RuntimeError(
        f"cpu update mode is delta-only: refusing to publish FULL version {version} "
        f"(no delta_encoding in {index_path}); publish a delta or use a disk-mode run"
    )


def _function_call_from_id(job_id: str) -> modal.FunctionCall:
    """Look up a spawned job by id (module-level so tests can stub it)."""
    return modal.FunctionCall.from_id(job_id)


# get() surfaces a REMOTE job failure as one of these modal wrappers, or as the
# deserialized remote exception itself (an arbitrary non-modal class). Any OTHER
# modal exception is a client-side query error (connection, auth, unknown call
# id) — transient, not a job failure.
_REMOTE_FAILURE_ERRORS = (
    modal_exc.RemoteError,
    modal_exc.ExecutionError,
    modal_exc.InternalFailure,
    modal_exc.FunctionTimeoutError,
    modal_exc.UserCodeException,
)


def _job_state(job_id: str) -> dict[str, Any]:
    """Non-blocking poll of a spawned materialization job.

    get(timeout=0) raises modal TimeoutError while the job is still running and
    re-raises the remote exception when the job failed. Transient query errors
    (lookup or poll) report 'unknown' — NOT 'failure' — so polling clients keep
    polling a healthy job.
    """
    try:
        call = _function_call_from_id(job_id)
    except Exception as exc:  # noqa: BLE001
        return {"job_id": job_id, "status": "unknown", "error": repr(exc)}
    try:
        result = call.get(timeout=0)
    # modal 1.5.4 raises *builtins* TimeoutError from get(timeout=0) while the
    # spawn is still running; keep the modal class too in case that changes.
    except (TimeoutError, modal_exc.TimeoutError):
        return {"job_id": job_id, "status": "pending"}
    except _REMOTE_FAILURE_ERRORS as exc:
        return {"job_id": job_id, "status": "failure", "error": repr(exc)}
    except modal_exc.Error as exc:
        return {"job_id": job_id, "status": "unknown", "error": repr(exc)}
    except Exception as exc:  # noqa: BLE001 — deserialized remote exception
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
    if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME:
        # See dirs staged by OTHER containers (e.g. a prior /fabricate job)
        # before deciding whether the target already exists or is valid.
        run_volume.reload()
    if source_path.resolve() != version_dir.resolve():
        action = _prepare_version_dir(version_dir, source_path)
        logger.info(
            "materialized source for version %d: %s (%s)", version, version_dir, action
        )

    _ensure_safetensors_index(version_dir, version)

    if publish:
        # Defense in depth: /publish checks the source pre-spawn, but a FULL dir
        # here (e.g. the index generated above) must never move the pointer.
        _reject_full_publish_in_cpu_mode(version_dir, version)
        publish_version.local(run_id=run_id, model_dir=version_dir)
        logger.info("published version %d via stitch", version)
    elif STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME:
        # Make the staged dir visible to the later /publish job's container
        # (fabricate_delta_version already does the same).
        run_volume.commit()

    return {"version": version, "path": str(version_dir), "published": publish}


# ── Fleet-mixed publish gate ─────────────────────────────────────────────────

# Sync states in which a replica is mid-transition toward a newer version;
# publishing into that fleet would strand the in-flight rollout.
# Lowercase because infos are compared via ``.lower()`` (sidecar casing varies).
_FLEET_TRANSITION_STATES = frozenset({"holding", "staging", "committing"})


def fleet_is_mixed(infos: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """True when replicas disagree on applied_version, or any replica is
    holding/staging/committing toward a newer target. Empty ``infos`` is
    not mixed (publisher must work standalone).
    """
    applied_versions = sorted(
        {info.get("applied_version") for info in infos},
        key=lambda v: (v is None, v),
    )
    offenders: list[dict[str, Any]] = []
    for info in infos:
        sync_state = info.get("sync_state")
        if not isinstance(sync_state, str):
            continue
        if sync_state.lower() not in _FLEET_TRANSITION_STATES:
            continue
        target = info.get("target_version")
        if target is None:
            continue
        applied = info.get("applied_version")
        if target > (applied if applied is not None else -1):
            offenders.append(
                {
                    "sync_state": sync_state,
                    "target_version": target,
                    "applied_version": applied,
                }
            )
    detail: dict[str, Any] = {
        "applied_versions": applied_versions,
        "transitioning_replicas": offenders,
    }
    return len(applied_versions) > 1 or bool(offenders), detail


async def _default_server_info_provider() -> list[dict[str, Any]]:
    """Discover pool replicas and GET each sidecar's /server_info (200s only)."""
    import httpx

    from stitch.pools.modal_flash import ModalFlashPool

    try:
        pool = ModalFlashPool(app_name=POOL_APP_NAME, cls_name=POOL_CLS_NAME)
        urls = await pool.discover_replicas_async()
    except Exception as exc:  # noqa: BLE001
        # Pool down/undiscoverable == empty probe, and an empty probe is NOT
        # mixed — the publisher must still work standalone (no 500 here).
        logger.warning("replica discovery failed; treating as empty probe: %s", exc)
        return []

    async def fetch(client: Any, url: str) -> dict[str, Any] | None:
        try:
            resp = await client.get(f"{url}/server_info")
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("server_info probe failed for %s: %s", url, exc)
        return None

    async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
        results = await asyncio.gather(*(fetch(client, url) for url in urls))
    return [info for info in results if info is not None]


def _default_volume_reloader() -> None:
    """Reload the run volume's view (same guard as ``materialize_version``)."""
    if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME:
        run_volume.reload()


class PublisherServer:
    """FastAPI app plus uvicorn lifecycle; instantiable without a container."""

    def __init__(
        self,
        *,
        store: Store,
        run_dir: Path,
        port: int = PUBLISHER_PORT,
        server_info_provider: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
        volume_reloader: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self.run_dir = run_dir
        self.updates_dir = _updates_dir(run_dir)
        self.port = port
        self._server_info_provider = server_info_provider or _default_server_info_provider
        self._volume_reloader = volume_reloader or _default_volume_reloader
        # In-memory single-writer guard. A Flash recycle resets it; the store
        # pointer's rewind guard is the durable backstop.
        self._current_job_id: str | None = None

    def _refresh_volume_state(self) -> None:
        """Reload the volume view and refresh the store BEFORE reading volume state.

        The long-lived server container never reloads its volume view on its
        own; without this, dirs staged by spawned job containers are invisible
        to /publish and /status (HIT LIVE: 400
        "base_version 4 not staged" seconds after v4 was staged).
        """
        self._volume_reloader()
        self.store.refresh()

    def _job_busy(self) -> bool:
        """True while a spawned job is still running."""
        if self._current_job_id is None:
            return False
        # 'unknown' (transient poll/lookup error) keeps the guard: the job may
        # still be running, and clearing the guard could admit a second writer.
        if _job_state(self._current_job_id)["status"] in ("pending", "unknown"):
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
            self._refresh_volume_state()
            _require_own_run_id(request.run_id)
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

            # force=True bypasses; an empty probe (pool down) is not mixed.
            if not request.force:
                infos = await self._server_info_provider()
                mixed, detail = fleet_is_mixed(infos)
                if mixed:
                    logger.warning(
                        "rejecting /publish for version %d: fleet mixed (%s)",
                        request.version,
                        detail,
                    )
                    return JSONResponse(
                        status_code=409,
                        content={
                            "status": "retryable",
                            "reason": "fleet mixed",
                            "detail": detail,
                        },
                    )

            source_path = Path(request.source)
            if not source_path.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"source directory not found: {source_path}",
                )

            try:
                _reject_full_publish_in_cpu_mode(source_path, request.version)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

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

            latest_version is the store POINTER (refreshed first for cross-host
            visibility); the on-disk dir listing is exposed separately as
            staged_versions, so a partial dir left by a recycled mid-copy
            container can never masquerade as latest.
            """
            self._refresh_volume_state()
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
