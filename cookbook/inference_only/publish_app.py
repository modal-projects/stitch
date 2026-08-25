"""Single-writer publisher: ``/publish``, ``/fabricate``, ``/fabricate_delta``, ``/job``, ``/status``.

Materialization is async (``.spawn`` a Modal function) because a synchronous
multi-hundred-GB copy inside the FastAPI handler dies at Modal ingress
(~18 min 502) and Flash then drains the container mid-copy. Separate app
from the inference pool so it scales independently.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from modal.exception import Error as _ModalError
from modal.exception import ExecutionError as _ModalExecutionError
from modal.exception import FunctionTimeoutError as _ModalFunctionTimeoutError
from modal.exception import InternalFailure as _ModalInternalFailure
from modal.exception import RemoteError as _ModalRemoteError
from modal.exception import TimeoutError as _ModalTimeoutError
from modal.exception import UserCodeException as _ModalUserCodeException
from pydantic import BaseModel

from cookbook.common import storage
from cookbook.common.constants import CHECKPOINTS_PATH, MINUTES, STITCH_PATH
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
        # synthetic-delta generator (zstd, xxh3)
        "xxhash",
        "zstandard",
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


class PublishRequest(BaseModel):
    run_id: str
    version: int
    source: str  # local path to the model directory
    # Accepted for API compat; full-vs-delta is derived from the index's
    # `delta_encoding` key by stitch.publish, not from this flag.
    delta: bool = False


class FabricateRequest(BaseModel):
    run_id: str
    from_version: int
    new_version: int


class FabricateDeltaRequest(BaseModel):
    run_id: str
    base_version: int  # 0 = the run's base checkpoint; else a staged version
    new_version: int
    num_tensors: int = 4


class StatusResponse(BaseModel):
    latest_version: int | None
    staged_versions: list[int]


def _updates_dir(run_dir: Path) -> Path:
    return run_dir / "updates"


def _version_dir(run_dir: Path, version: int) -> Path:
    return _updates_dir(run_dir) / f"weight_v{version:06d}"


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

    Keyed on the pointer — NOT on version_dir existence — so the fabricate-then-publish
    flow (where /fabricate creates the dir before /publish runs) still publishes.
    """
    store.refresh()
    pointer = store.read_pointer()
    return pointer is not None and pointer.version >= version


def _delta_index_metadata(version_dir: Path) -> dict[str, Any] | None:
    """Index metadata when ``version_dir`` is a delta (``delta_encoding`` set), else None."""
    index_path = version_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return None
    meta = index.get("metadata") or {}
    return meta if meta.get("delta_encoding") else None


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


def _target_is_valid(version_dir: Path, source_dir: Path) -> bool:
    """True when ``version_dir`` is a complete materialization of ``source_dir``.

    Deltas are checked against their own weight_map: a full source's shard
    superset would condemn every staged delta.
    """
    index_path = version_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    if _delta_index_metadata(version_dir) is not None:
        index = json.loads(index_path.read_text())
        delta_shards = {str(f) for f in (index.get("weight_map") or {}).values()}
        if not delta_shards:
            return False
        target_shards = {p.name for p in version_dir.glob("*.safetensors")}
        return delta_shards <= target_shards
    source_shards = {p.name: p for p in source_dir.glob("*.safetensors")}
    target_shards = {p.name: p for p in version_dir.glob("*.safetensors")}
    if not source_shards.keys() <= target_shards.keys():
        return False
    # Names alone are not enough: a container recycled mid-copy leaves a
    # TRUNCATED shard with the right name. Compare sizes against the source.
    return all(
        target_shards[name].stat().st_size == shard.stat().st_size
        for name, shard in source_shards.items()
    )


def _prepare_version_dir(version_dir: Path, source_dir: Path) -> str:
    """Materialize ``source_dir`` into ``version_dir``. A partial dir is removed,
    not resumed (copytree resume is not safe). Returns ``kept`` or ``copied``.
    """
    if version_dir.exists():
        if _target_is_valid(version_dir, source_dir):
            logger.info("version dir %s already fully materialized; keeping", version_dir)
            return "kept"
        logger.warning(
            "removing invalid partial version dir %s (missing index or shards)",
            version_dir,
        )
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True)
    for item in source_dir.iterdir():
        dest = version_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    return "copied"


def _function_call_from_id(job_id: str) -> modal.FunctionCall:
    """Look up a spawned job by id (module-level so tests can stub it)."""
    return modal.FunctionCall.from_id(job_id)


# get() surfaces a REMOTE job failure as one of these modal wrappers, or as the
# deserialized remote exception itself (an arbitrary non-modal class). Any OTHER
# modal exception is a client-side query error (connection, auth, unknown call
# id) — transient, not a job failure.
_REMOTE_FAILURE_ERRORS = (
    _ModalRemoteError,
    _ModalExecutionError,
    _ModalInternalFailure,
    _ModalFunctionTimeoutError,
    _ModalUserCodeException,
)


def _job_state(job_id: str) -> dict[str, Any]:
    """Non-blocking poll of a spawned materialization job.

    get(timeout=0) raises modal TimeoutError while the job is still running and
    re-raises the remote exception when the job failed. Transient query errors
    (lookup or poll) report 'unknown' — NOT 'failure' — so polling clients keep
    polling instead of giving up on a healthy job.
    """
    try:
        call = _function_call_from_id(job_id)
    except Exception as exc:  # noqa: BLE001
        return {"job_id": job_id, "status": "unknown", "error": repr(exc)}
    try:
        result = call.get(timeout=0)
    # modal 1.5.4 raises *builtins* TimeoutError from get(timeout=0) while the
    # spawn is still running; keep the modal class too in case that changes.
    except (TimeoutError, _ModalTimeoutError):
        return {"job_id": job_id, "status": "pending"}
    except _REMOTE_FAILURE_ERRORS as exc:
        return {"job_id": job_id, "status": "failure", "error": repr(exc)}
    except _ModalError as exc:
        return {"job_id": job_id, "status": "unknown", "error": repr(exc)}
    except Exception as exc:  # noqa: BLE001 — deserialized remote exception
        return {"job_id": job_id, "status": "failure", "error": repr(exc)}
    return {"job_id": job_id, "status": "success", "result": result}


def _ensure_safetensors_index(model_dir: Path, version: int) -> None:
    """Ensure model.safetensors.index.json exists with metadata.version=version.

    If the directory has safetensors files but no index, generate one from tensor names.
    Small single-file models won't have an index; create a minimal one. Any failure
    raises — a missing/wrong index makes stitch.publish derive a wrong manifest, so
    the request must fail loudly instead of publishing unversioned weights.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        data = json.loads(index_path.read_text())
        data.setdefault("metadata", {})["version"] = version
        index_path.write_text(json.dumps(data, indent=2))
        return

    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(
            f"no safetensors files found in {model_dir}; cannot generate index"
        )

    from safetensors import safe_open

    main_file = next(
        (f for f in safetensors_files if f.name == "model.safetensors"),
        safetensors_files[0],
    )
    # framework="numpy": only the header (tensor names) is read, and the publisher
    # image ships numpy but not torch.
    with safe_open(str(main_file), framework="numpy") as f:
        keys = list(f.keys())

    index = {
        "metadata": {"version": version},
        "weight_map": {key: main_file.name for key in keys},
    }
    index_path.write_text(json.dumps(index, indent=2))
    logger.info("generated safetensors index for %s (version=%d)", model_dir, version)


# ── Synthetic-delta fabrication (cpu update mode) ────────────────────────────

# Near-no-op perturbation: ~10 changed values per MiB of fp8 weights. The delta
# must change at least one value overall (the generator refuses an empty delta),
# but files and post-apply xxh3-128 checksums are real either way.
DELTA_TENSOR_DENSITY = 1e-5


def _touched_delta_tensors(run_dir: Path, through_version: int) -> set[str]:
    """Union of tensor names changed by staged delta versions 1..through_version.

    A chained delta (base_version > 0) is generated against the BASE checkpoint's
    bytes, which equal version-N bytes exactly for tensors no earlier delta
    touched — so selection must exclude everything any earlier delta changed.
    """
    touched: set[str] = set()
    for version in range(1, through_version + 1):
        index_path = _version_dir(run_dir, version) / "model.safetensors.index.json"
        if _delta_index_metadata(index_path.parent) is None:
            continue
        index = json.loads(index_path.read_text())
        touched.update((index.get("weight_map") or {}).keys())
    return touched


def _is_embed_or_lm_head(name: str) -> bool:
    """embed/lm_head-class tensors are never eligible for synthetic deltas."""
    return "embed_tokens" in name or "lm_head" in name


def _delta_anchor_dir(run_dir: Path, base_version: int) -> Path:
    """Raw-weight anchor for a delta on base_version.

    v0 anchors at the config's base checkpoint. A staged FULL dir anchors at
    itself. A staged DELTA dir holds XOR-encoded blobs, not raw weights, so the
    anchor falls back to the base checkpoint and tensor selection excludes
    everything earlier deltas touched (see _touched_delta_tensors).
    """
    if base_version > 0:
        staged = _version_dir(run_dir, base_version)
        if _delta_index_metadata(staged) is None:
            return staged
    anchor = getattr(exp, "ROLLOUT_CHECKPOINT_PATH", None)
    if anchor is None:
        raise ValueError(
            f"config {EXPERIMENT!r} has no ROLLOUT_CHECKPOINT_PATH; cannot anchor "
            f"a delta on base_version={base_version}"
        )
    return Path(anchor)


def _build_anchor_view(
    anchor_dir: Path,
    *,
    excluded: set[str],
    num_tensors: int,
    spec: Any,
    view_dir: Path,
) -> dict[str, list[str]]:
    """Symlink-view of ``anchor_dir`` exposing only tensors the delta may touch.

    The generator encodes every tensor in the view's weight_map, so the view
    is the first ``num_tensors`` classifiable, non-excluded names (shard then
    name order). Only selected shards are symlinked (a few GLM tensors, not a
    756GB scan). embed/lm_head are always excluded: they dominate compile scope.
    """
    from tools.profiling._synthetic_delta import _encoding_for, _safetensors_header

    index = json.loads((anchor_dir / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") or {}
    names_by_shard: dict[str, list[str]] = {}
    for name, filename in sorted(weight_map.items()):
        names_by_shard.setdefault(filename, []).append(name)

    selected: dict[str, list[str]] = {}
    remaining = num_tensors
    for filename in sorted(names_by_shard):
        if remaining <= 0:
            break
        _, header = _safetensors_header(anchor_dir / filename)
        for name in sorted(names_by_shard[filename]):
            if name in excluded:
                continue
            if _is_embed_or_lm_head(name):
                continue
            try:
                encoding = _encoding_for(
                    name=name, dtype=header[name]["dtype"], spec=spec
                )
            except ValueError:
                continue
            if encoding is None:
                continue  # immutable (e.g. static input_scale)
            selected.setdefault(filename, []).append(name)
            remaining -= 1
            if remaining <= 0:
                break
    if not selected:
        raise RuntimeError(
            f"no eligible tensors left in {anchor_dir} "
            f"({len(excluded)} excluded by earlier deltas)"
        )

    view_dir.mkdir(parents=True)
    for filename in selected:
        os.symlink(anchor_dir / filename, view_dir / filename)
    filtered = {
        "metadata": index.get("metadata") or {},
        "weight_map": {
            name: filename for filename, names in selected.items() for name in names
        },
    }
    (view_dir / "model.safetensors.index.json").write_text(json.dumps(filtered))
    return selected


def fabricate_delta_dir(
    run_dir: Path,
    base_version: int,
    new_version: int,
    num_tensors: int,
    *,
    anchor_dir: Path | None = None,
    density: float = DELTA_TENSOR_DENSITY,
    seed: int | None = None,
) -> Path:
    """Write a XOR delta as ``updates/weight_v{new_version:06d}``; do not publish.

    The generator always stamps version 1 / base 0; rewrite to the requested
    pair. Idempotent for an existing valid dir of the same version+base.
    """
    from tools.profiling._synthetic_delta import (
        SyntheticDeltaSpec,
        write_standard_delta,
    )

    version_dir = _version_dir(run_dir, new_version)
    if version_dir.exists():
        meta = _delta_index_metadata(version_dir)
        if (
            meta is not None
            and int(meta.get("version", -1)) == new_version
            and int(meta.get("base_version", -1)) == base_version
            and _target_is_valid(version_dir, version_dir)
        ):
            logger.info("delta dir %s already fabricated; keeping", version_dir)
            return version_dir
        logger.warning("replacing invalid/stale delta dir %s", version_dir)
        shutil.rmtree(version_dir)

    spec = SyntheticDeltaSpec(
        checkpoint_format=getattr(exp, "DELTA_CHECKPOINT_FORMAT", "fp8"),
        quantized_value_density=density,
        high_precision_value_density=density,
    )
    anchor = Path(anchor_dir) if anchor_dir is not None else _delta_anchor_dir(
        run_dir, base_version
    )
    excluded = _touched_delta_tensors(run_dir, base_version)

    staging = Path(tempfile.mkdtemp(prefix=f"delta-v{new_version:06d}-"))
    try:
        view_dir = staging / "anchor-view"
        selected = _build_anchor_view(
            anchor,
            excluded=excluded,
            num_tensors=num_tensors,
            spec=spec,
            view_dir=view_dir,
        )
        write_standard_delta(
            str(view_dir),
            str(staging),
            spec=spec,
            seed=seed if seed is not None else new_version,
        )
        generated = staging / "weight_v000001"
        index_path = generated / "model.safetensors.index.json"
        data = json.loads(index_path.read_text())
        data["metadata"]["version"] = new_version
        data["metadata"]["base_version"] = base_version
        index_path.write_text(json.dumps(data, indent=2))
        version_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(generated), str(version_dir))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info(
        "fabricated delta %s (base=%d, tensors=%s)",
        version_dir,
        base_version,
        sum(len(names) for names in selected.values()),
    )
    return version_dir


@app.function(
    image=publisher_image,
    volumes={str(STITCH_PATH): run_volume} if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME else {},
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
        **(
            {str(STITCH_PATH): run_volume}
            if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME
            else {}
        ),
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


@app.function(
    image=publisher_image,
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
        **(
            {str(STITCH_PATH): run_volume}
            if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME
            else {}
        ),
    },
    secrets=STORE_SECRETS,
    # Streams 8MiB chunks of the few selected anchor tensors (shards are ~5.4GB);
    # 4 cpu / 8GiB matches the materialize job and is ample for the generator's
    # per-shard thread pool. max_containers=1 serializes with the HTTP 409 guard.
    cpu=4,
    memory=8192,
    timeout=60 * MINUTES,
    max_containers=1,
)
def fabricate_delta_version(
    run_id: str, base_version: int, new_version: int, num_tensors: int
) -> dict[str, Any]:
    """Async delta-fabrication job spawned by /fabricate_delta.

    Writes updates/weight_v{new_version:06d} as a real XOR delta on base_version
    and stops — the pointer only moves when a later /publish runs for it.
    """
    version_dir = fabricate_delta_dir(
        RUN_DIR, base_version, new_version, num_tensors
    )
    if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME:
        # Make the new dir visible to the later /publish job's container.
        run_volume.commit()
    logger.info("fabricated delta version %d (base=%d)", new_version, base_version)
    return {
        "version": new_version,
        "base_version": base_version,
        "path": str(version_dir),
        "published": False,
    }


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
        volume_reloader: Callable[[], None] | None = None,
    ) -> None:
        self.store = store
        self.run_dir = run_dir
        self.updates_dir = _updates_dir(run_dir)
        self.port = port
        self._volume_reloader = volume_reloader or _default_volume_reloader
        # In-memory single-writer guard. A Flash recycle resets it; the store
        # pointer's rewind guard is the durable backstop.
        self._current_job_id: str | None = None

    def _refresh_volume_state(self) -> None:
        """Reload the volume view and refresh the store BEFORE reading volume state.

        The long-lived server container never reloads its volume view on its
        own; without this, dirs staged by spawned job containers are invisible
        to /publish, /fabricate, /fabricate_delta, and /status (HIT LIVE: 400
        "base_version 4 not staged" seconds after v4 was staged).
        """
        self._volume_reloader()
        self.store.refresh()

    def _job_busy(self) -> bool:
        """True while a previously spawned job is still running."""
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

    def _spawn_fabricate_delta(
        self, *, run_id: str, base_version: int, new_version: int, num_tensors: int
    ) -> str:
        """Spawn the async delta-fabrication job; 409 while another job runs."""
        if self._job_busy():
            raise HTTPException(
                status_code=409,
                detail=f"materialization job {self._current_job_id} still running",
            )
        call = fabricate_delta_version.spawn(
            run_id=run_id,
            base_version=base_version,
            new_version=new_version,
            num_tensors=num_tensors,
        )
        self._current_job_id = call.object_id
        logger.info(
            "spawned delta-fabrication job %s (base=%d, version=%d)",
            call.object_id,
            base_version,
            new_version,
        )
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

        @fastapi_app.post("/fabricate")
        async def fabricate(request: FabricateRequest) -> Any:
            """Spawn a copy of an existing version dir with rewritten ``metadata.version``."""
            self._refresh_volume_state()
            _require_own_run_id(request.run_id)
            from_dir = _version_dir(self.run_dir, request.from_version)
            new_dir = _version_dir(self.run_dir, request.new_version)

            if not from_dir.exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"from_version {request.from_version} not found",
                )

            job_id = self._spawn_materialize(
                run_id=request.run_id,
                version=request.new_version,
                source=str(from_dir),
                publish=False,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "accepted",
                    "job_id": job_id,
                    "version": request.new_version,
                    "path": str(new_dir),
                },
            )

        @fastapi_app.post("/fabricate_delta")
        async def fabricate_delta(request: FabricateDeltaRequest) -> Any:
            """Spawn a XOR delta on ``base_version``; the pointer moves only on a later /publish."""
            self._refresh_volume_state()
            _require_own_run_id(request.run_id)
            if request.base_version < 0:
                raise HTTPException(status_code=400, detail="base_version must be >= 0")
            if request.new_version <= request.base_version:
                raise HTTPException(
                    status_code=400,
                    detail="new_version must be greater than base_version",
                )
            if request.num_tensors < 1:
                raise HTTPException(status_code=400, detail="num_tensors must be >= 1")
            if request.base_version > 0 and not _version_dir(
                self.run_dir, request.base_version
            ).exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"base_version {request.base_version} not staged",
                )
            if request.base_version == 0:
                try:
                    anchor = _delta_anchor_dir(self.run_dir, 0)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                if not anchor.exists():
                    raise HTTPException(
                        status_code=400,
                        detail=f"base checkpoint not found: {anchor}",
                    )

            job_id = self._spawn_fabricate_delta(
                run_id=request.run_id,
                base_version=request.base_version,
                new_version=request.new_version,
                num_tensors=request.num_tensors,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "accepted",
                    "job_id": job_id,
                    "version": request.new_version,
                    "base_version": request.base_version,
                    "path": str(_version_dir(self.run_dir, request.new_version)),
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
        **(
            {str(STITCH_PATH): run_volume}
            if STORE_DEPLOYMENT.backend == storage.MODAL_VOLUME
            else {}
        ),
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
