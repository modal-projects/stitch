"""Provider-side hot-load API shim in front of one local SGLang server."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import shutil
import socket
import time
import urllib.parse
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from stitch.protocol import (
    RolloutPoolState,
    RolloutReplicaState,
    parse_weight_identity,
)
from stitch.servers.sglang import create_app as create_sglang_app
from stitch.sync import RolloutAdmissionGate


logger = logging.getLogger(__name__)
VERSIONED_ROUTES = frozenset({"generate", "v1/chat/completions", "v1/completions"})


@dataclass(frozen=True)
class ProviderSettings:
    upstream_url: str
    transport_root: Path | None
    s3_bucket: str | None
    s3_prefix: str
    snapshot_root: Path
    replica_id: str
    base_snapshot_identity: str
    state_ttl_seconds: float = 300.0
    api_key: str | None = None
    provider_model: str | None = None
    provider_deployment: str | None = None
    delta_load_format: str = "delta"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ProviderSettings":
        bucket, prefix = _parse_s3_uri(args.s3_uri) if args.s3_uri else (None, "")
        return cls(
            upstream_url=args.upstream_url.rstrip("/"),
            transport_root=Path(args.transport_root) if args.transport_root else None,
            s3_bucket=bucket,
            s3_prefix=prefix,
            snapshot_root=Path(args.snapshot_root),
            replica_id=args.replica_id,
            base_snapshot_identity=args.base_snapshot_identity,
            state_ttl_seconds=float(args.state_ttl_seconds),
            api_key=args.api_key,
            provider_model=args.provider_model,
            provider_deployment=args.provider_deployment,
            delta_load_format=args.delta_load_format,
        )

    def s3_prefix_for_identity(self, identity: str) -> str:
        return "/".join(part for part in (self.s3_prefix, identity) if part)

    def transport_path_for_identity(self, identity: str) -> Path | None:
        if self.transport_root is None:
            return None
        return self.transport_root / identity


class StateStore:
    """Shared desired-snapshot and per-replica state."""

    async def desired(self) -> dict[str, Any] | None:
        raise NotImplementedError

    async def set_desired(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def set_replica(self, replica_id: str, state: dict[str, Any]) -> None:
        raise NotImplementedError

    async def pool_state(self, *, state_ttl_seconds: float) -> RolloutPoolState:
        raise NotImplementedError


class InMemoryStateStore(StateStore):
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data = dict(data or {})

    async def desired(self) -> dict[str, Any] | None:
        value = self._data.get("desired")
        return dict(value) if isinstance(value, dict) else None

    async def set_desired(self, payload: dict[str, Any]) -> None:
        self._data["desired"] = dict(payload)

    async def set_replica(self, replica_id: str, state: dict[str, Any]) -> None:
        self._data[f"replicas/{replica_id}"] = dict(state)

    async def pool_state(self, *, state_ttl_seconds: float) -> RolloutPoolState:
        return _pool_state_from_items(
            self._data.items(), state_ttl_seconds=state_ttl_seconds
        )


class ModalStateStore(StateStore):
    def __init__(self, state_dict: Any) -> None:
        self._state_dict = state_dict

    async def desired(self) -> dict[str, Any] | None:
        value = await self._state_dict.get.aio("desired")
        return dict(value) if isinstance(value, dict) else None

    async def set_desired(self, payload: dict[str, Any]) -> None:
        await self._state_dict.put.aio("desired", dict(payload))

    async def set_replica(self, replica_id: str, state: dict[str, Any]) -> None:
        await self._state_dict.put.aio(f"replicas/{replica_id}", dict(state))

    async def pool_state(self, *, state_ttl_seconds: float) -> RolloutPoolState:
        return _pool_state_from_items(
            await _collect_modal_items(self._state_dict.items.aio()),
            state_ttl_seconds=state_ttl_seconds,
        )


async def _collect_modal_items(result: Any) -> list[tuple[Any, Any]]:
    if hasattr(result, "__aiter__"):
        return [(key, value) async for key, value in result]
    if inspect.isawaitable(result):
        result = await result
    return list(result)


def _pool_state_from_items(
    items: Iterable[tuple[Any, Any]], *, state_ttl_seconds: float
) -> RolloutPoolState:
    now = time.time()
    replicas: list[RolloutReplicaState] = []
    for key, value in items:
        if not str(key).startswith("replicas/") or not isinstance(value, dict):
            continue
        updated_at = float(value.get("updated_at", 0.0))
        if updated_at and now - updated_at > state_ttl_seconds:
            continue
        replicas.append(RolloutReplicaState.from_dict(value))
    return RolloutPoolState(replicas=replicas)


class ProviderShim(RolloutAdmissionGate):
    def __init__(
        self,
        *,
        settings: ProviderSettings,
        store: StateStore,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.store = store
        self.current_identity = settings.base_snapshot_identity
        self.sync_state = "IDLE"
        self.readiness = True
        self.readiness_reason: str | None = None
        self.last_sync_error: str | None = None
        self.debug_requests = False
        self._sync_lock = asyncio.Lock()
        self._watch_task: asyncio.Task[None] | None = None

    @property
    def current_version(self) -> int:
        version = _version_for_identity(
            self.settings.base_snapshot_identity, self.current_identity
        )
        return -1 if version is None else version

    async def startup_sync(self) -> None:
        await self._publish_state()
        self._watch_task = asyncio.create_task(self._watch_desired())

    async def shutdown_sync(self) -> None:
        if self._watch_task is None:
            return
        self._watch_task.cancel()
        try:
            await self._watch_task
        except asyncio.CancelledError:
            pass

    async def server_info(self) -> dict[str, Any]:
        return {
            "backend": "api-shim-hot-load",
            "current_version": self.current_version,
            "current_snapshot_identity": self.current_identity,
            "sync_state": self.sync_state,
            "last_sync_error": self.last_sync_error,
            "active_requests": self._active_requests,
            "readiness": self.readiness,
            "readiness_reason": self.readiness_reason,
        }

    async def signal_hot_load(self, payload: dict[str, Any]) -> dict[str, Any]:
        identity = str(payload["identity"])
        target = {
            "identity": identity,
            "incremental_snapshot_metadata": payload.get(
                "incremental_snapshot_metadata"
            ),
            "reset_prompt_cache": payload.get("reset_prompt_cache"),
            "created_at": time.time(),
        }
        await self.store.set_desired(target)
        if identity != self.current_identity:
            self.readiness = False
            self.readiness_reason = "queued"
            self.sync_state = "QUEUED"
            await self._publish_state()
        return {
            "accepted": True,
            "identity": identity,
            "current_snapshot_identity": self.current_identity,
            "sync_state": self.sync_state,
        }

    async def pool_state(self) -> RolloutPoolState:
        return await self.store.pool_state(
            state_ttl_seconds=self.settings.state_ttl_seconds
        )

    async def _watch_desired(self) -> None:
        while True:
            try:
                desired = await self.store.desired()
                if desired and desired.get("identity") != self.current_identity:
                    await self._sync_to(dict(desired))
                else:
                    await self._publish_state()
            except Exception as exc:  # noqa: BLE001
                self.readiness = False
                self.readiness_reason = str(exc)
                self.last_sync_error = str(exc)
                self.sync_state = "ERROR"
                await self._publish_state()
                logger.exception("hot-load sync failed")
            await asyncio.sleep(2.0)

    async def _sync_to(self, desired: dict[str, Any]) -> None:
        async with self._sync_lock:
            identity = str(desired["identity"])
            if identity == self.current_identity:
                return
            self.readiness = False
            self.readiness_reason = "downloading weights"
            self.sync_state = "PREFETCHING"
            self.last_sync_error = None
            await self._publish_state()

            snapshot_dir = self.settings.snapshot_root / identity
            await asyncio.to_thread(
                _materialize_snapshot, self.settings, identity, snapshot_dir
            )

            self.readiness_reason = "applying weights"
            self.sync_state = "COMMITTING"
            await self._publish_state()
            await self._begin_commit(lambda: self._active_requests == 0)
            try:
                await _flush_cache(self.settings.upstream_url)
                await _apply_snapshot(
                    upstream_url=self.settings.upstream_url,
                    snapshot_dir=snapshot_dir,
                    identity=identity,
                    incremental_snapshot_metadata=desired.get(
                        "incremental_snapshot_metadata"
                    ),
                    delta_load_format=self.settings.delta_load_format,
                )
                # Advance the served identity (and thus current_version) while the
                # commit gate is still held, so no request can be admitted observing
                # the stale version on already-mutated engine weights. Mirrors
                # WeightSyncManager._sync_once, which bumps current_version before
                # clearing _committing in its finally.
                self.current_identity = identity
            finally:
                await self._end_commit()

            self.readiness = True
            self.readiness_reason = None
            self.sync_state = "IDLE"
            await self._publish_state()

    async def _publish_state(self) -> None:
        current_version = _version_for_identity(
            self.settings.base_snapshot_identity, self.current_identity
        )
        state = {
            "replica_id": self.settings.replica_id,
            "readiness": self.readiness,
            "current_snapshot_identity": self.current_identity,
            "sync_state": self.sync_state,
            "readiness_reason": self.readiness_reason,
            "updated_at": time.time(),
            "metadata": {
                "last_sync_error": self.last_sync_error,
                "hostname": socket.gethostname(),
            },
        }
        if current_version is not None:
            state["current_version"] = current_version
        await self.store.set_replica(self.settings.replica_id, state)


def create_app(shim: ProviderShim):
    def register_hot_load_routes(app: Any) -> None:
        @app.post("/hot_load/v1/models/hot_load", response_model=None)
        async def post_hot_load(request: Request) -> dict[str, Any] | JSONResponse:
            error = _auth_error(shim.settings, request.headers)
            if error is not None:
                return error
            payload = await request.json()
            if not isinstance(payload, dict) or not payload.get("identity"):
                return JSONResponse(
                    {"error": "body.identity is required"}, status_code=400
                )
            return await shim.signal_hot_load(payload)

        @app.get("/hot_load/v1/models/hot_load", response_model=None)
        async def get_hot_load(request: Request) -> dict[str, Any] | JSONResponse:
            error = _auth_error(shim.settings, request.headers)
            if error is not None:
                return error
            return (await shim.pool_state()).to_dict()

    return create_sglang_app(
        shim,
        upstream_url=shim.settings.upstream_url,
        versioned_routes=VERSIONED_ROUTES,
        register_routes=register_hot_load_routes,
        include_sync_routes=False,
    )


def build_modal_state_store(name: str) -> StateStore:
    import modal

    return ModalStateStore(modal.Dict.from_name(name, create_if_missing=True))


async def _flush_cache(upstream_url: str) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.get(f"{upstream_url.rstrip('/')}/flush_cache")
        if response.status_code not in (200, 404):
            response.raise_for_status()


async def _apply_snapshot(
    *,
    upstream_url: str,
    snapshot_dir: Path,
    identity: str,
    incremental_snapshot_metadata: dict[str, Any] | None,
    delta_load_format: str,
) -> None:
    import httpx

    payload: dict[str, Any] = {
        "model_path": str(snapshot_dir),
        "weight_version": identity,
        "flush_cache": False,
    }
    if incremental_snapshot_metadata:
        payload["load_format"] = delta_load_format
        payload["files"] = sorted(
            path.name for path in snapshot_dir.glob("*.safetensors")
        )
        if not payload["files"]:
            raise RuntimeError(
                f"delta snapshot {snapshot_dir} has no safetensors files"
            )
    else:
        payload["load_format"] = "auto"

    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        response = await client.post(
            f"{upstream_url.rstrip('/')}/update_weights_from_disk", json=payload
        )
        response.raise_for_status()
        data = response.json()
        if data.get("success") is False:
            raise RuntimeError(f"SGLang rejected weight update: {data}")


def _materialize_snapshot(
    settings: ProviderSettings, identity: str, destination: Path
) -> None:
    source = settings.transport_path_for_identity(identity)
    if source is not None:
        _copy_snapshot_tree(source, destination)
        return

    if settings.s3_bucket is None:
        raise RuntimeError("Provider has no transport root or S3 URI configured")
    _download_s3_prefix(
        settings.s3_bucket,
        settings.s3_prefix_for_identity(identity),
        destination,
    )


def _copy_snapshot_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"No snapshot directory found at {source}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No snapshot files found under {source}")

    try:
        if source.resolve() == destination.resolve():
            return
    except FileNotFoundError:
        pass

    tmp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for path in files:
            rel = path.relative_to(source)
            target = tmp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if destination.exists():
            shutil.rmtree(destination)
        tmp.rename(destination)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _download_s3_prefix(bucket: str, prefix: str, destination: Path) -> None:
    import boto3

    destination.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = str(obj["Key"])
            if key.endswith("/"):
                continue
            rel = key[len(prefix.rstrip("/") + "/") :]
            if not rel:
                continue
            found = True
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            size = int(obj.get("Size", -1))
            if target.exists() and size >= 0 and target.stat().st_size == size:
                continue
            client.download_file(bucket, key, str(target))
    if not found:
        raise FileNotFoundError(f"No S3 objects found under s3://{bucket}/{prefix}/")


def _auth_error(settings: ProviderSettings, headers: Any):
    if settings.api_key:
        expected = f"Bearer {settings.api_key}"
        if headers.get("authorization") != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    if (
        settings.provider_model
        and headers.get("provider-model") != settings.provider_model
    ):
        return JSONResponse(
            {"error": "Provider-Model header does not match this deployment"},
            status_code=400,
        )
    if (
        settings.provider_deployment
        and headers.get("provider-deployment") != settings.provider_deployment
    ):
        return JSONResponse(
            {"error": "Provider-Deployment header does not match this deployment"},
            status_code=400,
        )
    return None


def _version_for_identity(base_identity: str, identity: str) -> int | None:
    if identity == base_identity:
        return 0
    return parse_weight_identity(identity)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3://bucket/prefix URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument(
        "--transport-root", default=os.environ.get("STITCH_SHIM_TRANSPORT_ROOT")
    )
    parser.add_argument(
        "--s3-uri", default=os.environ.get("STITCH_SHIM_S3_URI"), required=False
    )
    parser.add_argument("--state-dict-name", required=True)
    parser.add_argument(
        "--snapshot-root",
        default=os.environ.get("STITCH_SHIM_SNAPSHOT_ROOT", "/snapshots"),
    )
    parser.add_argument(
        "--replica-id", default=os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex
    )
    parser.add_argument(
        "--base-snapshot-identity",
        default=os.environ.get("STITCH_SHIM_BASE_SNAPSHOT_IDENTITY", "base"),
    )
    parser.add_argument("--state-ttl-seconds", type=float, default=300.0)
    parser.add_argument("--api-key", default=os.environ.get("STITCH_SHIM_API_KEY"))
    parser.add_argument(
        "--provider-model", default=os.environ.get("STITCH_SHIM_PROVIDER_MODEL")
    )
    parser.add_argument(
        "--provider-deployment",
        default=os.environ.get("STITCH_SHIM_PROVIDER_DEPLOYMENT"),
    )
    parser.add_argument(
        "--delta-load-format",
        default=os.environ.get("STITCH_SHIM_DELTA_LOAD_FORMAT", "delta"),
    )
    args = parser.parse_args()
    if not args.transport_root and not args.s3_uri:
        raise SystemExit(
            "--transport-root/STITCH_SHIM_TRANSPORT_ROOT or --s3-uri/STITCH_SHIM_S3_URI is required"
        )

    logging.basicConfig(level=logging.INFO)
    import uvicorn

    settings = ProviderSettings.from_args(args)
    shim = ProviderShim(
        settings=settings,
        store=build_modal_state_store(args.state_dict_name),
    )
    uvicorn.run(create_app(shim), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
