"""Front-door hot-load adapter for the standalone rollout provider.

The front door is the single public entry and the single writer of the
bulletin board's monotonic ``latest`` pointer. It implements the customer's
hot-load API as a thin projection of the canonical log-as-truth design:

- ``POST /hot_load/...`` advances ``latest`` to the signalled checkpoint
  (monotonic CAS — a rewind is rejected for now; rolling the fleet back from a
  recovery anchor is the future story), then best-effort wakes the pool. The
  elastic rollout pool reconciles to ``latest`` on its own.
- ``GET /hot_load/...`` reports pool readiness by enumerating the *live*
  containers and querying each ``/server_info`` — no self-reported replica
  state, so a scaled-down container can't haunt the readiness fraction.
- Everything else is proxied to the rollout gateway.

All I/O (reading/writing ``latest``, enumerating replicas, proxying, auth) is
injected so the adapter logic is testable without Modal; ``modal_serve.py``
supplies the real implementations.

No ``from __future__ import annotations`` here: the route handlers'
``request: Request`` annotation must evaluate eagerly against the factory-local
fastapi import, or FastAPI mistakes ``request`` for a query parameter.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from stitch.protocol import (
    RolloutPoolState,
    RolloutReplicaState,
    parse_weight_identity,
    weight_identity,
)


HOT_LOAD_PATH = "/hot_load/v1/models/hot_load"


def advance_latest_decision(current_version: int, identity: str) -> dict[str, Any]:
    """Decide whether a hot-load signal should advance the monotonic ``latest``.

    Returns ``{"version": int}`` to accept, or ``{"error": {...}}`` to reject:
    an unparseable identity (``InvalidIdentity``), or one at/behind the current
    version (``WeightRewindRejected`` — a rewind would poison warm replicas; the
    future path is rolling the fleet from the nearest recovery anchor).
    """
    version = parse_weight_identity(identity)
    if version is None:
        return {
            "error": {
                "type": "InvalidIdentity",
                "message": f"identity {identity!r} is not weight_v<NNNNNN>",
            }
        }
    if version <= current_version:
        return {
            "error": {
                "type": "WeightRewindRejected",
                "message": (
                    f"latest is at version {current_version}; refusing to rewind to {version}"
                ),
                "current_version": int(current_version),
                "requested_version": int(version),
            }
        }
    return {"version": int(version)}


def pool_state_from_server_infos(infos: list[dict[str, Any]]) -> RolloutPoolState:
    """Build a pool-readiness report from live ``/server_info`` responses.

    A replica is ready when it is reachable and idle (not mid-sync, no sticky
    sync error). The trainer separately matches ``current_snapshot_identity``
    against its target, so an idle replica still on an old version is observable
    but correctly not counted toward the target.
    """
    replicas: list[RolloutReplicaState] = []
    for info in infos:
        current_version = info.get("current_version")
        sync_state = info.get("sync_state")
        last_error = info.get("last_sync_error")
        ready = sync_state == "IDLE" and not last_error
        identity = (
            weight_identity(current_version)
            if isinstance(current_version, int) and current_version >= 0
            else None
        )
        replicas.append(
            RolloutReplicaState(
                readiness=ready,
                current_version=current_version if isinstance(current_version, int) else None,
                current_snapshot_identity=identity,
                replica_id=info.get("run_id") or info.get("replica_id"),
                sync_state=sync_state,
                readiness_reason=None if ready else (last_error or sync_state or "unreachable"),
            )
        )
    return RolloutPoolState(replicas=replicas)


def create_frontdoor_app(
    *,
    read_current_version: Callable[[], Awaitable[int]],
    advance_to: Callable[[int], Awaitable[None]],
    list_server_infos: Callable[[], Awaitable[list[dict[str, Any]]]],
    proxy: Callable[..., Awaitable[Any]],
    authorize: Callable[[Any], Any] | None = None,
    wake: Callable[[int], Awaitable[None]] | None = None,
):
    """Build the front-door FastAPI app from injected I/O.

    ``read_current_version``/``advance_to`` read and (atomically, monotonically)
    write the ``latest`` pointer; ``list_server_infos`` enumerates live replicas;
    ``proxy`` forwards non-hot-load requests; ``authorize`` returns a rejection
    Response or ``None``; ``wake`` is a best-effort post-advance nudge.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    app = FastAPI()

    def _auth(request: Request):
        return authorize(request.headers) if authorize is not None else None

    @app.post(HOT_LOAD_PATH, response_model=None)
    async def post_hot_load(request: Request) -> Response:
        rejected = _auth(request)
        if rejected is not None:
            return rejected
        payload = await request.json()
        if not isinstance(payload, dict) or not payload.get("identity"):
            return JSONResponse({"error": "body.identity is required"}, status_code=400)
        identity = str(payload["identity"])
        decision = advance_latest_decision(await read_current_version(), identity)
        if "error" in decision:
            status = 400 if decision["error"]["type"] == "InvalidIdentity" else 409
            return JSONResponse(decision, status_code=status)
        version = decision["version"]
        await advance_to(version)
        if wake is not None:
            try:
                await wake(version)
            except Exception:  # noqa: BLE001 — wake is a latency optimization only
                pass
        return JSONResponse(
            {"accepted": True, "identity": identity, "current_snapshot_identity": identity}
        )

    @app.get(HOT_LOAD_PATH, response_model=None)
    async def get_hot_load(request: Request) -> Response:
        rejected = _auth(request)
        if rejected is not None:
            return rejected
        infos = await list_server_infos()
        return JSONResponse(pool_state_from_server_infos(infos).to_dict())

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catch_all(path: str, request: Request) -> Response:
        rejected = _auth(request)
        if rejected is not None:
            return rejected
        return await proxy(request, path)

    return app
