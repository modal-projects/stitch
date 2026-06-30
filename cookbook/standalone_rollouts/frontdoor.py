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

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from stitch.protocol import (
    PointerRewind,
    RolloutPoolState,
    RolloutReplicaState,
    decide_pointer_move,
    format_snapshot_identity,
    parse_weight_identity,
)


HOT_LOAD_PATH = "/hot_load/v1/models/hot_load"


def advance_latest_decision(
    current_run_id: str | None,
    current_version: int,
    identity: str,
    request_run_id: str | None,
) -> dict[str, Any]:
    """Decide whether a hot-load signal should advance the ``latest`` pointer.

    Run-id partitioning makes this idempotent across runs. A signal from a
    *different* run (``request_run_id != current_run_id``) is a fresh chain whose
    version space restarts at 1, so it is accepted and resets the version space
    (``"reset": True``) — accepting v1 after a finished run reached v5 is the
    intended begin-a-new-run path, not a rewind. Within the same run (and the
    run-less customer layout where both are None) the monotonic CAS still applies.

    Returns ``{"run_id": str|None, "version": int, "reset": bool}`` to accept, or
    ``{"error": {...}}`` to reject (``InvalidIdentity`` / ``WeightRewindRejected``).
    The accept/reset/rewind call is :func:`stitch.protocol.decide_pointer_move`,
    the same rule the bulletin-board publish path advances through; this wrapper
    only parses the wire identity and shapes the error dict. An explicit claim is
    just a signal at ``weight_v000000`` with a fresh ``run_id`` (a cross-run move,
    so ``reset=True``).
    """
    version = parse_weight_identity(identity)
    if version is None:
        return {
            "error": {
                "type": "InvalidIdentity",
                "message": f"identity {identity!r} is not weight_v<NNNNNN>",
            }
        }
    try:
        move = decide_pointer_move(
            current_run_id, current_version, run_id=request_run_id, version=version
        )
    except PointerRewind as rewind:
        return {
            "error": {
                "type": "WeightRewindRejected",
                "message": str(rewind),
                "current_version": rewind.current_version,
                "requested_version": rewind.requested_version,
            }
        }
    return {"run_id": move.run_id, "version": move.version, "reset": move.reset}


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
        current_run_id = info.get("current_run_id")
        sync_state = info.get("sync_state")
        last_error = info.get("last_sync_error")
        ready = sync_state == "IDLE" and not last_error
        # The snapshot identity carries the run, so a replica still serving a
        # finished run's same-numbered version isn't counted ready for a new run.
        identity = (
            format_snapshot_identity(current_run_id, current_version)
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
    read_current_pointer: Callable[[], Awaitable[tuple[str | None, int]]],
    advance_to: Callable[[str | None, int], Awaitable[None]],
    list_server_infos: Callable[[], Awaitable[list[dict[str, Any]]]],
    proxy: Callable[..., Awaitable[Any]],
    authorize: Callable[[Any], Any] | None = None,
    wake: Callable[[int], Awaitable[None]] | None = None,
):
    """Build the front-door FastAPI app from injected I/O.

    ``read_current_pointer`` returns the active ``(run_id, version)``;
    ``advance_to(run_id, version)`` atomically writes the single self-identifying
    ``latest`` pointer. ``list_server_infos`` enumerates live replicas; ``proxy``
    forwards non-hot-load requests; ``authorize`` returns a rejection Response or
    ``None``; ``wake`` is a best-effort post-advance nudge.

    The read-decide-advance sequence is serialized so the singleton front door
    never races itself into a transient rewind (two POSTs both reading the same
    current version and advancing out of order).
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    app = FastAPI()
    advance_lock = asyncio.Lock()

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
        request_run_id = payload.get("run_id")
        async with advance_lock:
            current_run_id, current_version = await read_current_pointer()
            decision = advance_latest_decision(
                current_run_id, current_version, identity, request_run_id
            )
            if "error" in decision:
                status = 400 if decision["error"]["type"] == "InvalidIdentity" else 409
                return JSONResponse(decision, status_code=status)
            await advance_to(decision["run_id"], decision["version"])
        snapshot_identity = format_snapshot_identity(decision["run_id"], decision["version"])
        if wake is not None:
            try:
                await wake(decision["version"])
            except Exception:  # noqa: BLE001 — wake is a latency optimization only
                pass
        return JSONResponse(
            {
                "accepted": True,
                "identity": identity,
                "current_snapshot_identity": snapshot_identity,
                "run_id": decision["run_id"],
            }
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
