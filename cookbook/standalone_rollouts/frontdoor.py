"""Front-door hot-load adapter for the standalone rollout provider.

The front door is the single public entry and the single writer of both the
bulletin board's monotonic ``latest`` pointer and the identity ledger. It
implements the customer's hot-load API as an abstraction layer over the
log-as-truth pool, so the customer's contract (opaque checkpoint identities,
lineage via ``previous_snapshot_identity``, delta formats in the POST body)
never leaks the pool's integer-versioned internals:

- ``POST /hot_load/...`` maps the signalled opaque ``identity`` to a monotonic
  stitch version via the :class:`~cookbook.standalone_rollouts.ledger.IdentityLedger`
  (idempotent on re-signal), normalizes the customer's POST metadata into the
  version dir's index so the disk-delta decoder can apply it, advances ``latest``,
  and best-effort wakes the pool.
- ``GET /hot_load/...`` reports readiness by enumerating the *live* containers and
  querying each ``/server_info``, translating each replica's integer version back
  to the customer's identity string so their equality match works.
- Inference (``v1/*``, ``generate``) is proxied to the rollout gateway; every
  other route is rejected (allowlist).

All I/O (ledger load/save, index normalization, pointer write, replica
enumeration, proxy, auth) is injected so the adapter is testable without Modal;
``modal_serve.py`` supplies the real implementations.

No ``from __future__ import annotations`` here: the route handlers'
``request: Request`` annotation must evaluate eagerly against the factory-local
fastapi import, or FastAPI mistakes ``request`` for a query parameter.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from cookbook.standalone_rollouts.ledger import IdentityLedger, LedgerError
from stitch.protocol import RolloutPoolState, RolloutReplicaState


HOT_LOAD_PATH = "/hot_load/v1/models/hot_load"
MAX_IDENTITY_LEN = 512
# Path components that resolve outside the identity's own upload dir, plus the
# transport's own files (the pointer and the ledger), which an identity dir
# must never shadow or clobber.
RESERVED_IDENTITIES = {".", "..", "latest", "identities.json"}


def is_customer_inference_route(path: str) -> bool:
    """Whether a proxied path is a customer-facing inference route.

    The proxy is an allowlist, not a denylist: only OpenAI-compatible ``v1/*``
    routes and the SGLang-native ``generate`` route reach the rollout pool.
    Everything else a public client could name — the per-container sidecar's own
    control routes (``rpc_sync_from_bulletin_board``, ``server_info``,
    ``get_weight_version``) and the SGLang engine routes behind it
    (``update_weights_*``, ``start_profile``, …) — is not reachable through the
    front door, so a new engine/sidecar route can never become customer-exposed
    by omission from a denylist.
    """
    route = path.strip("/")
    return route == "generate" or route == "v1" or route.startswith("v1/")


def is_valid_identity(identity: str) -> bool:
    """A customer checkpoint identity is opaque, but it becomes an S3 path
    component (``<prefix>/<identity>/``) and a ledger key, so it must be
    non-empty, bounded, single-segment (no ``/``, which would escape the prefix),
    and free of control characters. Bounding it also caps ledger/key growth from
    a hostile client."""
    if not identity or len(identity) > MAX_IDENTITY_LEN or identity in RESERVED_IDENTITIES:
        return False
    return "/" not in identity and not any(ord(c) < 0x20 for c in identity)


def delta_index_metadata(
    version: int, base_version: int, incremental: dict[str, Any]
) -> dict[str, str]:
    """The slime disk-delta ``metadata`` block the decoder requires, derived from
    the customer's POST. ``version``/``base_version`` are the ledger-assigned
    integers as zero-padded 6-digit strings (the decoder compares ``base_version``
    to the applied-version marker by string equality). ``delta_encoding`` is not
    in the customer's API — spec §3 pins XOR — so it defaults to ``xor``;
    ``compression_format``/``checksum_format`` come from the POST body (spec §2a),
    defaulting to the spec's stated ``zstd``/``adler32`` when omitted."""
    return {
        "version": f"{int(version):06d}",
        "base_version": f"{int(base_version):06d}",
        "delta_encoding": str(incremental.get("delta_encoding", "xor")),
        "compression_format": str(incremental.get("compression_format", "zstd")),
        "checksum_format": str(incremental.get("checksum_format", "adler32")),
    }


def pool_state_from_server_infos(
    infos: list[dict[str, Any]], ledger: IdentityLedger
) -> RolloutPoolState:
    """Build a pool-readiness report from live ``/server_info`` responses.

    A replica is ready when it is reachable and idle (not mid-sync, no sticky
    sync error). Each replica reports an integer ``current_version``; the ledger
    translates it back to the customer's opaque identity so their readiness match
    (``current_snapshot_identity == <signalled identity>``) works. An idle replica
    on an old version is observable but correctly not counted toward the target.
    """
    replicas: list[RolloutReplicaState] = []
    for info in infos:
        current_version = info.get("current_version")
        sync_state = info.get("sync_state")
        last_error = info.get("last_sync_error")
        ready = sync_state == "IDLE" and not last_error
        identity = (
            ledger.identity_for(current_version)
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
    load_ledger: Callable[[], Awaitable[dict[str, Any]]],
    save_ledger: Callable[[dict[str, Any]], Awaitable[None]],
    normalize_index: Callable[[str, dict[str, str]], Awaitable[None]],
    advance_to: Callable[[int], Awaitable[None]],
    list_server_infos: Callable[[], Awaitable[list[dict[str, Any]]]],
    proxy: Callable[..., Awaitable[Any]],
    authorize: Callable[[Any], Any] | None = None,
    wake: Callable[[int], Awaitable[None]] | None = None,
):
    """Build the front-door FastAPI app from injected I/O.

    ``load_ledger``/``save_ledger`` read/write the identity↔version ledger on the
    transport; ``normalize_index(identity, metadata)`` writes the disk-delta
    metadata block into that version dir's index; ``advance_to(version)`` writes
    the ``latest`` pointer; ``list_server_infos`` enumerates live replicas;
    ``proxy`` forwards inference; ``authorize`` returns a rejection Response or
    ``None``; ``wake`` is a best-effort post-advance nudge.

    The load-record-normalize-save-advance sequence is serialized under
    ``advance_lock`` so the singleton front door never races itself. The ledger
    is loaded fresh from the transport inside the lock (not cached across POSTs),
    so a partial failure leaves the transport ledger authoritative and a retry
    converges: normalize is idempotent, save is the commit point, and the pointer
    is re-advanced to the head even on an idempotent re-signal.
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
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 — a malformed/non-JSON body is a 400, never a 500
            return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)
        if not isinstance(payload, dict) or not payload.get("identity"):
            return JSONResponse({"error": "body.identity is required"}, status_code=400)
        identity = str(payload["identity"])
        if not is_valid_identity(identity):
            return JSONResponse(
                {"error": "body.identity must be a non-empty, single-segment string <= 512 chars"},
                status_code=400,
            )
        incremental = payload.get("incremental_snapshot_metadata")
        if incremental is not None and not isinstance(incremental, dict):
            return JSONResponse(
                {"error": "body.incremental_snapshot_metadata must be an object"}, status_code=400
            )
        previous = incremental.get("previous_snapshot_identity") if incremental else None
        if incremental is not None and (not isinstance(previous, str) or not previous):
            # A delta without lineage is indistinguishable from a full snapshot
            # and would be recorded as one; require the parent explicitly.
            return JSONResponse(
                {
                    "error": "incremental_snapshot_metadata.previous_snapshot_identity "
                    "must be a non-empty string"
                },
                status_code=400,
            )

        async with advance_lock:
            ledger = IdentityLedger.from_dict(await load_ledger())
            try:
                entry, is_new = ledger.record(identity, previous)
            except LedgerError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            if entry.version != ledger.head_version:
                # Re-signalling an older identity is a rewind request the pool
                # cannot serve (versions only move forward); reject it loudly
                # rather than answering accepted:true for weights that will
                # never be reported ready.
                return JSONResponse(
                    {
                        "error": {
                            "type": "WeightRewindRejected",
                            "message": f"{identity!r} is older than the served head",
                            "current_version": ledger.head_version,
                            "requested_version": entry.version,
                        }
                    },
                    status_code=409,
                )
            if is_new:
                # A delta (incremental metadata present) needs its index normalized
                # so the decoder can apply it; a full snapshot (base) is served
                # from the booted base checkpoint and needs no delta metadata.
                # Normalize before saving/advancing, so a missing upload leaves the
                # transport ledger and pointer untouched (fail fast, self-clean).
                if incremental is not None:
                    try:
                        await normalize_index(
                            identity,
                            delta_index_metadata(
                                entry.version, ledger.base_version_for(identity), incremental
                            ),
                        )
                    except FileNotFoundError:
                        return JSONResponse(
                            {"error": f"checkpoint {identity!r} not found on the transport; upload before signalling"},
                            status_code=409,
                        )
                await save_ledger(ledger.to_dict())
            # This identity is the chain head (rewinds were rejected above).
            # Advance idempotently, so a retry after a save-succeeded/
            # advance-failed POST still moves latest.
            await advance_to(entry.version)

        if is_new and wake is not None:
            try:
                await wake(entry.version)
            except Exception:  # noqa: BLE001 — wake is a latency optimization only
                pass
        return JSONResponse(
            {
                "accepted": True,
                "identity": identity,
                "current_snapshot_identity": identity,
                "version": entry.version,
                "already_current": not is_new,
            }
        )

    @app.get(HOT_LOAD_PATH, response_model=None)
    async def get_hot_load(request: Request) -> Response:
        rejected = _auth(request)
        if rejected is not None:
            return rejected
        ledger = IdentityLedger.from_dict(await load_ledger())
        infos = await list_server_infos()
        return JSONResponse(pool_state_from_server_infos(infos, ledger).to_dict())

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def catch_all(path: str, request: Request) -> Response:
        rejected = _auth(request)
        if rejected is not None:
            return rejected
        if not is_customer_inference_route(path):
            return JSONResponse({"error": f"no such route: /{path.strip('/')}"}, status_code=404)
        return await proxy(request, path)

    return app
