"""Front-door hot-load adapter for the standalone rollout provider.

The single public entry and the single writer of the ``latest`` pointer and the
identity ledger. It implements the customer's hot-load API over the
integer-versioned pool: ``POST /hot_load`` maps the opaque ``identity`` to a
minted version (idempotent on re-signal), derives the decoder index from the
POST metadata, advances ``latest``, and best-effort wakes the pool;
``GET /hot_load`` reports readiness by querying live replicas, translating each
integer version back to the customer's identity; inference (``v1/*``,
``generate``) is proxied and every other route 404s (allowlist).

All I/O is injected so the adapter is testable without Modal; ``modal_serve.py``
supplies the real implementations.

No ``from __future__ import annotations`` here: the route handlers'
``request: Request`` annotation must evaluate eagerly against the factory-local
fastapi import, or FastAPI mistakes ``request`` for a query parameter.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from cookbook.standalone_rollouts.ledger import LEDGER_FILENAME, IdentityLedger, LedgerError
from stitch.protocol import RolloutPoolState, RolloutReplicaState


HOT_LOAD_PATH = "/hot_load/v1/models/hot_load"
MAX_IDENTITY_LEN = 512
# Path components that resolve outside the identity's own upload dir, plus the
# transport's own files (the pointer and the ledger), which an identity dir
# must never shadow or clobber.
RESERVED_IDENTITIES = {".", "..", "latest", LEDGER_FILENAME}


def is_customer_inference_route(path: str) -> bool:
    """Allowlist, not denylist: only OpenAI-compatible ``v1/*`` and SGLang's
    ``generate`` reach the pool. Sidecar control routes and engine routes are
    unreachable through the front door, and a newly added one can never become
    customer-exposed by omission."""
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
    to the applied-version marker by string equality). The customer's API has no
    ``delta_encoding`` field (their contract fixes XOR) and states ``zstd``/
    ``adler32`` as the format defaults when the POST omits them."""
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
    authorize: Callable[[Any], Any],
    wake: Callable[[int], Awaitable[None]] | None = None,
    expected_base_identity: str | None = None,
):
    """Build the front-door FastAPI app from injected I/O.

    ``authorize`` (a rejection Response, or ``None`` to allow) is required — the
    app is fail-closed by construction; a test that wants an open app says so
    with an explicit allow-all. ``wake`` is a best-effort post-advance nudge.
    ``expected_base_identity``, when set, pins the one identity allowed to
    anchor a chain: a full snapshot must name it (the pool serves the booted
    base, never a customer upload), and so must a first delta's unknown parent
    (anything else is a typo, not the booted base).

    The load-record-normalize-save-advance sequence is serialized under
    ``advance_lock``, and the ledger is loaded fresh from the transport inside
    the lock, so a partial failure leaves the transport authoritative and a
    retry converges: normalize is idempotent, save is the commit point, and the
    pointer is re-advanced to the head even on an idempotent re-signal.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    app = FastAPI()
    advance_lock = asyncio.Lock()

    def _auth(request: Request):
        return authorize(request.headers)

    def _error(status: int, message) -> Any:
        return JSONResponse({"error": message}, status_code=status)

    @app.post(HOT_LOAD_PATH, response_model=None)
    async def post_hot_load(request: Request) -> Response:
        rejected = _auth(request)
        if rejected is not None:
            return rejected
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 — a malformed/non-JSON body is a 400, never a 500
            return _error(400, "request body must be a JSON object")
        if not isinstance(payload, dict) or not payload.get("identity"):
            return _error(400, "body.identity is required")
        identity = str(payload["identity"])
        if not is_valid_identity(identity):
            return _error(400, "body.identity must be a non-empty, single-segment string <= 512 chars")
        incremental = payload.get("incremental_snapshot_metadata")
        if incremental is not None and not isinstance(incremental, dict):
            return _error(400, "body.incremental_snapshot_metadata must be an object")
        previous = incremental.get("previous_snapshot_identity") if incremental else None
        if incremental is not None and (not isinstance(previous, str) or not previous):
            # A lineage-less delta would be recorded as a full snapshot.
            return _error(
                400,
                "incremental_snapshot_metadata.previous_snapshot_identity must be a non-empty string",
            )

        async with advance_lock:
            ledger = IdentityLedger.from_dict(await load_ledger())
            if expected_base_identity is not None:
                if incremental is None and identity != expected_base_identity:
                    return _error(
                        409,
                        f"this deployment serves base {expected_base_identity!r}; "
                        f"full snapshot {identity!r} cannot be activated",
                    )
                if incremental is not None and previous != expected_base_identity and ledger.version_for(previous) is None:
                    return _error(
                        409,
                        f"previous_snapshot_identity {previous!r} is neither a signalled "
                        f"checkpoint nor the deployment base {expected_base_identity!r}",
                    )
            try:
                entry, is_new = ledger.record(identity, previous)
            except LedgerError as exc:
                return _error(409, str(exc))
            if entry.version != ledger.head_version:
                # A rewind the pool cannot serve; reject loudly rather than
                # answering accepted:true for weights that never become ready.
                return _error(
                    409,
                    {
                        "type": "WeightRewindRejected",
                        "message": f"{identity!r} is older than the served head",
                        "current_version": ledger.head_version,
                        "requested_version": entry.version,
                    },
                )
            if is_new:
                # Only a delta needs a derived decoder index (a full snapshot is
                # the booted base). Derive before save/advance, so a missing or
                # malformed upload leaves ledger and pointer untouched.
                if incremental is not None:
                    try:
                        await normalize_index(
                            identity,
                            delta_index_metadata(
                                entry.version, ledger.base_version_for(identity), incremental
                            ),
                        )
                    except FileNotFoundError:
                        return _error(
                            409,
                            f"checkpoint {identity!r} not found on the transport; upload before signalling",
                        )
                    except ValueError as exc:
                        # A malformed uploaded index is the customer's to fix:
                        # 4xx, never a 500.
                        return _error(400, str(exc))
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
            return _error(404, f"no such route: /{path.strip('/')}")
        return await proxy(request, path)

    return app
