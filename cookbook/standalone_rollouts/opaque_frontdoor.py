"""Transactional FastAPI adapter for the opaque hot-load protocol.

This factory is additive until the following deployment-wiring PR switches the
Modal front door to it. Keeping the old factory live makes every stack tip
runnable while the transaction can be reviewed independently.

No ``from __future__ import annotations`` here. FastAPI must evaluate the route
handlers' factory-local ``Request`` annotations eagerly.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from cookbook.standalone_rollouts.delta_view import (
    DeltaIndexError,
    DerivedDeltaConflict,
)
from cookbook.standalone_rollouts.ledger import (
    IdentityLedger,
    LedgerConflict,
    LedgerRewind,
)
from cookbook.standalone_rollouts.opaque_protocol import (
    HotLoadRequestError,
    parse_hot_load_payload,
    pool_state_from_server_infos,
)


HOT_LOAD_PATH = "/hot_load/v1/models/hot_load"
INFERENCE_PATHS = (
    "/generate",
    "/v1/chat/completions",
    "/v1/completions",
)


def create_opaque_frontdoor_app(
    *,
    ledger: IdentityLedger,
    save_ledger: Callable[[dict[str, Any]], Awaitable[None]],
    derive_delta: Callable[..., Awaitable[None]],
    advance_to: Callable[[int], Awaitable[None]],
    list_server_infos: Callable[[], Awaitable[list[dict[str, Any]]]],
    proxy: Callable[..., Awaitable[Any]],
    authorize: Callable[[Any], Any],
    wake: Callable[[int], Awaitable[None]] | None = None,
):
    """Build the authenticated singleton front door from injected I/O."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    advance_lock = asyncio.Lock()
    current_ledger = ledger
    ledger_write_blocked = False

    def _auth(request: Request):
        return authorize(request.headers)

    def _error(status: int, error_type: str, message: str) -> Response:
        return JSONResponse(
            {"error": {"type": error_type, "message": message}}, status_code=status
        )

    @app.post(HOT_LOAD_PATH, response_model=None)
    async def post_hot_load(request: Request) -> Response:
        nonlocal current_ledger, ledger_write_blocked
        rejected = _auth(request)
        if rejected is not None:
            return rejected
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 - malformed JSON is a customer error
            return _error(400, "InvalidRequest", "request body must be a JSON object")
        try:
            signal = parse_hot_load_payload(payload)
        except HotLoadRequestError as exc:
            return _error(400, type(exc).__name__, str(exc))

        async with advance_lock:
            if ledger_write_blocked:
                return _error(
                    503,
                    "LedgerStateUncertain",
                    "a prior ledger save had an uncertain outcome; restart to recover durable state",
                )
            candidate = IdentityLedger.from_dict(
                current_ledger.to_dict(),
                expected_base_identity=current_ledger.base_identity,
            )
            try:
                if signal.is_delta:
                    assert signal.previous_snapshot_identity is not None
                    assert signal.formats is not None
                    result = candidate.append_delta(
                        signal.identity,
                        signal.previous_snapshot_identity,
                        signal.formats,
                    )
                    await derive_delta(result.entry, committed=not result.is_new)
                else:
                    result = candidate.confirm_base(signal.identity)
            except LedgerRewind as exc:
                return _error(409, "WeightRewindRejected", str(exc))
            except LedgerConflict as exc:
                return _error(409, type(exc).__name__, str(exc))
            except FileNotFoundError as exc:
                return _error(409, "CheckpointNotFound", str(exc))
            except DerivedDeltaConflict as exc:
                return _error(409, type(exc).__name__, str(exc))
            except DeltaIndexError as exc:
                return _error(400, type(exc).__name__, str(exc))

            if result.is_new:
                # An exception can arrive after the backend durably publishes.
                # Poison writes until startup recovery reloads the transport
                # rather than guessing whether the candidate committed.
                ledger_write_blocked = True
                try:
                    await save_ledger(candidate.to_dict())
                except BaseException:
                    raise
                current_ledger = candidate
                ledger_write_blocked = False

            # Pointer failures are unambiguous: the ledger already committed, so
            # memory stays advanced and an exact retry can repair latest.
            await advance_to(result.entry.version)

        if wake is not None:
            try:
                await wake(result.entry.version)
            except Exception:  # noqa: BLE001 - wake is only a latency optimization
                pass
        return JSONResponse(
            {
                "accepted": True,
                "identity": result.entry.identity,
                "current_snapshot_identity": result.entry.identity,
                "version": result.entry.version,
                "already_current": not result.is_new,
            }
        )

    @app.get(HOT_LOAD_PATH, response_model=None)
    async def get_hot_load(request: Request) -> Response:
        rejected = _auth(request)
        if rejected is not None:
            return rejected
        infos = await list_server_infos()
        return JSONResponse(
            pool_state_from_server_infos(infos, current_ledger).to_dict()
        )

    async def proxy_inference(request: Request) -> Response:
        rejected = _auth(request)
        if rejected is not None:
            return rejected
        return await proxy(request, request.url.path.lstrip("/"))

    for path in INFERENCE_PATHS:
        app.add_api_route(path, proxy_inference, methods=["POST"], response_model=None)

    return app
