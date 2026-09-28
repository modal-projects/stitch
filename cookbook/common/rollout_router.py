"""A streaming front door for independently scaled rollout pools."""

import asyncio
import hashlib
import json
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

from .constants import STITCH_WEIGHT_VIEW_VERSIONS_HEADER

_DROP_REQUEST_HEADERS = {"connection", "content-length", "host"}
_DROP_RESPONSE_HEADERS = {"connection", "content-length", "transfer-encoding"}
_RETRYABLE_STATUS = {404, 409, 429, 502, 503, 504}


def ordered_upstreams(
    upstreams: Mapping[str, str], affinity_key: str
) -> list[tuple[str, str]]:
    """Rendezvous-hash one session across a changing set of rollout pools."""

    def score(name: str) -> bytes:
        return hashlib.blake2b(
            f"{affinity_key}\0{name}".encode(), digest_size=16
        ).digest()

    return sorted(upstreams.items(), key=lambda item: score(item[0]), reverse=True)


def create_app(
    upstreams: Mapping[str, str],
    *,
    affinity_header: str,
    upstream_affinity_header: str | None = None,
    source_header: str | None = None,
    strict_affinity: bool = False,
    weight_views: Mapping[str, str] | None = None,
    request_timeout: float = 3600.0,
):
    """Create an ASGI proxy that gives Miles one endpoint for several GPU pools."""
    import httpx
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.background import BackgroundTask

    normalized = {name: url.rstrip("/") for name, url in upstreams.items()}
    if not normalized:
        raise ValueError("rollout router requires at least one upstream")
    weight_views = dict(weight_views or {})
    if weight_views:
        if not strict_affinity:
            raise ValueError("weight-view routing requires strict affinity")
        if weight_views.keys() != normalized.keys():
            raise ValueError("every rollout upstream must identify one weight view")
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["client"] = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout, connect=30.0),
            trust_env=False,
            limits=httpx.Limits(
                max_connections=2048,
                max_keepalive_connections=256,
            ),
        )
        try:
            yield
        finally:
            await state.pop("client").aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        client = state["client"]

        async def probe(name: str, url: str) -> str | None:
            try:
                response = await client.get(f"{url}/health", timeout=10.0)
                return name if response.status_code == 200 else None
            except httpx.RequestError:
                return None

        ready = [
            name
            for name in await asyncio.gather(
                *(probe(name, url) for name, url in normalized.items())
            )
            if name is not None
        ]
        return JSONResponse(
            {"ready": bool(ready), "ready_pools": ready},
            status_code=200 if ready else 503,
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(path: str, request: Request) -> StreamingResponse:
        client = state["client"]
        body = await request.body()
        affinity_key = request.headers.get(affinity_header) or uuid.uuid4().hex
        raw_versions = request.headers.get(STITCH_WEIGHT_VIEW_VERSIONS_HEADER)
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _DROP_REQUEST_HEADERS
            and key.lower() != affinity_header.lower()
            and key.lower() != STITCH_WEIGHT_VIEW_VERSIONS_HEADER.lower()
        }
        if upstream_affinity_header is not None:
            headers[upstream_affinity_header] = affinity_key
        ordered = ordered_upstreams(normalized, affinity_key)
        if strict_affinity:
            ordered = ordered[:1]

        if weight_views:
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError
                if raw_versions is not None:
                    versions = json.loads(raw_versions)
                    selected_view = weight_views[ordered[0][0]]
                    selected_version = versions[selected_view]
                    if not isinstance(selected_version, int) or selected_version < 0:
                        raise ValueError
                    payload["weight_version"] = {
                        "min_version": selected_version,
                        "exact_version": None,
                    }
                elif "weight_version" not in payload:
                    raise ValueError
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail="invalid rollout weight-view version contract",
                ) from exc
            body = json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode()

        response = None
        selected_name = None
        for index, (name, upstream) in enumerate(ordered):
            selected_name = name
            try:
                response = await client.send(
                    client.build_request(
                        request.method,
                        f"{upstream}/{path}",
                        params=request.query_params,
                        headers=headers,
                        content=body,
                    ),
                    stream=True,
                )
            except httpx.RequestError:
                if index + 1 == len(ordered):
                    raise HTTPException(
                        status_code=502,
                        detail="all rollout pools rejected the connection",
                    ) from None
                continue
            if response.status_code not in _RETRYABLE_STATUS or index + 1 == len(
                ordered
            ):
                break
            await response.aread()
            await response.aclose()

        assert response is not None
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in _DROP_RESPONSE_HEADERS
        }
        if source_header is not None:
            response_headers[source_header] = str(selected_name)
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=response_headers,
            background=BackgroundTask(response.aclose),
        )

    return app
