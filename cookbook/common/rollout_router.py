"""A streaming front door for independently scaled rollout pools."""

import asyncio
import hashlib
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

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
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _DROP_REQUEST_HEADERS
            and key.lower() != affinity_header.lower()
        }
        if upstream_affinity_header is not None:
            headers[upstream_affinity_header] = affinity_key
        ordered = ordered_upstreams(normalized, affinity_key)

        response = None
        for index, (_name, upstream) in enumerate(ordered):
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
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=response_headers,
            background=BackgroundTask(response.aclose),
        )

    return app
