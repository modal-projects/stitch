import asyncio
import json
from typing import Any

import httpx
import pytest

from cookbook.common.constants import STITCH_WEIGHT_VIEW_VERSIONS_HEADER
from cookbook.common.rollout_router import create_app, ordered_upstreams


def test_rendezvous_order_is_stable_per_session() -> None:
    upstreams = {
        "H100": "https://h100",
        "H200": "https://h200",
        "B200": "https://b200",
        "B300": "https://b300",
    }

    first = ordered_upstreams(upstreams, "session-1")
    assert ordered_upstreams(dict(reversed(upstreams.items())), "session-1") == first
    assert {item[0] for item in first} == set(upstreams)


def test_proxy_treats_request_as_the_asgi_request() -> None:
    app = create_app({"H100": "https://h100"}, affinity_header="X-Stitch-Session-ID")
    route = next(route for route in app.routes if route.path == "/{path:path}")

    assert route.dependant.query_params == []


def test_removing_pool_only_moves_sessions_assigned_to_it() -> None:
    upstreams = {
        "H100": "https://h100",
        "H200": "https://h200",
        "B300": "https://b300",
    }
    reduced = {"H100": "https://h100", "B300": "https://b300"}

    for session in ("a", "b", "c", "d", "e"):
        before = ordered_upstreams(upstreams, session)[0][0]
        after = ordered_upstreams(reduced, session)[0][0]
        if before != "H200":
            assert after == before


def test_weighted_rendezvous_routes_in_proportion_to_pool_capacity() -> None:
    upstreams = {"small": "https://small", "large": "https://large"}
    weights = {"small": 1.0, "large": 4.0}

    selected = [
        ordered_upstreams(upstreams, str(index), weights=weights)[0][0]
        for index in range(10_000)
    ]

    large_ratio = selected.count("large") / len(selected)
    assert 0.78 < large_ratio < 0.82


def test_proxy_fails_over_when_modal_pool_endpoint_is_not_ready(
    monkeypatch: Any,
) -> None:
    upstreams = {"pending": "https://pending", "ready": "https://ready"}
    affinity_key = next(
        str(index)
        for index in range(100)
        if ordered_upstreams(upstreams, str(index))[0][0] == "pending"
    )

    class UpstreamClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.requests: list[httpx.Request] = []

        def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
            return httpx.Request(method, url, **kwargs)

        async def send(
            self, request: httpx.Request, *, stream: bool = False
        ) -> httpx.Response:
            assert stream
            self.requests.append(request)
            status = 404 if request.url.host == "pending" else 200
            return httpx.Response(
                status,
                request=request,
                stream=httpx.ByteStream(b"ready"),
            )

        async def aclose(self) -> None:
            pass

    upstream = UpstreamClient()
    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)
    app = create_app(upstreams, affinity_header="X-Stitch-Session-ID")

    async def request() -> httpx.Response:
        async with app.router.lifespan_context(app):
            async with async_client(
                transport=httpx.ASGITransport(app=app), base_url="http://router"
            ) as client:
                return await client.post(
                    "/v1/chat/completions",
                    headers={"X-Stitch-Session-ID": affinity_key},
                    json={"model": "model"},
                )

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.content == b"ready"
    assert [request.url.host for request in upstream.requests] == ["pending", "ready"]


def test_strict_affinity_never_switches_weight_views_and_reports_source(
    monkeypatch: Any,
) -> None:
    upstreams = {"fp8": "https://fp8", "nvfp4": "https://nvfp4"}
    affinity_key = "trajectory-7"
    selected = ordered_upstreams(upstreams, affinity_key)[0][0]

    class UpstreamClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.requests: list[httpx.Request] = []

        def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
            return httpx.Request(method, url, **kwargs)

        async def send(
            self, request: httpx.Request, *, stream: bool = False
        ) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(
                503,
                request=request,
                stream=httpx.ByteStream(b"unavailable"),
            )

        async def aclose(self) -> None:
            pass

    upstream = UpstreamClient()
    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)
    app = create_app(
        upstreams,
        affinity_header="X-Stitch-Session-ID",
        source_header="X-Stitch-Rollout-Pool",
        strict_affinity=True,
    )

    async def request() -> httpx.Response:
        async with app.router.lifespan_context(app):
            async with async_client(
                transport=httpx.ASGITransport(app=app), base_url="http://router"
            ) as client:
                return await client.post(
                    "/v1/chat/completions",
                    headers={"X-Stitch-Session-ID": affinity_key},
                    json={"model": "model"},
                )

    response = asyncio.run(request())

    assert response.status_code == 503
    assert response.headers["X-Stitch-Rollout-Pool"] == selected
    assert [request.url.host for request in upstream.requests] == [selected]


@pytest.mark.parametrize("selected", ["h200:fp8", "b300:nvfp4"])
def test_router_requires_the_selected_weight_views_latest_version(
    monkeypatch: Any, selected: str
) -> None:
    upstreams = {"h200:fp8": "https://fp8", "b300:nvfp4": "https://nvfp4"}
    weight_views = {"h200:fp8": "fp8", "b300:nvfp4": "nvfp4"}
    affinity_key = next(
        str(index)
        for index in range(100)
        if ordered_upstreams(upstreams, str(index))[0][0] == selected
    )

    class UpstreamClient:
        def __init__(self, **_kwargs: Any) -> None:
            self.requests: list[httpx.Request] = []

        def build_request(self, method: str, url: str, **kwargs: Any) -> httpx.Request:
            return httpx.Request(method, url, **kwargs)

        async def send(
            self, request: httpx.Request, *, stream: bool = False
        ) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(
                200,
                request=request,
                stream=httpx.ByteStream(b"ok"),
            )

        async def aclose(self) -> None:
            pass

    upstream = UpstreamClient()
    async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)
    app = create_app(
        upstreams,
        affinity_header="X-Stitch-Session-ID",
        strict_affinity=True,
        weight_views=weight_views,
    )

    async def request() -> httpx.Response:
        async with app.router.lifespan_context(app):
            async with async_client(
                transport=httpx.ASGITransport(app=app), base_url="http://router"
            ) as client:
                return await client.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Stitch-Session-ID": affinity_key,
                        STITCH_WEIGHT_VIEW_VERSIONS_HEADER: json.dumps(
                            {"fp8": 12, "nvfp4": 10}
                        ),
                    },
                    json={"model": "model"},
                )

    response = asyncio.run(request())

    assert response.status_code == 200
    assert len(upstream.requests) == 1
    assert upstream.requests[0].url == upstreams[selected] + "/v1/chat/completions"
    payload = json.loads(upstream.requests[0].content)
    assert payload["weight_version"] == {
        "min_version": {"fp8": 12, "nvfp4": 10}[weight_views[selected]],
        "exact_version": None,
    }
    assert (
        STITCH_WEIGHT_VIEW_VERSIONS_HEADER.lower() not in upstream.requests[0].headers
    )
