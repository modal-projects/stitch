"""Router harness: the pure routing helpers — ``route_session`` stickiness / overload
fallback / TTL eviction, ``select_underloaded_container``, ``filter_headers``, and
``_container_addr`` — against a fake session-routes dict (no Modal involved)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import httpx

from cookbook.common import router
from cookbook.common.router import (
    SESSION_ROUTE_TTL_SECONDS,
    ContainerInfo,
    ContainerInfoList,
    RouteEntry,
    RouteEntryList,
    _container_addr,
    _ProxyApp,
    _RegistryApp,
    filter_headers,
    relay_lease_header,
    route_session,
    select_underloaded_container,
    session_id_from_headers,
)


class _Image:
    def __init__(self) -> None:
        self.environment: dict[str, str] = {}

    def pip_install(self, *_packages: str) -> _Image:
        return self

    def env(self, values: dict[str, str]) -> _Image:
        self.environment.update(values)
        return self

    def add_local_python_source(self, _module: str) -> _Image:
        return self

    def add_local_dir(self, *_args, **_kwargs) -> _Image:
        return self


class _SyncAsync:
    """Duck-types a Modal SDK synchronicity-wrapped method (callable, with ``.aio``)."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)

    async def aio(self, *args):
        return self.fn(*args)


class FakeRoutes:
    """Stands in for the session-routes ``modal.Dict``; stores the dumped JSON forms."""

    def __init__(self) -> None:
        self.store: dict[str, list] = {}
        self.get = _SyncAsync(lambda key, default: self.store.get(key, default))
        self.put = _SyncAsync(lambda key, value: self.store.__setitem__(key, value))


def _containers(*loads: int) -> dict[str, ContainerInfo]:
    return {
        f"ta-{i}": ContainerInfo(task_id=f"ta-{i}", upstream=f"h{i}:8000", load=load)
        for i, load in enumerate(loads)
    }


def _seeded(routes: FakeRoutes, session: str, entries: list[dict]) -> None:
    routes.store[session] = RouteEntryList.dump_python(
        [RouteEntry(**entry) for entry in entries], mode="json"
    )


def test_router_image_preserves_store_environment(monkeypatch) -> None:
    image = _Image()
    monkeypatch.setattr(router.modal.Image, "debian_slim", lambda: image)

    router.build_router_image(
        "experiment",
        "run-a",
        extra_env={"STITCH_STORE_BACKEND": "s3"},
    )

    assert image.environment == {
        "STITCH_STORE_BACKEND": "s3",
        "EXPERIMENT_CONFIG": "experiment",
        "RUN_ID": "run-a",
    }


def test_route_session_pins_only_available_underloaded_replica() -> None:
    routes, containers = FakeRoutes(), _containers(0, 5, 5)
    picked = asyncio.run(route_session(routes, "s1", containers, 4))
    assert picked.task_id == "ta-0"
    assert routes.store["s1"][0]["task_id"] == "ta-0"


def test_route_session_is_sticky_for_known_healthy_replica() -> None:
    routes, containers = FakeRoutes(), _containers(0, 5, 5)
    first = asyncio.run(route_session(routes, "s1", containers, 4))
    # ta-1 becomes strictly less loaded, but stickiness holds while the pinned
    # replica stays below the configured overload threshold.
    containers["ta-0"] = containers["ta-0"].model_copy(update={"load": 3})
    containers["ta-1"] = containers["ta-1"].model_copy(update={"load": 0})
    second = asyncio.run(route_session(routes, "s1", containers, 4))
    assert second.task_id == first.task_id == "ta-0"


def test_route_session_sheds_overloaded_replica_to_replica_with_headroom() -> None:
    routes, containers = FakeRoutes(), _containers(0, 0, 20)
    first = asyncio.run(route_session(routes, "s1", containers, 10))
    containers[first.task_id] = containers[first.task_id].model_copy(
        update={"load": 20}
    )
    second = asyncio.run(route_session(routes, "s1", containers, 10))
    assert second.task_id != first.task_id
    assert second.load == 0


def test_route_session_drops_expired_and_undiscovered_routes() -> None:
    routes, containers = FakeRoutes(), _containers(0, 5, 5)
    _seeded(
        routes,
        "s1",
        [
            {
                "task_id": "ta-2",
                "last_sent": time.time() - SESSION_ROUTE_TTL_SECONDS - 1,
            },
            {"task_id": "ta-gone", "last_sent": time.time()},
        ],
    )
    picked = asyncio.run(route_session(routes, "s1", containers, 4))
    assert picked.task_id == "ta-0"
    assert [entry["task_id"] for entry in routes.store["s1"]] == ["ta-0"]


def test_select_underloaded_container_spreads_across_headroom(monkeypatch) -> None:
    containers = _containers(7, 2, 3, 2)
    candidates = []

    def choose(options):
        candidates.extend(options)
        return options[0]

    monkeypatch.setattr("cookbook.common.router.random.choice", choose)
    picked = select_underloaded_container(containers, overload_threshold=4)
    assert picked.load < 4
    assert {container.task_id for container in candidates} == {"ta-1", "ta-2", "ta-3"}


def test_session_and_lease_headers() -> None:
    headers = {
        "Host": "example",
        "Content-Length": "10",
        "modal-session-id": "s1",
        "modal-flash-upstream": "h:8000",
        "X-Forwarded-For": "1.2.3.4",
        "Authorization": "Bearer tok",
        "content-type": "application/json",
    }
    assert filter_headers(headers) == {
        "Authorization": "Bearer tok",
        "content-type": "application/json",
    }
    assert filter_headers(
        {
            "stitch-exact-version": "3",
            "modal-session-id": "s1",
            "Stitch-Lease-Key": "forged",
        }
    ) == {"stitch-exact-version": "3"}
    assert relay_lease_header({"content-type": "application/json"}, "s1") == {
        "content-type": "application/json",
        "Stitch-Lease-Key": "s1",
    }
    assert relay_lease_header({}, None) == {}
    assert session_id_from_headers({"Modal-Session-ID": "modal-s"}) == "modal-s"
    assert session_id_from_headers({}) is None


def test_container_addr_normalizes_host_and_port() -> None:
    assert _container_addr({"task_id": "t", "host": "h", "port": 8000}) == "h:8000"
    assert _container_addr({"host": "h:8000", "port": 8000}) == "h:8000"
    assert _container_addr({"host": "h"}) == "h"
    assert _container_addr({}) is None
    assert (
        _container_addr(SimpleNamespace(host="h", port=8000, task_id="t")) == "h:8000"
    )


def test_upstream_stream_reserves_capacity_before_connecting() -> None:
    container = _containers(0)["ta-0"]

    class FakeResponse:
        async def aiter_raw(self):
            yield b"ok"

    class FakeStream:
        load_on_enter: int | None = None

        async def __aenter__(self):
            self.load_on_enter = container.load
            return FakeResponse()

        async def __aexit__(self, *args):
            return None

    class FakeClient:
        def __init__(self) -> None:
            self.stream_context = FakeStream()

        def stream(self, **kwargs):
            return self.stream_context

    async def run() -> None:
        app = _ProxyApp(
            registry_url="https://registry",
            upstream_url="https://upstream",
            session_routes=FakeRoutes(),
            overload_threshold=4,
        )
        client = FakeClient()
        app.client = client
        stream = app.upstream_stream(
            SimpleNamespace(method="POST", query_params={}),
            "v1/chat/completions",
            {},
            b"{}",
            container,
            "request",
        )

        await anext(stream)
        assert client.stream_context.load_on_enter == 1
        assert container.load == 1
        assert [chunk async for chunk in stream] == [b"ok"]
        assert container.load == 0

    asyncio.run(run())


def _leased(
    task_id: str, load: int, applied_version: int | None, leases: dict[str, int]
) -> ContainerInfo:
    return ContainerInfo(
        task_id=task_id,
        upstream=f"{task_id}:8000",
        load=load,
        applied_version=applied_version,
        leases=leases,
    )


def test_route_session_version_boundary() -> None:
    """Version preference, saturation queues, ready-excluded, 409-no-repin —
    the pinned-session boundary."""
    containers = {
        "ta-0": ContainerInfo(
            task_id="ta-0", upstream="h0:8000", load=0, applied_version=1
        ),
        "ta-1": ContainerInfo(
            task_id="ta-1", upstream="h1:8000", load=0, applied_version=2
        ),
    }
    routes = FakeRoutes()
    picked = asyncio.run(route_session(routes, "s1", containers, 4, exact_version=2))
    assert picked.task_id == "ta-1", "version preference picks the matching replica"
    assert picked.applied_version == 2

    routes = FakeRoutes()
    picked = asyncio.run(
        route_session(routes, "s1", {"ta-0": containers["ta-0"]}, 4, exact_version=99)
    )
    assert picked is None, "no version match returns None"
    assert "s1" not in routes.store, "no match must not re-pin"

    routes = FakeRoutes()
    _seeded(routes, "s1", [{"task_id": "ta-0", "last_sent": time.time()}])
    picked = asyncio.run(route_session(routes, "s1", containers, 4, exact_version=2))
    assert picked.task_id == "ta-1", "sticky version mismatch is skipped"

    routes = FakeRoutes()
    picked = asyncio.run(
        route_session(
            routes,
            "s1",
            {
                "ta-0": _leased("ta-0", load=0, applied_version=2, leases={"2": 7}),
                "ta-1": _leased("ta-1", load=0, applied_version=None, leases={}),
            },
            4,
            exact_version=1,
        )
    )
    assert picked is None, "no applied_version match returns None"
    assert "s1" not in routes.store

    routes = FakeRoutes()
    picked = asyncio.run(
        route_session(
            routes,
            "s1",
            {
                "ta-0": _leased("ta-0", load=9, applied_version=1, leases={"1": 3}),
                "ta-1": _leased("ta-1", load=12, applied_version=1, leases={"1": 5}),
                "ta-2": _leased("ta-2", load=0, applied_version=2, leases={"2": 1}),
            },
            4,
            exact_version=1,
        )
    )
    assert picked.task_id == "ta-0", "saturated matching replica queues rather than 409s"
    assert routes.store["s1"][0]["task_id"] == "ta-0"

    ready = {
        "ta-0": _leased("ta-0", 0, 1, {"1": 1}).model_copy(update={"ready": False}),
        "ta-1": _leased("ta-1", 3, 1, {"1": 1}),
    }
    for _ in range(5):
        picked = asyncio.run(route_session(FakeRoutes(), "s1", ready, 4))
        assert picked.task_id == "ta-1", "not-ready replicas are excluded"
    ready["ta-1"] = ready["ta-1"].model_copy(update={"ready": False})
    routes = FakeRoutes()
    assert asyncio.run(route_session(routes, "s2", ready, 4)) is None
    assert routes.store == {}, "all-not-ready yields no pin"

    routes = FakeRoutes()
    _seeded(routes, "s1", [{"task_id": "ta-0", "last_sent": time.time()}])
    picked = asyncio.run(
        route_session(
            routes,
            "s1",
            {
                "ta-0": _leased("ta-0", 0, 1, {"1": 1}).model_copy(
                    update={"ready": False}
                ),
                "ta-1": _leased("ta-1", 0, 1, {"1": 1}),
            },
            4,
        )
    )
    assert picked.task_id == "ta-1", "sticky not-ready route is deleted"
    assert [entry["task_id"] for entry in routes.store["s1"]] == ["ta-1"]

    fleet: dict[str, ContainerInfo] = {}
    assert (
        asyncio.run(route_session(FakeRoutes(), "s1", fleet, 4, exact_version=1))
        is None
    ), "empty fleet yields None, not IndexError"
    assert asyncio.run(route_session(FakeRoutes(), "s1", fleet, 4)) is None


def test_container_info_list_json_round_trip() -> None:
    original = [
        _leased("ta-0", load=1, applied_version=2, leases={"2": 3, "1": 1}).model_copy(
            update={"sync_state": "HOLDING", "target_version": 3, "draining": True}
        )
    ]
    payload = ContainerInfoList.dump_python(original, mode="json")
    assert payload[0]["leases"] == {"2": 3, "1": 1}
    assert payload[0]["sync_state"] == "HOLDING"
    assert payload[0]["target_version"] == 3
    assert payload[0]["draining"] is True
    parsed = ContainerInfoList.validate_python(payload)
    assert parsed[0].leases == {"2": 3, "1": 1}
    assert parsed[0].sync_state == "HOLDING"
    assert parsed[0].target_version == 3
    assert parsed[0].draining is True


class _FakeResp:
    def __init__(self, payload: dict, *, ok: bool = True) -> None:
        self._payload = payload
        self._ok = ok

    def raise_for_status(self) -> None:
        if not self._ok:
            raise RuntimeError("HTTP 503")

    def json(self) -> dict:
        return self._payload


def _registry_client(
    server_info: dict | Exception, loads: dict | Exception
) -> object:
    class FakeClient:
        async def get(self, url: str, **_kwargs) -> _FakeResp:
            if url.endswith("/server_info"):
                if isinstance(server_info, Exception):
                    raise server_info
                return _FakeResp(server_info)
            if url.endswith("/v1/loads"):
                if isinstance(loads, Exception):
                    raise loads
                return _FakeResp(loads)
            raise AssertionError(f"unexpected url: {url}")

    return FakeClient()


def _loads_payload(waiting: int, running: int) -> dict:
    return {"loads": [{"num_waiting_reqs": waiting, "num_running_reqs": running}]}


def _patch_discovery(monkeypatch, task_ids: list[str]) -> None:
    async def discover(_app_name: str, _cls: str) -> list[dict]:
        return [{"task_id": t, "host": f"h-{t}", "port": 8000} for t in task_ids]

    monkeypatch.setattr(router, "list_flash_containers_async", discover)


def test_registry_poll_membership_vs_load(monkeypatch) -> None:
    """Loads failure must not drop membership; server_info failure does."""
    registry = _RegistryApp(app_name="app", upstream_cls="Server")
    registry.client = _registry_client(
        {
            "applied_version": 3,
            "leases": {"3": 2, "1": 1},
            "sync_state": "HOLDING",
            "target_version": 4,
        },
        _loads_payload(1, 2),
    )
    poll = asyncio.run(registry._poll_container("h0:8000"))
    assert poll.load == 3
    assert poll.applied_version == 3
    assert poll.leases == {"3": 2, "1": 1}
    assert poll.sync_state == "HOLDING"
    assert poll.target_version == 4

    registry.client = _registry_client(
        {"applied_version": 3, "sync_state": "ERROR", "target_version": 5},
        _loads_payload(0, 0),
    )
    poll = asyncio.run(registry._poll_container("h0:8000"))
    assert poll.target_version is None, (
        "an errored replica has no target: a leftover target_version is nulled"
    )

    registry.client = _registry_client(
        {"applied_version": 1, "ready": False}, _loads_payload(0, 0)
    )
    assert asyncio.run(registry._poll_container("h0:8000")).ready is False
    registry.client = _registry_client({"applied_version": 1}, _loads_payload(0, 0))
    assert asyncio.run(registry._poll_container("h0:8000")).ready is True

    registry.client = _registry_client(
        {"applied_version": 3, "leases": {"3": 1}}, RuntimeError("loads down")
    )
    poll = asyncio.run(registry._poll_container("h0:8000"))
    assert poll.load is None, "loads failure keeps membership (load=None)"
    assert poll.applied_version == 3
    registry.client = _registry_client(
        RuntimeError("server_info down"), _loads_payload(0, 0)
    )
    try:
        asyncio.run(registry._poll_container("h0:8000"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("server_info failure must propagate (drops the replica)")

    _patch_discovery(monkeypatch, ["ta-0"])
    registry = _RegistryApp(app_name="app", upstream_cls="Server")
    registry.client = _registry_client(
        {"applied_version": 1, "leases": {}}, _loads_payload(1, 2)
    )
    [info] = asyncio.run(registry._poll_once())
    assert (info.load, info.load_stale) == (3, False)
    registry.client = _registry_client(
        {"applied_version": 1, "leases": {}}, RuntimeError("loads down")
    )
    [info] = asyncio.run(registry._poll_once())
    assert (info.load, info.load_stale) == (3, True), "loads fail keeps last-known load"

    registry.client = _registry_client(
        {"applied_version": 1, "ready": False}, _loads_payload(0, 0)
    )
    [info] = asyncio.run(registry._poll_once())
    assert info.ready is False, "not-ready replica stays a registry member"

    registry.client = _registry_client(
        RuntimeError("server_info down"), _loads_payload(0, 0)
    )
    assert asyncio.run(registry._poll_once()) == []


class _FakeUpstreamResponse:
    def __init__(self, status: int, body: bytes = b"ok") -> None:
        self.status_code = status
        self.headers = {"content-type": "text/plain"}
        self._body = body

    async def aiter_raw(self):
        yield self._body


class _FakeUpstreamStream:
    def __init__(self, response: _FakeUpstreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeUpstreamResponse:
        return self._response

    async def __aexit__(self, *args) -> None:
        return None


class _FakeUpstreamClient:
    def __init__(self, status_by_upstream: dict[str, int]) -> None:
        self.status_by_upstream = status_by_upstream
        self.requested_upstreams: list[str | None] = []

    def stream(self, *, headers, **_kwargs) -> _FakeUpstreamStream:
        upstream = headers.get("modal-flash-upstream")
        self.requested_upstreams.append(upstream)
        status = self.status_by_upstream.get(upstream, 200)
        return _FakeUpstreamStream(_FakeUpstreamResponse(status))


def _proxy(fleet: dict[str, ContainerInfo]) -> tuple[_ProxyApp, Any]:
    proxy = _ProxyApp(
        registry_url="https://registry",
        upstream_url="https://upstream",
        session_routes=FakeRoutes(),
        overload_threshold=4,
    )
    app = proxy.build_app()
    proxy.containers = fleet
    return proxy, app


async def _post(app: Any, session_id: str, exact_version: int) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        return await client.post(
            "/v1/chat/completions",
            headers={
                "modal-session-id": session_id,
                "stitch-exact-version": str(exact_version),
            },
            content=b"{}",
        )


def test_forward_409_passthrough_and_503_eviction() -> None:
    """A 409 is passed through without eviction; only 503 evicts; a pin with no
    matching version is a 409 that never touches an upstream."""

    async def run() -> None:
        fleet = {
            "ta-s1": _leased("ta-s1", 0, 2, {"2": 4}),
            "ta-s2": _leased("ta-s2", 0, 2, {"2": 2}),
        }
        proxy, app = _proxy(fleet)
        fake = _FakeUpstreamClient({"ta-s1:8000": 409, "ta-s2:8000": 409})
        proxy.client = fake
        response = await _post(app, "s1", 2)
        assert response.status_code == 409, "409 from a pinned replica is passed through"
        assert len(fake.requested_upstreams) == 1
        assert fake.requested_upstreams[0] in {"ta-s1:8000", "ta-s2:8000"}
        assert set(proxy.containers) == {"ta-s1", "ta-s2"}, "409 does not evict"

        proxy, app = _proxy({"ta-s1": _leased("ta-s1", 0, 2, {"2": 4})})
        fake = _FakeUpstreamClient({})
        proxy.client = fake
        response = await _post(app, "s1", 1)
        assert response.status_code == 409, (
            "no v1 replica: 409, never a wrong-version upstream"
        )
        assert fake.requested_upstreams == []

        fleet = {
            "ta-s1": _leased("ta-s1", 0, 2, {"2": 4}),
            "ta-s2": _leased("ta-s2", 1, 2, {"2": 2}),
        }
        proxy, app = _proxy(fleet)
        fake = _FakeUpstreamClient({"ta-s1:8000": 503, "ta-s2:8000": 200})
        proxy.client = fake
        response = await _post(app, "s1", 2)
        assert response.status_code == 200
        assert fake.requested_upstreams == ["ta-s1:8000", "ta-s2:8000"], (
            "ta-s1 is the lowest-load v2 match, so the 503-retry lands on ta-s2"
        )
        assert "ta-s1" not in proxy.containers
        assert set(proxy.containers) == {"ta-s2"}

    asyncio.run(run())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"router harness: {len(tests)} PASS")
