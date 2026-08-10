"""Router harness: the pure routing helpers — ``route_session`` stickiness / overload
fallback / TTL eviction, ``select_underloaded_container``, ``filter_headers``, and
``_container_addr`` — against a fake session-routes dict (no Modal involved)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from cookbook.common.router import (
    SESSION_ROUTE_TTL_SECONDS,
    ContainerInfo,
    RouteEntry,
    RouteEntryList,
    _container_addr,
    _ProxyApp,
    filter_headers,
    route_session,
    select_underloaded_container,
)


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


def test_filter_headers_drops_routing_and_hop_by_hop() -> None:
    headers = {
        "Host": "example",
        "Content-Length": "10",
        "x-session-affinity": "s1",
        "modal-flash-upstream": "h:8000",
        "X-Forwarded-For": "1.2.3.4",
        "Authorization": "Bearer tok",
        "content-type": "application/json",
    }
    assert filter_headers(headers) == {
        "Authorization": "Bearer tok",
        "content-type": "application/json",
    }


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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"router harness: {len(tests)} PASS")
