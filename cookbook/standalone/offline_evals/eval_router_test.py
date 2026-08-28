"""Eval router: three end-to-end walks against in-memory fakes — version-aware
session routing, the /server_info-membership registry poll, and the ASGI forward
path (pin cells, 409/503 semantics, eviction, header strip)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from cookbook.standalone.offline_evals import eval_router
from cookbook.standalone.offline_evals.eval_router import (
    EvalContainerInfo,
    EvalProxyApp,
    EvalRegistryApp,
    EvalRouteEntry,
    EvalRouteEntryList,
    route_session,
    select_packed_container,
)
from cookbook.standalone.offline_evals.testing import _LocalDict


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


def _seeded(routes: FakeRoutes, session: str, entries: list[dict]) -> None:
    routes.store[session] = EvalRouteEntryList.dump_python(
        [
            EvalRouteEntry(
                **{
                    "version": None,
                    "tombstoned": False,
                    **entry,
                    "last_seen": entry.get("last_seen", entry["last_sent"]),
                }
            )
            for entry in entries
        ],
        mode="json",
    )


def _versioned(
    task_id: str,
    load: int,
    applied_version: int | None,
    *,
    ready: bool = True,
    draining: bool = False,
) -> EvalContainerInfo:
    return EvalContainerInfo(
        task_id=task_id,
        upstream=f"{task_id}:8000",
        load=load,
        applied_version=applied_version,
        ready=ready,
        draining=draining,
    )


def test_route_session_walk() -> None:
    """One walk of version-aware session routing: preference, sticky skips and
    evictions, no-match no-repin, saturation queuing, and the concurrent-pop race."""

    def pick(
        fleet: dict, pin: int | None, routes: FakeRoutes | None = None, session="s1"
    ) -> EvalContainerInfo | None:
        routes = routes if routes is not None else FakeRoutes()
        return asyncio.run(route_session(routes, session, fleet, 4, exact_version=pin))

    containers = {
        "ta-0": _versioned("ta-0", 0, 1),
        "ta-1": _versioned("ta-1", 0, 2),
    }
    routes = FakeRoutes()
    picked = pick(containers, 2, routes)
    assert picked.task_id == "ta-1", "version preference picks the matching replica"
    assert picked.applied_version == 2

    picked = pick(containers, 2, routes)
    assert picked.task_id == "ta-1", "sticky route holds across requests"
    routes2 = FakeRoutes()
    _seeded(routes2, "s2", [{"task_id": "ta-0", "last_sent": time.time()}])
    picked = pick(containers, 2, routes2, session="s2")
    assert picked.task_id == "ta-1", "sticky wrong-version route is skipped"

    routes = FakeRoutes()
    picked = pick({"ta-0": containers["ta-0"]}, 99, routes)
    assert picked is None, "no version match returns None"
    assert "s1" not in routes.store, "no match must not re-pin"

    picked = pick(
        {"ta-0": _versioned("ta-0", 0, 2), "ta-1": _versioned("ta-1", 0, None)}, 1
    )
    assert picked is None, "an unknown (None) applied_version never matches a pin"

    picked = pick({"ta-0": _versioned("ta-0", 0, 1, draining=True)}, 1)
    assert picked is None, "a draining replica never matches, even version-aligned"

    routes = FakeRoutes()
    picked = pick(
        {
            "ta-0": _versioned("ta-0", 9, 1),
            "ta-1": _versioned("ta-1", 12, 1),
            "ta-2": _versioned("ta-2", 0, 2),
        },
        1,
        routes,
    )
    assert picked.task_id == "ta-0", "saturated matching replica queues, never 409s"
    assert routes.store["s1"][0]["task_id"] == "ta-0"

    routes = FakeRoutes()
    _seeded(routes, "s3", [{"task_id": "ta-0", "last_sent": time.time()}])
    fleet = {
        "ta-0": _versioned("ta-0", 0, 1, draining=True),
        "ta-1": _versioned("ta-1", 0, 1),
    }
    picked = pick(fleet, None, routes, session="s3")
    assert picked.task_id == "ta-1"
    assert [entry["task_id"] for entry in routes.store["s3"]] == ["ta-1"], (
        "a sticky route to a draining replica is deleted, not kept"
    )

    async def race() -> None:
        # A container-map pop racing a selection suspended in save_routes degrades
        # to another replica or None, never a KeyError.
        routes = FakeRoutes()

        class _YieldingPut:
            async def aio(self, key: str, value: list) -> None:
                await asyncio.sleep(0)  # let the popper interleave mid-save
                routes.store[key] = value

        routes.put = _YieldingPut()  # type: ignore[assignment]
        fleet = {f"ta-{i}": _versioned(f"ta-{i}", 0, 1) for i in range(3)}
        _seeded(routes, "s", [{"task_id": "ta-1", "last_sent": time.time()}])
        picks: list[str] = []

        async def popper() -> None:
            for _ in range(400):
                popped = fleet.pop("ta-1", None)
                await asyncio.sleep(0)
                if popped is not None:
                    fleet["ta-1"] = popped
                await asyncio.sleep(0)

        async def select_loop() -> None:
            for _ in range(200):
                picked = await route_session(routes, "s", fleet, 4)
                if picked is not None:
                    picks.append(picked.task_id)

        await asyncio.gather(popper(), *(select_loop() for _ in range(8)))
        assert picks, "selections kept succeeding around concurrent pops"
        fleet.pop("ta-1", None)
        final = await route_session(routes, "s", fleet, 4)
        assert final is not None and final.task_id != "ta-1"

    asyncio.run(race())


class _FakeResp:
    def __init__(self, payload: dict, *, ok: bool = True) -> None:
        self._payload = payload
        self._ok = ok

    def raise_for_status(self) -> None:
        if not self._ok:
            raise RuntimeError("HTTP 503")

    def json(self) -> dict:
        return self._payload


def _loads_payload(waiting: int, running: int) -> dict:
    return {"loads": [{"num_waiting_reqs": waiting, "num_running_reqs": running}]}


def _registry_client(server_info: dict | Exception, loads: dict | Exception) -> object:
    class FakeClient:
        def __init__(self) -> None:
            self.seen_upstreams: list[str | None] = []

        async def get(self, url: str, **kwargs) -> _FakeResp:
            self.seen_upstreams.append(
                (kwargs.get("headers") or {}).get("modal-flash-upstream")
            )
            payload = (
                server_info
                if url.endswith("/server_info")
                else loads
                if url.endswith("/v1/loads")
                else None
            )
            if payload is None:
                raise AssertionError(f"unexpected url: {url}")
            if isinstance(payload, Exception):
                raise payload
            return _FakeResp(payload)

    return FakeClient()


def test_registry_poll_walk(monkeypatch) -> None:
    """One walk of the registry poll: load/version passthrough through the pool
    URL, unparseable and absent applied identities, server_info failure dropping
    the replica, and a loads failure keeping it on a stale load."""

    def registry(client: object) -> EvalRegistryApp:
        app = EvalRegistryApp(
            app_name="app",
            upstream_cls="Server",
            upstream_url="https://pool",
            session_routes=_LocalDict(),
            control=_LocalDict(),
            store=None,
        )
        app.client = client
        return app

    client = _registry_client(
        {"applied": "run-a/weight_v000003", "ready": True}, _loads_payload(1, 2)
    )
    poll = asyncio.run(registry(client)._poll_container("h0:8000"))
    assert poll.load == 3
    assert poll.applied_version == 3, "applied identity parses via VersionRef"
    assert poll.ready is True
    assert client.seen_upstreams == ["h0:8000", "h0:8000"], (
        "both polls traverse the pool URL with the modal-flash-upstream pin"
    )

    poll = asyncio.run(
        registry(
            _registry_client({"applied": None, "ready": False}, _loads_payload(0, 0))
        )._poll_container("h0:8000")
    )
    assert poll.applied_version is None and poll.ready is False

    poll = asyncio.run(
        registry(
            _registry_client({"applied": "garbage"}, _loads_payload(0, 0))
        )._poll_container("h0:8000")
    )
    assert poll.applied_version is None, "unparseable identity reads as unknown"

    try:
        asyncio.run(
            registry(
                _registry_client(RuntimeError("server_info down"), _loads_payload(0, 0))
            )._poll_container("h0:8000")
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("server_info failure must propagate (drops the replica)")

    async def discover(_app_name: str, _cls: str) -> list[dict]:
        return [{"task_id": "ta-0", "host": "h-ta-0", "port": 8000}]

    monkeypatch.setattr(eval_router, "list_flash_containers_async", discover)
    down = registry(
        _registry_client(RuntimeError("server_info down"), _loads_payload(0, 0))
    )
    assert asyncio.run(down._poll_once()) == []

    app = registry(
        _registry_client({"applied": "weight_v000001"}, _loads_payload(2, 1))
    )
    (first,) = asyncio.run(app._poll_once())
    assert first.load == 3 and not first.load_stale
    app.client = _registry_client(
        {"applied": "weight_v000001"}, RuntimeError("loads down")
    )
    (second,) = asyncio.run(app._poll_once())
    assert second.load == 3 and second.load_stale, (
        "a failed loads poll keeps the replica on its last-known load"
    )


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
        self.requested_headers: list[dict[str, str]] = []

    def stream(self, *, headers, **_kwargs) -> _FakeUpstreamStream:
        upstream = headers.get("modal-flash-upstream")
        self.requested_upstreams.append(upstream)
        self.requested_headers.append(dict(headers))
        status = self.status_by_upstream.get(upstream, 200)
        return _FakeUpstreamStream(_FakeUpstreamResponse(status))


def _proxy(
    fleet: dict[str, EvalContainerInfo], statuses: dict[str, int] | None = None
) -> tuple[EvalProxyApp, Any, _FakeUpstreamClient]:
    proxy = EvalProxyApp(
        registry_url="https://registry",
        upstream_url="https://upstream",
        session_routes=FakeRoutes(),
        overload_threshold=4,
    )
    app = proxy.build_app()
    proxy.containers = fleet
    proxy.client = _FakeUpstreamClient(statuses or {})
    return proxy, app, proxy.client


async def _post(
    app: Any,
    session_id: str | None,
    exact_version: int | None | str = None,
    extra_headers: dict[str, str] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    headers = {"modal-session-id": session_id} if session_id is not None else {}
    if exact_version is not None:
        headers["stitch-exact-version"] = str(exact_version)
    headers.update(extra_headers or {})
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/chat/completions", headers=headers, content=b"{}")


def test_forward_walk() -> None:
    """One walk of the ASGI forward path: every pin cell (pinned with/without a
    session, cold registry, no match), 409 passthrough without eviction, 503
    eviction with retry, the unpinned 503, malformed pins, and the Modal-* strip."""

    async def run() -> None:
        fleet = {
            "ta-s1": _versioned("ta-s1", 0, 2),
            "ta-s2": _versioned("ta-s2", 0, 2),
        }
        proxy, app, fake = _proxy(fleet, {"ta-s1:8000": 409, "ta-s2:8000": 409})
        response = await _post(
            app, "s1", 2, extra_headers={"Modal-Anything": "forged", "x-keep": "v"}
        )
        assert response.status_code == 409, "409 from a pinned replica passes through"
        assert len(fake.requested_upstreams) == 1
        assert set(proxy.containers) == {"ta-s1", "ta-s2"}, "409 does not evict"
        (sent,) = fake.requested_headers
        assert sent.get("x-keep") == "v"
        forged = [
            k
            for k in sent
            if k.lower().startswith("modal-") and k.lower() != "modal-flash-upstream"
        ]
        assert forged == [], "only the router's own pin survives in Modal-*"

        proxy, app, fake = _proxy({"ta-s1": _versioned("ta-s1", 0, 2)})
        response = await _post(app, "s1", 1)
        assert response.status_code == 409, "no v1 replica: 409, never wrong-version"
        assert fake.requested_upstreams == []

        proxy, app, fake = _proxy({"ta-boot": _versioned("ta-boot", 0, 2, ready=False)})
        response = await _post(app, "s1", None)
        assert response.status_code == 503, "unpinned, nothing ready: retryable 503"
        assert response.headers["retry-after"] == "1"
        assert fake.requested_upstreams == []

        fleet = {
            "ta-s1": _versioned("ta-s1", 0, 2),
            "ta-s2": _versioned("ta-s2", 1, 2),
        }
        proxy, app, fake = _proxy(fleet, {"ta-s1:8000": 503, "ta-s2:8000": 200})
        response = await _post(app, "s1", 2)
        assert response.status_code == 200
        assert fake.requested_upstreams == ["ta-s1:8000", "ta-s2:8000"], (
            "ta-s1 is the lowest-load v2 match, so the 503-retry lands on ta-s2"
        )
        assert set(proxy.containers) == {"ta-s2"}, "503 evicts"

        proxy, app, fake = _proxy({})
        for session_id in (None, "s1"):
            response = await _post(app, session_id, 2)
            assert response.status_code == 409, "cold registry: pinned is 409"
            assert response.text == "No healthy upstream"
        assert fake.requested_upstreams == []

        fleet = {
            "ta-v1": _versioned("ta-v1", 0, 1),
            "ta-v2": _versioned("ta-v2", 0, 2),
        }
        proxy, app, fake = _proxy(fleet)
        response = await _post(app, None, 2)
        assert response.status_code == 200
        assert fake.requested_upstreams == ["ta-v2:8000"], (
            "pinned and sessionless: version-aware placement, no fallthrough"
        )
        assert proxy.session_routes.store == {}, "no sticky lookup, no record"

        proxy, app, fake = _proxy({"ta-s1": _versioned("ta-s1", 0, 2)})
        response = await _post(app, "s1", "abc")
        assert response.status_code == 200
        assert fake.requested_upstreams == ["ta-s1:8000"], (
            "a malformed pin is dropped; placement proceeds unpinned"
        )

    asyncio.run(run())


def test_end_session_tombstones_idempotently() -> None:
    """POST /sessions/{id}/end: 200 always, marks the record tombstoned, and a
    tombstoned session re-pins clean on its next request."""

    async def run() -> None:
        proxy, app, _ = _proxy({"ta-s1": _versioned("ta-s1", 0, 2)})
        assert (await _post(app, "s1", 2)).status_code == 200
        assert proxy.session_routes.store["s1"][0]["task_id"] == "ta-s1"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for session in ("s1", "s1", "never-seen"):
                end = await client.post(f"/sessions/{session}/end")
                assert end.status_code == 200, "idempotent: 200 always"
        record = proxy.session_routes.store["s1"]
        assert all(entry["tombstoned"] for entry in record)

        assert (await _post(app, "s1", 2)).status_code == 200
        record = proxy.session_routes.store["s1"]
        assert not record[0]["tombstoned"], "a reused session id starts clean"

    asyncio.run(run())


def test_select_packed_container_packs_live_holders() -> None:
    candidates = [
        _versioned("ta-0", 0, 1),
        _versioned("ta-1", 3, 1),
        _versioned("ta-2", 5, 1),
    ]
    candidates[1].live_sessions = 2
    candidates[2].live_sessions = 1
    picked = select_packed_container(candidates)
    assert picked.task_id == "ta-1", "lowest load among live-session holders"
    for candidate in candidates:
        candidate.live_sessions = 0
    assert select_packed_container(candidates).task_id == "ta-0", (
        "no holders: lowest load overall"
    )
    candidates[0].load_stale = True
    assert select_packed_container(candidates).task_id == "ta-1", (
        "a stale (frozen) load reading never wins the pack"
    )
