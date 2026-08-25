"""Versioned sidecar proxy behavior."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import stitch.service as stitch_service
from stitch.engines.base import Engine
from stitch.service import create_app
from stitch.sync import AdmissionGate
from stitch.types import PoolState, ReplicaState, VersionRef


class _ProxyEngine(Engine):
    def base_url(self) -> str:
        return "http://local-engine:8001"

    def blocked_routes(self) -> frozenset[str]:
        return frozenset()

    def stamp_request(self, request: dict[str, Any], served: VersionRef) -> None:
        request["served_version"] = served.version

    def stamp_response(
        self, response: dict[str, Any], served: VersionRef, current: VersionRef
    ) -> None:
        response["served_version"] = served.version


class _GateSidecar:
    """The status surface create_app consumes beside the gate; these tests exercise
    only the admission slice, so stand in the rest (no store, engine, or reconcile
    loop). The stubs are type-level only — lifespan never runs under ASGITransport."""

    def __init__(self, applied: VersionRef | None = None) -> None:
        self.applied = applied
        self.ready = True
        self.gate = AdmissionGate(served_version=lambda: self.applied)

    def readiness_reason(self) -> str:
        return ""

    def server_info(self) -> dict[str, Any]:
        return {"ready": self.ready}

    def wake(self) -> None:
        pass

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class _FailingUpstream:
    """Let a test observe admission, then fail the engine request at a controlled point."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.fail = asyncio.Event()
        self.abort_rids: list[str] = []

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/abort_request"):
            self.abort_rids.append(kwargs["json"]["rid"])
            return httpx.Response(200)
        self.started.set()
        await self.fail.wait()
        raise httpx.ConnectError(
            "all connection attempts failed", request=httpx.Request(method, url)
        )


class _HangingUpstream:
    """An engine request which runs until the proxy cancels it on client disconnect."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.abort_rids: list[str] = []

    async def request(self, _method: str, url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/abort_request"):
            self.abort_rids.append(kwargs["json"]["rid"])
            return httpx.Response(200)
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _MetricsUpstream:
    """Return a minimal Prometheus payload and record whether it was requested."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.started = asyncio.Event()
        self.finish = asyncio.Event()

    async def request(self, method: str, url: str, **_kwargs: Any) -> httpx.Response:
        self.requests.append((method, url))
        self.started.set()
        await self.finish.wait()
        return httpx.Response(
            200,
            content=b"# TYPE sglang:num_running_reqs gauge\nsglang:num_running_reqs 0\n",
            headers={"content-type": "text/plain; version=0.0.4; charset=utf-8"},
        )


class _JsonUpstream:
    """A healthy engine: answer every request 200 with a small JSON body."""

    async def request(self, _method: str, _url: str, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})


async def _asgi_post(
    app: Any,
    payload: dict[str, Any],
    *,
    disconnect_on: asyncio.Event | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
):
    """Issue one request directly to the ASGI app, optionally disconnecting after its body."""
    body = json.dumps(payload).encode()
    body_sent = False
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        if disconnect_on is not None:
            await disconnect_on.wait()
            return {"type": "http.disconnect"}
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/generate",
            "raw_path": b"/generate",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json"), *(extra_headers or [])],
            "client": ("127.0.0.1", 1234),
            "server": ("sidecar", 8000),
        },
        receive,
        send,
    )
    start = next(m for m in sent if m["type"] == "http.response.start")
    response_body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    return start["status"], headers, response_body


def test_upstream_transport_failure_is_retryable_and_releases_admission(
    monkeypatch, caplog
):
    async def go():
        upstream = _FailingUpstream()
        gate_sidecar = _GateSidecar(VersionRef("run", 3))
        monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)
        app = create_app(gate_sidecar.gate, gate_sidecar, _ProxyEngine())

        request = asyncio.create_task(_asgi_post(app, {"rid": "rollout-1"}))
        await upstream.started.wait()
        assert gate_sidecar.gate.active_requests == 1
        upstream.fail.set()
        status, headers, body = await request

        assert gate_sidecar.gate.active_requests == 0
        assert upstream.abort_rids == ["rollout-1"]
        return status, headers, json.loads(body)

    with caplog.at_level(logging.WARNING, logger="stitch.service"):
        status, headers, data = asyncio.run(go())

    assert status == 503
    assert headers["retry-after"] == "1"
    assert data == {
        "error": {
            "type": "EngineUnavailable",
            "message": "the local inference engine is unavailable",
            "retryable": True,
        }
    }
    records = [r for r in caplog.records if "local engine request failed" in r.message]
    assert len(records) == 1
    assert (
        records[0].exc_info is None
    )  # concise per-request signal, not a traceback storm


def test_client_disconnect_cancels_aborts_and_releases_admission(monkeypatch):
    async def go():
        upstream = _HangingUpstream()
        allow_disconnect = asyncio.Event()
        gate_sidecar = _GateSidecar(VersionRef("run", 3))
        monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)
        app = create_app(gate_sidecar.gate, gate_sidecar, _ProxyEngine())

        request = asyncio.create_task(
            _asgi_post(app, {"rid": "rollout-2"}, disconnect_on=allow_disconnect)
        )
        await upstream.started.wait()
        assert gate_sidecar.gate.active_requests == 1
        allow_disconnect.set()
        status, _headers, _body = await request

        assert upstream.cancelled.is_set()
        assert upstream.abort_rids == ["rollout-2"]
        assert gate_sidecar.gate.active_requests == 0
        assert status == 499

    asyncio.run(go())


def test_metrics_bypasses_weight_admission_before_first_pointer(monkeypatch):
    async def go():
        upstream = _MetricsUpstream()
        gate_sidecar = _GateSidecar()
        app = create_app(gate_sidecar.gate, gate_sidecar, _ProxyEngine())
        sidecar = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://sidecar"
        )
        monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)

        async with sidecar:
            blocked = await sidecar.get("/v1/models")
            request = asyncio.create_task(sidecar.get("/metrics"))
            await upstream.started.wait()

            async def apply() -> None:
                pass

            assert gate_sidecar.gate.active_requests == 0
            await asyncio.wait_for(
                gate_sidecar.gate.commit(
                    apply=apply, on_applied=lambda: None, drain_all=True
                ),
                timeout=1.0,
            )
            assert not request.done()

            upstream.finish.set()
            response = await request

        assert blocked.status_code == 409
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
        assert response.content.startswith(b"# TYPE sglang:num_running_reqs gauge")
        assert upstream.requests == [("GET", "http://local-engine:8001/metrics")]
        assert gate_sidecar.gate.active_requests == 0

    asyncio.run(go())


def test_proxy_lease_headers(monkeypatch):
    """Lease-header variants: default header keys the lease; headerless requests
    add none; a configured lease_header takes over the keying."""

    async def go():
        gate = AdmissionGate(
            served_version=lambda: VersionRef("run", 3), version_lease_ttl=60.0
        )
        gate_sidecar = _GateSidecar(VersionRef("run", 3))
        monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: _JsonUpstream())
        app = create_app(gate, gate_sidecar, _ProxyEngine())
        pin = {"weight_version": {"exact_version": 3}}

        status, _, _ = await _asgi_post(
            app, pin, extra_headers=[(b"modal-session-id", b"sess-1")]
        )
        assert status == 200
        assert gate.leases_snapshot() == {3: 1}, (
            "default Modal-Session-Id header keys the lease"
        )

        status, _, _ = await _asgi_post(app, pin)
        assert status == 200
        assert gate.leases_snapshot() == {3: 1}, (
            "headerless request is served on the body pin and adds no lease"
        )

        gate = AdmissionGate(
            served_version=lambda: VersionRef("run", 3), version_lease_ttl=60.0
        )
        gate_sidecar = _GateSidecar(VersionRef("run", 3))
        app = create_app(gate, gate_sidecar, _ProxyEngine(), lease_header="X-Session")

        status, _, _ = await _asgi_post(
            app, pin, extra_headers=[(b"x-session", b"sess-9")]
        )
        assert status == 200
        assert gate.leases_snapshot() == {3: 1}, (
            "configured lease_header keys the lease"
        )

        status, _, _ = await _asgi_post(
            app, pin, extra_headers=[(b"modal-session-id", b"sess-10")]
        )
        assert status == 200
        assert gate.leases_snapshot() == {3: 1}, (
            "default header no longer keys leases once lease_header is set"
        )

    asyncio.run(go())


def test_await_pool_ready_waits_for_replica_threshold(monkeypatch) -> None:
    states = iter(
        [
            PoolState(
                [
                    ReplicaState(ready=True),
                    ReplicaState(ready=True),
                    ReplicaState(),
                    ReplicaState(),
                ]
            ),
            PoolState(
                [
                    ReplicaState(ready=True),
                    ReplicaState(ready=True),
                    ReplicaState(ready=True),
                    ReplicaState(),
                ]
            ),
        ]
    )

    async def pool_readiness(_pool):
        return next(states)

    async def no_sleep(_seconds):
        pass

    class Pool:
        async def gateway_url_async(self):
            return "http://gateway"

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def get(self, _url, **_kwargs):
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(stitch_service, "readiness", pool_readiness)
    monkeypatch.setattr(stitch_service.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: Client())

    assert stitch_service.await_pool_ready(Pool(), replica_floor=4, interval=0)


def test_await_pool_ready_fails_closed_below_threshold(monkeypatch) -> None:
    async def pool_readiness(_pool):
        return PoolState([ReplicaState(ready=True), ReplicaState()])

    async def no_sleep(_seconds):
        pass

    times = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(stitch_service, "readiness", pool_readiness)
    monkeypatch.setattr(stitch_service.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(
        stitch_service,
        "time",
        SimpleNamespace(monotonic=lambda: next(times)),
    )

    with pytest.raises(
        TimeoutError,
        match=r"1/2 required \(2 discovered\)",
    ):
        stitch_service.await_pool_ready(
            object(), replica_floor=2, timeout=1, interval=0
        )
