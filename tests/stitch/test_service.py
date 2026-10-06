"""Versioned sidecar proxy behavior."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
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


async def _asgi_post(
    app: Any, payload: dict[str, Any], *, disconnect_on: asyncio.Event | None = None
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
            "headers": [(b"content-type", b"application/json")],
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


@pytest.mark.parametrize("limit", [None, 128], ids=["default-100", "configured-128"])
def test_proxy_connection_limits(monkeypatch, limit):
    async def go():
        arrived = {n: asyncio.Event() for n in (100, 128)}
        release = asyncio.Event()
        count = 0

        async def handle(reader, writer):
            nonlocal count
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                for line in headers.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        await reader.readexactly(int(line.split(b":", 1)[1]))
                count += 1
                if count in arrived:
                    arrived[count].set()
                # Hold all responses so the test exercises real HTTPX pool capacity.
                await release.wait()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 2\r\nConnection: close\r\n\r\n{}"
                )
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0, backlog=128)
        port = server.sockets[0].getsockname()[1]
        engine = _ProxyEngine()
        monkeypatch.setattr(engine, "base_url", lambda: f"http://127.0.0.1:{port}")
        sidecar = _GateSidecar(VersionRef("run", 3))
        options = {} if limit is None else {"proxy_max_connections": limit}
        app = create_app(sidecar.gate, sidecar, engine, **options)
        async with server, app.router.lifespan_context(app):
            requests = [asyncio.create_task(_asgi_post(app, {})) for _ in range(128)]
            try:
                await asyncio.wait_for(arrived[100].wait(), timeout=10)
                if limit is None:
                    # The default pool cannot forward the remaining 28 requests yet.
                    with pytest.raises(TimeoutError):
                        await asyncio.wait_for(arrived[128].wait(), timeout=1)
                    assert count == 100
                else:
                    # Ignoring the configured limit must fail here, not be suppressed.
                    await asyncio.wait_for(arrived[128].wait(), timeout=10)
                    assert count == 128
            finally:
                release.set()
                responses = await asyncio.wait_for(
                    asyncio.gather(*requests), timeout=10
                )
            assert count == 128
            assert all(status == 200 for status, _, _ in responses)

    asyncio.run(go())


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


def test_json_response_rendering_runs_off_event_loop(monkeypatch):
    async def go():
        class Upstream:
            async def request(
                self, _method: str, _url: str, **_kwargs: Any
            ) -> httpx.Response:
                return httpx.Response(
                    200,
                    content=b'{"choices":[],"payload":"large-response"}',
                    headers={"content-type": "application/json"},
                )

        monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: Upstream())
        sidecar = _GateSidecar(VersionRef("run", 3))
        app = create_app(sidecar.gate, sidecar, _ProxyEngine())

        event_loop_thread = threading.get_ident()
        render_threads = []
        original = stitch_service._render_json_response

        def record_render_thread(*args):
            render_threads.append(threading.get_ident())
            return original(*args)

        monkeypatch.setattr(
            stitch_service, "_render_json_response", record_render_thread
        )
        status, headers, body = await _asgi_post(app, {})

        assert status == 200
        assert headers["content-type"] == "application/json"
        assert render_threads and render_threads[0] != event_loop_thread
        assert json.loads(body) == {
            "choices": [],
            "payload": "large-response",
            "served_version": 3,
        }

    asyncio.run(go())


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


class _SlowUpstream:
    """An engine request that answers (or fails) when the test says, and records an
    abort or a cancellation."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.abort_rids: list[str] = []

    async def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/abort_request"):
            self.abort_rids.append(kwargs["json"]["rid"])
            return httpx.Response(200)
        self.started.set()
        try:
            await self.finish.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.fail:
            raise httpx.ReadError(
                "engine connection reset", request=httpx.Request(method, url)
            )
        return httpx.Response(
            200,
            content=b'{"choices":[{"text":"done"}]}',
            headers={"content-type": "application/json"},
        )


class _Exchange:
    """One ASGI request, observable while its response streams. A client on ASGI spec
    2.4 learns of a disconnect only when a send fails; an older one is told via
    ``receive``."""

    def __init__(
        self,
        app: Any,
        payload: dict[str, Any] | None,
        *,
        method: str = "POST",
        path: str = "/generate",
        spec_version: str = "2.3",
    ) -> None:
        self.app = app
        self.body_in = b"" if payload is None else json.dumps(payload).encode()
        self.method = method
        self.path = path
        self.spec_version = spec_version
        self.started = asyncio.Event()
        self.chunk = asyncio.Event()
        self.disconnect = asyncio.Event()
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.body = b""
        self._body_sent = False

    async def receive(self) -> dict[str, Any]:
        if not self._body_sent:
            self._body_sent = True
            return {"type": "http.request", "body": self.body_in, "more_body": False}
        await self.disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(self, message: dict[str, Any]) -> None:
        if self.disconnect.is_set() and self.spec_version >= "2.4":
            raise OSError("client went away")
        if message["type"] == "http.response.start":
            self.status = message["status"]
            self.headers = {
                k.decode().lower(): v.decode() for k, v in message["headers"]
            }
            self.started.set()
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")
            self.chunk.set()

    async def run(self) -> None:
        headers = [(b"content-type", b"application/json")] if self.body_in else []
        await self.app(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": self.spec_version},
                "http_version": "1.1",
                "method": self.method,
                "scheme": "http",
                "path": self.path,
                "raw_path": self.path.encode(),
                "query_string": b"",
                "headers": headers,
                "client": ("127.0.0.1", 1234),
                "server": ("sidecar", 8000),
            },
            self.receive,
            self.send,
        )


def _keepalive_app(monkeypatch, upstream: Any, sidecar: _GateSidecar) -> Any:
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)
    return create_app(
        sidecar.gate,
        sidecar,
        _ProxyEngine(),
        response_keepalive_after=0.05,
        response_keepalive_interval=0.01,
    )


def test_long_generation_is_kept_alive_and_holds_admission(monkeypatch):
    async def go():
        upstream = _SlowUpstream()
        sidecar = _GateSidecar(VersionRef("run", 3))
        exchange = _Exchange(
            _keepalive_app(monkeypatch, upstream, sidecar), {"rid": "rollout-3"}
        )
        request = asyncio.create_task(exchange.run())

        await asyncio.wait_for(exchange.started.wait(), timeout=5)
        assert exchange.status == 200
        assert exchange.headers["content-type"] == "application/json"
        await asyncio.sleep(0.1)
        assert len(exchange.body) >= 2 and not exchange.body.strip()

        # The stream holds the admission lease, so a commit cannot advance the
        # version under the generation the response will be stamped with.
        assert sidecar.gate.active_requests == 1

        async def apply() -> None:
            pass

        def on_applied() -> None:
            sidecar.applied = VersionRef("run", 4)

        commit = asyncio.create_task(
            sidecar.gate.commit(apply=apply, on_applied=on_applied, drain_all=True)
        )
        await asyncio.sleep(0.05)
        assert not commit.done()

        upstream.finish.set()
        await asyncio.wait_for(request, timeout=5)
        await asyncio.wait_for(commit, timeout=5)

        assert sidecar.gate.active_requests == 0
        assert upstream.abort_rids == []
        assert exchange.body.startswith(b" ")
        assert json.loads(exchange.body) == {
            "choices": [{"text": "done"}],
            "served_version": 3,
        }

    asyncio.run(go())


def test_in_place_commit_crosses_a_kept_alive_generation(monkeypatch):
    """In-place commits do not drain non-exact requests: a long generation keeps
    running across the weight update, its keep-alive bytes keep flowing while the
    engine is paused, and its response spans both versions."""

    class _SpanEngine(_ProxyEngine):
        def stamp_response(self, response, served, current) -> None:
            response["weight_version_start"] = served.version
            response["weight_version_end"] = current.version

    async def go():
        upstream = _SlowUpstream()
        sidecar = _GateSidecar(VersionRef("run", 3))
        monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)
        app = create_app(
            sidecar.gate,
            sidecar,
            _SpanEngine(),
            response_keepalive_after=0.05,
            response_keepalive_interval=0.01,
        )
        exchange = _Exchange(app, {"rid": "rollout-6"})
        request = asyncio.create_task(exchange.run())
        await asyncio.wait_for(exchange.started.wait(), timeout=5)

        paused = asyncio.Event()
        finish_apply = asyncio.Event()
        calls: list[str] = []

        async def pause() -> None:
            calls.append("pause")
            paused.set()

        async def apply() -> None:
            await finish_apply.wait()

        async def resume() -> None:
            calls.append("resume")

        def on_applied() -> None:
            sidecar.applied = VersionRef("run", 4)

        commit = asyncio.create_task(
            sidecar.gate.commit(
                apply=apply, on_applied=on_applied, pause=pause, resume=resume
            )
        )
        # The commit does not wait for the kept-alive request to finish.
        await asyncio.wait_for(paused.wait(), timeout=5)
        assert sidecar.gate.active_requests == 1
        before = len(exchange.body)
        await asyncio.sleep(0.1)
        assert len(exchange.body) > before  # bytes keep flowing during the pause
        finish_apply.set()
        await asyncio.wait_for(commit, timeout=5)
        assert calls == ["pause", "resume"]
        assert not request.done()

        upstream.finish.set()
        await asyncio.wait_for(request, timeout=5)
        assert json.loads(exchange.body) == {
            "choices": [{"text": "done"}],
            "weight_version_start": 3,
            "weight_version_end": 4,
        }
        assert sidecar.gate.active_requests == 0

    asyncio.run(go())


def test_fast_response_is_unchanged_with_keepalive_enabled(monkeypatch):
    async def go():
        upstream = _SlowUpstream()
        upstream.finish.set()
        sidecar = _GateSidecar(VersionRef("run", 3))
        exchange = _Exchange(_keepalive_app(monkeypatch, upstream, sidecar), {})
        await asyncio.wait_for(exchange.run(), timeout=5)

        assert exchange.status == 200
        assert exchange.body == b'{"choices":[{"text":"done"}],"served_version":3}'
        assert exchange.headers["content-length"] == str(len(exchange.body))
        assert sidecar.gate.active_requests == 0

    asyncio.run(go())


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_disconnect_during_keepalive_aborts_and_releases_admission(
    monkeypatch, spec_version
):
    async def go():
        upstream = _SlowUpstream()
        sidecar = _GateSidecar(VersionRef("run", 3))
        exchange = _Exchange(
            _keepalive_app(monkeypatch, upstream, sidecar),
            {"rid": "rollout-4"},
            spec_version=spec_version,
        )
        request = asyncio.create_task(exchange.run())
        await asyncio.wait_for(exchange.chunk.wait(), timeout=5)
        assert sidecar.gate.active_requests == 1

        exchange.disconnect.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(request, timeout=5)

        assert upstream.cancelled.is_set()
        assert upstream.abort_rids == ["rollout-4"]
        assert sidecar.gate.active_requests == 0

    asyncio.run(go())


def test_engine_failure_during_keepalive_ends_the_body_and_releases(
    monkeypatch, caplog
):
    async def go():
        upstream = _SlowUpstream(fail=True)
        sidecar = _GateSidecar(VersionRef("run", 3))
        exchange = _Exchange(
            _keepalive_app(monkeypatch, upstream, sidecar), {"rid": "rollout-5"}
        )
        request = asyncio.create_task(exchange.run())
        await asyncio.wait_for(exchange.started.wait(), timeout=5)
        upstream.finish.set()
        await asyncio.wait_for(request, timeout=5)

        # The 200 is already on the wire; the body carries the retryable error, which
        # the session server rejects as a response without choices.
        assert exchange.status == 200
        assert json.loads(exchange.body)["error"]["type"] == "EngineUnavailable"
        assert upstream.abort_rids == ["rollout-5"]
        assert sidecar.gate.active_requests == 0

    with caplog.at_level(logging.WARNING, logger="stitch.service"):
        asyncio.run(go())
    assert any("local engine request failed" in r.message for r in caplog.records)


def test_unversioned_routes_are_never_kept_alive(monkeypatch):
    async def go():
        upstream = _MetricsUpstream()
        sidecar = _GateSidecar(VersionRef("run", 3))
        exchange = _Exchange(
            _keepalive_app(monkeypatch, upstream, sidecar),
            None,
            method="GET",
            path="/metrics",
        )
        request = asyncio.create_task(exchange.run())
        await upstream.started.wait()
        await asyncio.sleep(0.15)
        assert not exchange.started.is_set()

        upstream.finish.set()
        await asyncio.wait_for(request, timeout=5)
        assert exchange.status == 200
        assert exchange.body.startswith(b"# TYPE sglang:num_running_reqs gauge")

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


def test_await_pool_ready_excludes_replicas_ahead_of_latest(monkeypatch) -> None:
    latest = VersionRef("run", 3)
    stale = ReplicaState(ready=True, applied=VersionRef("run", 5))
    replaced = ReplicaState(ready=True, applied=latest)
    states = iter(
        [
            PoolState([stale, stale, ReplicaState()]),
            PoolState([replaced, replaced, ReplicaState()]),
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

    assert stitch_service.await_pool_ready(
        Pool(), replica_floor=2, interval=0, latest=latest
    )
    assert next(states, None) is None  # the stale-but-ready fleet did not pass


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
