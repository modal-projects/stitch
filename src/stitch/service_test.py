"""``sync_in_progress`` — the shared /server_info interpretation a deployment's engine-health
probe uses to suppress health blips while the reconciler commits staged weights."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
import pytest

from stitch.engines.base import Engine
from stitch.service import create_app, sync_in_progress
from stitch.sync import AdmissionGate
from stitch.types import VersionRef


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


class _RestoreGate(AdmissionGate):
    def __init__(self) -> None:
        super().__init__()
        self.applied = VersionRef("run", 10)
        self.restores: list[tuple[VersionRef, str]] = []

    async def restore(self, target: VersionRef, checkpoint_dir: str) -> None:
        self.restores.append((target, checkpoint_dir))
        self.applied = target

    def server_info(self) -> dict[str, Any]:
        return {"applied": self.applied.identity}


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


def test_upstream_transport_failure_is_retryable_and_releases_admission(
    monkeypatch, caplog
):
    async def go():
        upstream = _FailingUpstream()
        gate = AdmissionGate()
        gate.applied = VersionRef("run", 3)
        monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)
        app = create_app(gate, _ProxyEngine())  # type: ignore[arg-type]

        request = asyncio.create_task(_asgi_post(app, {"rid": "rollout-1"}))
        await upstream.started.wait()
        assert gate.active_requests == 1
        upstream.fail.set()
        status, headers, body = await request

        assert gate.active_requests == 0
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


def test_restore_endpoint_forwards_checkpoint_and_target() -> None:
    async def go():
        gate = _RestoreGate()
        app = create_app(gate, _ProxyEngine())  # type: ignore[arg-type]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://sidecar",
        ) as client:
            response = await client.post(
                "/restore",
                json={
                    "target": "run/weight_v000005",
                    "checkpoint_dir": "/saved/weight_v000004",
                },
            )
        return gate, response

    gate, response = asyncio.run(go())
    assert response.status_code == 200
    assert response.json()["applied"] == "run/weight_v000005"
    assert gate.restores == [(VersionRef("run", 5), "/saved/weight_v000004")]


def test_client_disconnect_cancels_aborts_and_releases_admission(monkeypatch):
    async def go():
        upstream = _HangingUpstream()
        allow_disconnect = asyncio.Event()
        gate = AdmissionGate()
        gate.applied = VersionRef("run", 3)
        monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: upstream)
        app = create_app(gate, _ProxyEngine())  # type: ignore[arg-type]

        request = asyncio.create_task(
            _asgi_post(app, {"rid": "rollout-2"}, disconnect_on=allow_disconnect)
        )
        await upstream.started.wait()
        assert gate.active_requests == 1
        allow_disconnect.set()
        status, _headers, _body = await request

        assert upstream.cancelled.is_set()
        assert upstream.abort_rids == ["rollout-2"]
        assert gate.active_requests == 0
        assert status == 499

    asyncio.run(go())


def test_metrics_bypasses_weight_admission_before_first_pointer(monkeypatch):
    async def go():
        upstream = _MetricsUpstream()
        gate = AdmissionGate()
        app = create_app(gate, _ProxyEngine())  # type: ignore[arg-type]
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

            assert gate.active_requests == 0
            await asyncio.wait_for(
                gate.commit(apply=apply, on_applied=lambda: None, drain_all=True),
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
        assert gate.active_requests == 0

    asyncio.run(go())


@contextlib.contextmanager
def _server_info(payload):
    """Serve ``payload`` as JSON at /server_info on a throwaway localhost port."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *_):  # silence
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/server_info"
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "info,expected",
    [
        ({"update_destination_ready": True, "sync_state": "COMMITTING"}, True),
        ({"update_destination_ready": True, "sync_state": "STAGING"}, True),
        ({"update_destination_ready": True, "sync_state": "FETCHING"}, False),
        (
            {
                "update_destination_ready": False,
                "update_destination_error": None,
            },
            True,
        ),
        ({"update_destination_ready": True, "sync_state": "IDLE"}, False),
        (
            {
                "update_destination_ready": False,
                "update_destination_error": "boom",
            },
            False,
        ),
    ],
)
def test_sync_in_progress(info, expected):
    with _server_info(info) as url:
        assert sync_in_progress(url) is expected


def test_unreachable_sidecar_reports_error():
    # Nothing listening: best-effort False so the caller surfaces the engine error.
    assert sync_in_progress("http://127.0.0.1:1/server_info", timeout=0.2) is False
