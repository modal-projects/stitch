"""Exercise the session proxy's admission retries in the Miles trainer image."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("miles") is None,
    reason="requires the Miles trainer image",
)


@pytest.mark.parametrize(
    ("status", "body", "retried"),
    [
        (409, {"error": "stale weights"}, True),
        (429, {"error": "rate limited"}, True),
        (503, "Server is at capacity", True),
        (502, "Server is at capacity", False),
        (503, {"error": {"message": "The request queue is full."}}, True),
        (503, {"detail": "The request queue is full."}, True),
        (503, {"error": {"message": "backend disconnected"}}, False),
        (500, {"error": {"message": "The request queue is full."}}, False),
        (503, "upstream unavailable", False),
        (503, ["The request queue is full."], False),
    ],
)
def test_admission_retry_preserves_request_and_hook_budget(monkeypatch, status, body, retried):
    from miles.rollout.session import server as session_server

    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr(session_server.asyncio, "sleep", sleep)

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            content = body if isinstance(body, str) else json.dumps(body)
            return httpx.Response(status, text=content)
        return httpx.Response(200, json={"ok": True})

    result = asyncio.run(_proxy(session_server, respond, attempts=3))

    assert result["status_code"] == (200 if retried else status)
    assert len(requests) == (2 if retried else 1)
    if retried:
        sleep.assert_awaited_once_with(0.125)
        assert requests[0].content == requests[1].content
        assert requests[0].headers == requests[1].headers
        assert requests[0].url == requests[1].url
    else:
        sleep.assert_not_awaited()


@pytest.mark.parametrize(
    "body", ["Server is at capacity", {"error": {"message": "The request queue is full."}}]
)
def test_admission_rejection_stops_at_configured_attempt_limit(monkeypatch, body):
    from miles.rollout.session import server as session_server

    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr(session_server.asyncio, "sleep", sleep)

    def respond(request):
        requests.append(request)
        content = body if isinstance(body, str) else json.dumps(body)
        return httpx.Response(503, text=content)

    result = asyncio.run(_proxy(session_server, respond, attempts=3))

    assert result["status_code"] == 503
    assert len(requests) == 3
    assert sleep.await_count == 2


async def _proxy(session_server, respond, *, attempts):
    server = object.__new__(session_server.SessionServer)
    server.args = SimpleNamespace(
        custom_rollout_request_hook_path="cookbook.common.hooks.gated_rollout_request_hook",
        rollout_request_weight_version_mode="none",
        rollout_request_retry_attempts=attempts,
        rollout_request_retry_sleep=0.125,
    )
    server.backend_url = "http://backend"
    request = SimpleNamespace(method="POST", query="", session_id="retry-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        server.client = client
        return await server.do_proxy(
            request,
            "generate",
            body=b'{"input_ids": [1, 2, 3]}',
            headers={"x-test": "preserved"},
        )
