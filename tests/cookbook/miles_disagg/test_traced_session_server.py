"""The traced session server: one rid per proxied request, logged at send and at its end."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from cookbook.miles_disagg import traced_session_server as traced


def test_a_request_gets_a_fresh_rid_as_its_first_field():
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "{x}"}]})

    tagged, rid = traced.tag_body(body.encode())

    payload = json.loads(tagged)
    assert rid is not None and len(rid) == 32
    assert list(payload) == ["rid", "model", "messages"]
    assert payload == {"rid": rid, **json.loads(body)}
    assert traced.tag_body(body.encode())[1] != rid


@pytest.mark.parametrize(
    "body", [b"{}", b" { } ", b"[1, 2]", b"", b"not json", b'{"rid": "kept"}']
)
def test_only_an_untagged_json_object_gets_a_rid(body):
    tagged, rid = traced.tag_body(body)

    if body.strip().startswith(b"{") and b"rid" not in body:
        assert json.loads(tagged) == {"rid": rid}
    else:
        assert (tagged, rid) == (body, None)


def test_max_tokens_is_read_without_parsing_the_body():
    assert traced.max_tokens(b'{"max_tokens": 32768, "messages": []}') == 32768
    assert traced.max_tokens(b'{"max_completion_tokens":512}') == 512
    assert traced.max_tokens(b'{"messages": []}') is None


def _request(session_id: str = "s1") -> Any:
    return SimpleNamespace(method="POST", query="", session_id=session_id)


def test_a_returned_request_logs_send_and_recv_under_one_rid(caplog):
    seen = {}

    async def do_proxy(_server, _request, _path, *, body, headers):
        seen["body"] = json.loads(body)
        return {"status_code": 200, "response_body": b'{"ok": 1}', "headers": {}}

    tracer = traced.Tracer()
    with caplog.at_level(logging.INFO, logger="stitch.trace"):
        result = asyncio.run(
            tracer.proxy(
                do_proxy,
                None,
                _request(),
                "v1/chat/completions",
                body=b'{"max_tokens": 64}',
                headers={},
            )
        )

    rid = seen["body"]["rid"]
    messages = [r.getMessage() for r in caplog.records]
    assert result["status_code"] == 200
    assert messages[0].startswith(f"trace send rid={rid} session=s1 ")
    assert "max_tokens=64" in messages[0]
    assert messages[1].startswith(f"trace recv rid={rid} status=200 bytes=9 after=")
    assert tracer.inflight == {}


def test_a_cancelled_request_logs_its_cancel_and_still_cancels(caplog):
    started = asyncio.Event()

    async def do_proxy(_server, _request, _path, *, body, headers):
        started.set()
        await asyncio.Future()

    async def go(tracer):
        task = asyncio.ensure_future(
            tracer.proxy(do_proxy, None, _request(), "p", body=b"{}", headers={})
        )
        await started.wait()
        assert len(tracer.inflight) == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    tracer = traced.Tracer()
    with caplog.at_level(logging.INFO, logger="stitch.trace"):
        asyncio.run(go(tracer))

    assert caplog.records[-1].getMessage().startswith("trace cancel rid=")
    assert tracer.inflight == {}


def test_install_wraps_the_session_server_and_starts_its_statistics(monkeypatch):
    calls = []

    class SessionServer:
        async def _do_proxy(self, request, path, *, body, headers):
            calls.append(json.loads(body))
            return {"status_code": 200, "response_body": b"", "headers": {}}

    module = types.ModuleType("miles.rollout.session.server")
    module.SessionServer = SessionServer
    for name in ("miles", "miles.rollout", "miles.rollout.session"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.rollout.session.server", module)
    sys.modules["miles.rollout.session"].server = module

    tracer = traced.install()

    async def go():
        await SessionServer()._do_proxy(_request(), "p", body=b'{"a": 1}', headers={})
        assert tracer.watcher is not None and not tracer.watcher.done()
        tracer.watcher.cancel()

    asyncio.run(go())

    assert calls[0]["a"] == 1 and "rid" in calls[0]
    assert tracer.inflight == {}
