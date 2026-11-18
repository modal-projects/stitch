import asyncio
import contextlib
import json
from types import SimpleNamespace

import httpx

from cookbook.miles_disagg import session_tunnel

SESSION = "0123456789abcdef0123456789abcdef"
CHAT = f"/sessions/{SESSION}/v1/chat/completions"
TITLE_BODY = {
    "model": "model",
    "stream": True,
    "messages": [
        {"role": "system", "content": "You are a title generator. You output ONLY a thread title."},
        {"role": "user", "content": "Generate a title for this conversation:\n"},
    ],
}
MAIN_BODY = {
    "model": "model",
    "stream": True,
    "tools": [{"type": "function", "function": {"name": "bash"}}],
    "messages": [{"role": "system", "content": "You are opencode."}, {"role": "user", "content": "Fix it."}],
}


def _call(method, path, body=None, upstream_status=200, upstream_body=b"data: {}\n\ndata: [DONE]\n\n"):
    """Send one request through the filter; returns (response, requests the session server received)."""
    received = []

    def session_server(request: httpx.Request) -> httpx.Response:
        received.append((request.method, request.url.path, json.loads(request.content)))
        kind = "text/event-stream" if upstream_status == 200 else "application/json"
        return httpx.Response(upstream_status, content=upstream_body, headers={"content-type": kind})

    app = session_tunnel.filter_app("http://10.0.0.5:30003", transport=httpx.MockTransport(session_server))

    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tunnel") as client:
            content = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
            return await client.request(method, path, content=content)

    return asyncio.run(send()), received


def test_chat_completions_reach_the_session_server_unchanged():
    response, received = _call("POST", CHAT, MAIN_BODY)

    assert received == [("POST", CHAT, MAIN_BODY)]
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b"data: {}\n\ndata: [DONE]\n\n"


def test_title_calls_are_answered_without_the_session_server():
    response, received = _call("POST", CHAT, TITLE_BODY)

    assert received == []
    events = [line.removeprefix("data: ") for line in response.text.split("\n\n") if line]
    assert events[-1] == "[DONE]"
    chunk = json.loads(events[0])
    assert chunk["choices"][0]["delta"]["content"] == session_tunnel.TITLE
    assert chunk["choices"][0]["finish_reason"] == "stop"

    response, received = _call("POST", CHAT, {**TITLE_BODY, "stream": False})
    assert received == []
    assert response.json()["choices"][0]["message"]["content"] == session_tunnel.TITLE

    # The same system prompt with tools is a main-conversation call.
    _, received = _call("POST", CHAT, {**TITLE_BODY, "tools": MAIN_BODY["tools"]})
    assert len(received) == 1


def test_a_409_becomes_a_400_that_opencode_does_not_retry():
    response, _ = _call("POST", CHAT, MAIN_BODY, upstream_status=409, upstream_body=b'{"error": "truncated"}')

    assert response.status_code == 400
    assert response.json() == {
        "error": {"message": session_tunnel.CUT_OFF_ERROR, "type": "invalid_request_error"}
    }
    response, _ = _call("POST", CHAT, MAIN_BODY, upstream_status=503, upstream_body=b'{"error": "aborted"}')
    assert (response.status_code, response.json()) == (503, {"error": "aborted"})


def test_everything_but_a_session_chat_call_is_refused():
    for method, path, body in [
        ("GET", CHAT, None),
        ("DELETE", f"/sessions/{SESSION}", None),
        ("POST", "/sessions", {}),
        ("POST", f"/sessions/{SESSION}/samples", {}),
        ("GET", f"/sessions/{SESSION}", None),
        ("GET", "/health", None),
        ("POST", f"/sessions/{SESSION}/v1/messages", MAIN_BODY),
        ("POST", "/sessions/../health/v1/chat/completions", MAIN_BODY),
        ("POST", "/sessions/not-a-session-id/v1/chat/completions", MAIN_BODY),
        ("POST", CHAT, b"not json"),
        ("POST", CHAT, [1, 2]),
    ]:
        response, received = _call(method, path, body)
        assert received == [], (method, path)
        assert response.status_code in (400, 404, 405), (method, path, response.status_code)


def test_rollout_workers_hand_agents_the_tunnel_of_their_session_server(monkeypatch):
    tunnels = {"30000": "https://a.modal.host", "30003": "https://d.modal.host"}
    monkeypatch.setenv(session_tunnel.TUNNELS_ENV, json.dumps(tunnels))

    url = session_tunnel.public_session_url(f"http://10.0.0.5:30003/sessions/{SESSION}")

    assert url == f"https://d.modal.host/sessions/{SESSION}"


def test_open_route_serves_a_filter_per_port_and_publishes_tunnels(monkeypatch):
    configs, forwarded = [], []

    class Server:
        def __init__(self, config):
            configs.append(config)
            self.started = False

        async def serve(self):
            self.started = True

    @contextlib.contextmanager
    def forward(port):
        forwarded.append(port)
        yield SimpleNamespace(url=f"https://t{port}.modal.host")

    monkeypatch.setattr(
        session_tunnel, "uvicorn", SimpleNamespace(Server=Server, Config=lambda app, **kw: kw)
    )
    monkeypatch.setattr(session_tunnel.modal, "forward", forward)

    stack, env = session_tunnel.open_route("10.0.0.5", [30000, 30001])

    assert [(c["host"], c["port"]) for c in configs] == [("0.0.0.0", 31000), ("0.0.0.0", 31001)]
    assert forwarded == [31000, 31001]
    assert json.loads(env[session_tunnel.TUNNELS_ENV]) == {
        "30000": "https://t31000.modal.host",
        "30001": "https://t31001.modal.host",
    }
    assert env[session_tunnel.HOSTS_ENV] == "t31000.modal.host,t31001.modal.host"
    stack.close()
