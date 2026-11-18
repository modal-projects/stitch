"""Public route from agents inside task sandboxes (OpenCode) to the trainer's session servers.

Rollout workers hand an agent `http://<trainer ip>:<port>/sessions/<id>`, which a sandbox cannot reach. For OpenCode
recipes, rank 0 of the trainer opens one modal.forward tunnel per session-server port. Tunnels are public, so each one
ends at a filter in front of its session server, not at the server itself. The filter:
- passes only POST /sessions/<id>/v1/chat/completions; anything else gets 404 or 405;
- answers OpenCode's title-generator call itself (a side call at the start of each OpenCode session): forwarded, it
  would become one more training sample with the trajectory's reward;
- answers a 409 from the session server (a request that extends a reply cut off at the reply limit) with a 400,
  which OpenCode does not retry (it retries a 409 five times over about a minute).
Rollout workers read the tunnel URLs from SESSION_TUNNELS (set before Ray starts) and give the agent
`<tunnel url>/sessions/<id>`; HARBOR_AGENT_ALLOWED_HOSTS opens the tunnel hosts for the agent phase of offline tasks.
"""

import asyncio
import contextlib
import json
import os
import re
import threading
import time
from urllib.parse import urlsplit

import httpx
import modal
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

try:
    import uvicorn
except ImportError:  # outside the trainer image (tests)
    uvicorn = None

TUNNELS_ENV = "SESSION_TUNNELS"
FILTER_PORT_OFFSET = 1000  # the filter for session-server port p listens on p + 1000
SESSION_ID = re.compile(r"[0-9a-f]{32}")  # Miles' session ids: uuid4().hex
TITLE_PROMPT = "You are a title generator"
TITLE = "Practice task"
# No words that OpenCode reads as retryable (server error, timeout, connection, ...) or as a context overflow.
CUT_OFF_ERROR = "The previous reply was cut off at the reply limit, so this conversation cannot continue."


def is_title_call(body: dict) -> bool:
    """OpenCode's title generator: a system prompt that starts with TITLE_PROMPT and no tools."""
    messages = body.get("messages") or [{}]
    first = messages[0]
    return (
        not body.get("tools")
        and first.get("role") == "system"
        and str(first.get("content", "")).startswith(TITLE_PROMPT)
    )


def title_reply(body: dict) -> Response:
    message = {"role": "assistant", "content": TITLE}
    reply = {
        "id": "chatcmpl-title",
        "created": int(time.time()),
        "model": body.get("model"),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if not body.get("stream"):
        choice = {"index": 0, "message": message, "finish_reason": "stop"}
        return JSONResponse({**reply, "object": "chat.completion", "choices": [choice]})
    choice = {"index": 0, "delta": message, "finish_reason": "stop"}
    chunk = json.dumps({**reply, "object": "chat.completion.chunk", "choices": [choice]})
    return Response(f"data: {chunk}\n\ndata: [DONE]\n\n", media_type="text/event-stream")


def filter_app(target: str, transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    """The filter in front of the session server at `target` (http://host:port)."""
    # No read timeout: a reply arrives whole (the session server fakes streaming) after up to an hour.
    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(None, connect=30.0),
        limits=httpx.Limits(max_connections=1024),
    )

    async def chat(request: Request) -> Response:
        session = request.path_params["session"]
        if not SESSION_ID.fullmatch(session):
            return Response(status_code=404)
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            return JSONResponse({"error": {"message": "invalid JSON"}}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": {"message": "expected a JSON object"}}, status_code=400)
        if is_title_call(body):
            print(f"[session-tunnel] answered a title call for session {session}", flush=True)
            return title_reply(body)
        upstream = await client.post(
            f"{target}{request.url.path}", content=raw, headers={"content-type": "application/json"}
        )
        if upstream.status_code != 200:
            print(f"[session-tunnel] session {session}: HTTP {upstream.status_code} from the session server", flush=True)
        if upstream.status_code == 409:
            error = {"message": CUT_OFF_ERROR, "type": "invalid_request_error"}
            return JSONResponse({"error": error}, status_code=400)
        return Response(
            upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get("content-type")
        )

    return Starlette(routes=[Route("/sessions/{session}/v1/chat/completions", chat, methods=["POST"])])


def open_route(host_ip: str, ports: list[int]) -> tuple[contextlib.ExitStack, dict[str, str]]:
    """Serve a filter for each session-server port and open its tunnel.

    Returns the stack that keeps the tunnels open and the environment rollout workers need.
    """
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                filter_app(f"http://{host_ip}:{port}"),
                host="0.0.0.0",
                port=port + FILTER_PORT_OFFSET,
                log_level="warning",
            )
        )
        for port in ports
    ]

    async def serve() -> None:
        await asyncio.gather(*(server.serve() for server in servers))

    threading.Thread(target=asyncio.run, args=(serve(),), name="session-tunnel", daemon=True).start()
    deadline = time.monotonic() + 60
    while not all(server.started for server in servers):
        if time.monotonic() > deadline:
            raise RuntimeError(f"session filters did not start on ports {[p + FILTER_PORT_OFFSET for p in ports]}")
        time.sleep(0.2)
    stack = contextlib.ExitStack()
    urls = {str(port): stack.enter_context(modal.forward(port + FILTER_PORT_OFFSET)).url for port in ports}
    print(f"[session-tunnel] {len(urls)} tunnels open for session-server ports {ports}", flush=True)
    hosts = ",".join(urlsplit(url).hostname for url in urls.values())
    return stack, {TUNNELS_ENV: json.dumps(urls), "HARBOR_AGENT_ALLOWED_HOSTS": hosts}


def public_session_url(base_url: str) -> str:
    """`http://<ip>:<port>/sessions/<id>` with its port's tunnel in place of `http://<ip>:<port>`."""
    url = urlsplit(base_url)
    return json.loads(os.environ[TUNNELS_ENV])[str(url.port)] + url.path
