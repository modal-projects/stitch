"""Miles' session server with a trace line per proxied model request, for finding where
a request is lost between the session server and the engines.

Each proxied request gets a fresh SGLang request id (``rid``) in its JSON body. The stitch
sidecar keeps a body's rid and passes it to SGLang, so one id names the request at every
hop. This server logs the rid when the request leaves, when its response arrives, and when
it fails or is cancelled (the session server's deadline cancels it). Every 10 s it also
logs its event loop's lag, its requests in flight and the oldest one's age, its CPU use
and its memory, from its first request on.

``eval_driver.start_session_servers`` starts it in place of
``python -m miles.rollout.session.server`` when ``STITCH_TRACE_REQUESTS`` is set; it
takes the same arguments.
"""

from __future__ import annotations

import asyncio
import logging
import re
import resource
import time
import uuid
from typing import Any

logger = logging.getLogger("stitch.trace")

TRACE_ENV = "STITCH_TRACE_REQUESTS"
STATS_INTERVAL_SECONDS = 10.0
_MAX_TOKENS = re.compile(rb'"max_(?:completion_)?tokens"\s*:\s*(\d+)')


def tag_body(body: bytes) -> tuple[bytes, str | None]:
    """``body`` with a fresh ``rid`` as its first field, and that rid. A body that is
    not a JSON object, or already names a rid, passes through untagged. The rid is
    spliced into the bytes rather than re-encoding the whole conversation."""
    start = body.find(b"{")
    if start < 0 or body[:start].strip() or b'"rid"' in body:
        return body, None
    rid = uuid.uuid4().hex
    rest = body[start + 1 :]
    separator = b"" if rest.lstrip().startswith(b"}") else b","
    field = b'"rid":"' + rid.encode() + b'"' + separator
    return body[: start + 1] + field + rest, rid


def max_tokens(body: bytes) -> int | None:
    match = _MAX_TOKENS.search(body)
    return int(match.group(1)) if match else None


class Tracer:
    """The requests in flight, by rid, and the loop statistics over them."""

    def __init__(self) -> None:
        self.inflight: dict[str, float] = {}
        self.watcher: asyncio.Task | None = None

    async def proxy(
        self, do_proxy: Any, server: Any, request: Any, path: str, *, body, headers
    ) -> dict:
        if self.watcher is None:  # in the server's own loop, from its first request
            self.watcher = asyncio.ensure_future(self.watch())
        body, rid = tag_body(body)
        key = rid or uuid.uuid4().hex
        start = time.monotonic()
        self.inflight[key] = start
        logger.info(
            "trace send rid=%s session=%s path=%s bytes=%d max_tokens=%s",
            rid,
            request.session_id,
            path,
            len(body),
            max_tokens(body),
        )
        try:
            result = await do_proxy(server, request, path, body=body, headers=headers)
        except asyncio.CancelledError:
            logger.warning(
                "trace cancel rid=%s after=%.1f", rid, time.monotonic() - start
            )
            raise
        except BaseException as exc:
            logger.warning(
                "trace error rid=%s after=%.1f error=%s: %s",
                rid,
                time.monotonic() - start,
                type(exc).__name__,
                exc,
            )
            raise
        finally:
            self.inflight.pop(key, None)
        response = result.get("response_body") or b""
        status = result.get("status_code")
        logger.info(
            "trace recv rid=%s status=%s bytes=%d after=%.1f%s",
            rid,
            status,
            len(response),
            time.monotonic() - start,
            "" if status == 200 else f" body={response[:200]!r}",
        )
        return result

    async def watch(self, interval: float = STATS_INTERVAL_SECONDS) -> None:
        """Log loop lag, requests in flight, CPU and memory every ``interval``."""
        cpu = _cpu_seconds()
        while True:
            before = time.monotonic()
            await asyncio.sleep(interval)
            now = time.monotonic()
            used, cpu = _cpu_seconds() - cpu, _cpu_seconds()
            oldest = now - min(self.inflight.values()) if self.inflight else 0.0
            logger.info(
                "trace loop lag=%.2f inflight=%d oldest=%.0f cpu=%.0f%% rss_gb=%.1f",
                now - before - interval,
                len(self.inflight),
                oldest,
                100 * used / (now - before),
                _rss_gb(),
            )


def _cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _rss_gb() -> float:
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
    except OSError:
        return 0.0
    return pages * resource.getpagesize() / 2**30


def install() -> Tracer:
    """Trace every request Miles' ``SessionServer`` proxies."""
    from miles.rollout.session import server

    tracer = Tracer()
    do_proxy = server.SessionServer._do_proxy

    async def traced(self, request, path, *, body, headers):
        return await tracer.proxy(
            do_proxy, self, request, path, body=body, headers=headers
        )

    server.SessionServer._do_proxy = traced
    return tracer


def main(argv: list[str] | None = None) -> None:
    from miles.rollout.session import server

    install()
    server.main(argv)


if __name__ == "__main__":
    main()
