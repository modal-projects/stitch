"""``sync_in_progress`` — the shared /server_info interpretation a deployment's engine-health
probe uses to suppress health blips while the reconciler is reloading weights."""

from __future__ import annotations

import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from stitch.service import sync_in_progress


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
        ({"prefetch_done": True, "sync_state": "COMMITTING"}, True),   # reloading
        ({"prefetch_done": True, "sync_state": "PREFETCHING"}, True),  # staging deltas
        ({"prefetch_done": False, "prefetch_error": None}, True),      # boot base-seed (IDLE)
        ({"prefetch_done": True, "sync_state": "IDLE"}, False),        # settled: a blip is real
        ({"prefetch_done": False, "prefetch_error": "boom"}, False),   # seed failed: report it
    ],
)
def test_sync_in_progress(info, expected):
    with _server_info(info) as url:
        assert sync_in_progress(url) is expected


def test_unreachable_sidecar_reports_error():
    # Nothing listening: best-effort False so the caller surfaces the engine error.
    assert sync_in_progress("http://127.0.0.1:1/server_info", timeout=0.2) is False
