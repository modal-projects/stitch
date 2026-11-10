"""SGLang's calls as Stitch makes them, against a fake SGLang that records each request."""

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from stitch.sglang import SGLang, SGLangConfig


class FakeSGLang:
    """Answers our endpoints as SGLang does; `refuse` makes the update fail with that message."""

    def __init__(self):
        self.calls: list[tuple[str, dict, str]] = []
        self.refuse: str | None = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.calls.append((self.path, body, self.headers["Authorization"]))
                status, reply = fake.reply(self.path)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, path: str) -> tuple[int, dict]:
        if path == "/prepare_weights_from_delta":
            return 200, {"success": True, "stats": {"read_s": 1.0}}
        if path == "/update_weights_from_delta" and self.refuse:
            return 400, {"success": False, "message": self.refuse}
        if path == "/update_weights_from_delta":
            return 200, {"success": True, "stats": {"gpu_s": 0.1}}
        return 200, {}


@pytest.fixture
def fake():
    fake = FakeSGLang()
    yield fake
    fake.server.shutdown()


def _sglang(fake: FakeSGLang, pause_mode: str = "in_place", env=None) -> SGLang:
    config = SGLangConfig(server_args={}, env=env or {}, pause_mode=pause_mode)
    return SGLang("/ckpt", version=3, port=fake.port, config=config, startup_timeout=60)


def _prepare_and_apply(sglang: SGLang, version: int) -> tuple:
    manifest = Path("/v/manifest.json")
    return sglang.prepare(manifest), sglang.apply(manifest, version)


def test_an_in_place_pause_never_flushes(fake):
    sglang = _sglang(fake)
    stats = _prepare_and_apply(sglang, 4)

    assert [(path, body) for path, body, _ in fake.calls] == [
        ("/prepare_weights_from_delta", {"manifest": "/v/manifest.json"}),
        ("/pause_generation", {"mode": "in_place"}),
        (
            "/update_weights_from_delta",
            {
                "manifest": "/v/manifest.json",
                "weight_version": "4",
                "flush_cache": False,
            },
        ),
        ("/continue_generation", {}),
    ]
    assert stats == ({"read_s": 1.0}, {"gpu_s": 0.1})
    assert sglang.version == 4


@pytest.mark.parametrize("mode", ["retract", "abort"])
def test_other_pauses_flush_with_the_update(fake, mode):
    _prepare_and_apply(_sglang(fake, mode), 4)

    assert [(path, body.get("flush_cache")) for path, body, _ in fake.calls] == [
        ("/prepare_weights_from_delta", None),
        ("/pause_generation", None),
        ("/update_weights_from_delta", True),
        ("/continue_generation", None),
    ]


def test_a_refused_update_still_continues_and_raises_sglangs_reason(fake):
    fake.refuse = "1 tensors differ from the trainer's after applying"
    sglang = _sglang(fake)

    with pytest.raises(RuntimeError, match="1 tensors differ"):
        _prepare_and_apply(sglang, 4)
    assert fake.calls[-1][0] == "/continue_generation"
    assert sglang.version == 3


def test_only_the_abort_before_the_container_ends_has_its_own_timeout(
    fake, monkeypatch
):
    timeouts = {}
    urlopen = urllib.request.urlopen

    def recording(request, timeout=None):
        timeouts[request.full_url.rsplit("/", 1)[-1]] = timeout
        return urlopen(request, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", recording)
    sglang = _sglang(fake)
    _prepare_and_apply(sglang, 4)
    sglang.abort_all()

    assert timeouts == {
        "prepare_weights_from_delta": None,
        "pause_generation": None,
        "update_weights_from_delta": None,
        "continue_generation": None,
        "abort_request": 60,
    }


def test_every_call_carries_the_admin_key(fake):
    sglang = _sglang(fake)
    _prepare_and_apply(sglang, 4)
    sglang.abort_all()

    assert {auth for _, _, auth in fake.calls} == {f"Bearer {sglang.admin_key}"}


def test_engine_env_defaults_unless_the_run_says_otherwise(fake):
    env = {"SGLANG_DSA_FUSE_TOPK": "1"}

    assert _sglang(fake, env=env).endpoint.extra_env == {
        "SGLANG_DSA_FUSE_TOPK": "1",
        "SGLANG_ENABLE_RESPONSE_COMPRESSION": "1",
    }
    off = _sglang(fake, env={**env, "SGLANG_ENABLE_RESPONSE_COMPRESSION": "0"})
    assert off.endpoint.extra_env["SGLANG_ENABLE_RESPONSE_COMPRESSION"] == "0"
    assert env == {"SGLANG_DSA_FUSE_TOPK": "1"}


def test_an_unknown_pause_mode_is_refused():
    with pytest.raises(ValueError, match="pause mode"):
        SGLangConfig(server_args={}, pause_mode="drain")
