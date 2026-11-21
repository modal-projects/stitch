"""SGLang's calls as Stitch makes them, against a fake SGLang that records each request."""

import json
import sys
import threading
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from stitch.sglang import SGLang, SGLangConfig

IDENTITY = {"engine_id": "e0", "rank_id": "0"}
PUBLICATION = {
    "manifest_path": "/v/s/weight_v000004/manifest.json",
    "target_version": 4,
}


class FakeSGLang:
    """Answers the GPU-delta controls as SGLang does; `refuse` makes the update fail with that message."""

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
        if path == "/get_gpu_delta_info":
            row = {"identity": IDENTITY, "stream_id": "s", "version": 3}
            return 200, {"success": True, "participants": [row]}
        if path == "/update_weights_from_gpu_delta" and self.refuse:
            return 409, {"success": False, "message": self.refuse}
        if path == "/update_weights_from_gpu_delta":
            return 200, {"success": True, "timings": {"pause_s": 0.1}}
        return 200, {}


@pytest.fixture
def fake():
    fake = FakeSGLang()
    yield fake
    fake.server.shutdown()


@pytest.fixture(autouse=True)
def miles_session(monkeypatch):
    """Miles' GPU-delta session, which engine images ship: its activation posts one update through the client."""
    module = types.ModuleType(
        "miles.backends.training_utils.weight_update.protocols.gpu_delta.session"
    )

    async def activate_engine(
        client,
        publication,
        participants,
        session_id,
        weight_version,
        timeout,
        pause_mode,
    ):
        request = dict(
            publication,
            participants=participants,
            session_id=session_id,
            pause_mode=pause_mode,
        )
        if weight_version is not None:
            request["weight_version"] = weight_version
        return await client._make_request("update_weights_from_gpu_delta", request)

    def committed_endpoint(info, participants, plan_digest):
        assert participants == [IDENTITY] and plan_digest == "plan"
        (row,) = info["participants"]
        return row["stream_id"], row["version"]

    module.activate_engine, module.committed_endpoint = (
        activate_engine,
        committed_endpoint,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)


def _sglang(fake: FakeSGLang, pause_mode: str = "in_place", env=None) -> SGLang:
    config = SGLangConfig(server_args={}, env=env or {}, pause_mode=pause_mode)
    return SGLang(
        "/ckpt",
        version=3,
        port=fake.port,
        config=config,
        startup_timeout=60,
        engine_id="e0",
    )


@pytest.mark.parametrize("mode", ["in_place", "retract"])
def test_a_publication_applies_through_miles_activation_in_the_runs_pause_mode(
    fake, mode
):
    sglang = _sglang(fake, mode)

    sglang.committed("plan")
    stats = sglang.apply(PUBLICATION, 4)

    (info, update) = [(path, body) for path, body, _ in fake.calls]
    assert info == ("/get_gpu_delta_info", {"engine_id": "e0"})
    assert update[0] == "/update_weights_from_gpu_delta"
    assert update[1]["participants"] == [IDENTITY] and update[1]["pause_mode"] == mode
    assert (update[1]["manifest_path"], update[1]["weight_version"]) == (
        PUBLICATION["manifest_path"],
        "4",
    )
    assert stats == {"pause_s": 0.1} and sglang.version == 4


def test_a_publication_before_the_last_of_a_catch_up_carries_no_label(fake):
    sglang = _sglang(fake)

    sglang.committed("plan")
    sglang.apply(PUBLICATION, None)

    assert "weight_version" not in fake.calls[-1][1] and sglang.version == 3


def test_the_committed_weights_are_the_receivers(fake):
    assert _sglang(fake).committed("plan") == ("s", 3)


def test_a_refused_update_raises_sglangs_reason_and_keeps_the_label(fake):
    fake.refuse = "stale source: weights at s:2"
    sglang = _sglang(fake)

    with pytest.raises(RuntimeError, match="stale source"):
        sglang.apply(PUBLICATION, 4)
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
    sglang.committed("plan")
    sglang.apply(PUBLICATION, 4)
    sglang.abort_all()

    assert timeouts == {
        "get_gpu_delta_info": None,
        "update_weights_from_gpu_delta": None,
        "abort_request": 60,
    }


def test_every_call_carries_the_admin_key(fake):
    sglang = _sglang(fake)
    sglang.apply(PUBLICATION, 4)
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


@pytest.mark.parametrize("mode", ["drain", "abort"])
def test_a_pause_sglangs_scheduler_lacks_is_refused(mode):
    """An abort is a tokenizer-side abort, which no GPU-delta apply takes."""
    with pytest.raises(ValueError, match="pause mode"):
        SGLangConfig(server_args={}, pause_mode=mode)


@pytest.mark.parametrize(
    "server_args, ranks",
    [
        ({"--tp-size": "4", "--attn-dp-size": "4"}, 4),
        # the deprecated spelling: SGLang moves the width into attention DP, the same 8 ranks
        ({"--tp-size": "8", "--dp-size": "8", "--enable-dp-attention": ""}, 8),
        ({"--tp-size": "4", "--attention-data-parallel-size": "4"}, 4),
        ({"--tp-size": "4"}, 1),
    ],
)
def test_dp_ranks_count_as_sglang_does(server_args, ranks):
    assert SGLangConfig(server_args=server_args).dp_ranks == ranks
