import json

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from stitch.pools.base import Pool
from stitch.routing.service import create_router_app
from stitch.versions import VersionRef

REPLICAS = ["http://replica-a", "http://replica-b"]


class FakePool(Pool):
    def __init__(self, urls):
        self.urls = list(urls)

    def gateway_url(self) -> str:
        return "http://gateway"

    def discover_replicas(self) -> list[str]:
        return list(self.urls)

    def wake(self, replicas, ref: VersionRef) -> None:  # pragma: no cover
        pass

    def scale(self, *, min=None, max=None) -> None:  # pragma: no cover
        pass


def make_upstream(record: list[dict]):
    """A stub replica sidecar serving every REPLICAS host. Records the
    requests it receives (host + path + body) for assertions."""
    upstream = FastAPI()

    @upstream.get("/health")
    async def health():
        return {"ok": True}

    @upstream.get("/server_info")
    async def server_info(request: Request):
        return {
            "ready": True,
            "applied": "run-x/weight_v000005",
            "sync_state": "IDLE",
            "active_requests": 0,
            "host": request.headers.get("host"),
        }

    @upstream.get("/metrics")
    async def metrics():
        return Response(
            content="sglang:num_running_reqs 1\nsglang:num_used_tokens 10\n",
            media_type="text/plain",
        )

    @upstream.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        record.append(
            {"host": request.headers.get("host"), "path": "/generate", "body": body}
        )
        if body.get("simulate") == "409":
            return JSONResponse({"error": {"type": "ConstraintUnmet"}}, status_code=409)
        return {
            "text": "ok",
            "meta_info": {"prompt_tokens": 3, "completion_tokens": 5},
        }

    @upstream.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        record.append({"host": request.headers.get("host"), "path": "/chat", "body": body})
        if body.get("stream"):
            async def sse():
                yield b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
                yield b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
                yield (
                    b'data: {"choices": [], "usage": '
                    b'{"prompt_tokens": 7, "completion_tokens": 1}}\n\n'
                )
                yield b"data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")
        return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 1}}

    return upstream


@pytest.fixture()
def rig():
    record: list[dict] = []
    upstream = make_upstream(record)
    transport = httpx.ASGITransport(app=upstream)
    app = create_router_app(
        FakePool(REPLICAS),
        policy="least-request",
        metrics_interval=3600,
        discovery_interval=3600,
        upstream_transport=transport,
    )
    with TestClient(app) as client:
        yield client, record, app


def test_startup_primes_discovery_and_scrape(rig):
    client, _, _ = rig
    r = client.get("/router/replicas")
    assert r.status_code == 200
    body = r.json()
    assert body["replicas"] == REPLICAS
    assert body["server_info"][REPLICAS[0]]["ready"] is True
    stats = client.get("/router/stats").json()
    assert stats["replicas"][REPLICAS[0]]["metrics"]["num_running_reqs"] == 1
    assert stats["replicas"][REPLICAS[0]]["network_rtt_s"] is not None


def test_generate_proxies_body_untouched_and_records(rig):
    client, record, _ = rig
    payload = {
        "input_ids": [1, 2, 3],
        "weight_version": {"min_version": 1},
        "sampling_params": {"max_new_tokens": 5},
    }
    r = client.post("/generate", json=payload)
    assert r.status_code == 200
    assert r.json()["text"] == "ok"
    # The upstream saw the body byte-identical, weight_version included —
    # constraint gating belongs to the sidecar.
    assert record[-1]["body"] == payload
    stats = client.get("/router/stats").json()
    assert stats["total_requests"] == 1
    assert stats["radix_trie"]["num_sequences"] == 1  # 2xx -> trie tagged
    # Counters fully released after completion.
    for u in REPLICAS:
        assert stats["replicas"][u]["queued_tokens"] == 0
        assert stats["replicas"][u]["inflight_requests"] == 0
    assert client.get("/router/samples").json()["buffered_samples"] == 1


def test_409_passes_through_without_sample(rig):
    client, _, _ = rig
    r = client.post("/generate", json={"input_ids": [1], "simulate": "409"})
    assert r.status_code == 409
    stats = client.get("/router/stats").json()
    assert stats["radix_trie"]["num_sequences"] == 0  # no trie tag on non-2xx
    assert client.get("/router/samples").json()["buffered_samples"] == 0
    for u in REPLICAS:
        assert stats["replicas"][u]["inflight_requests"] == 0


def test_sse_streaming_tees_and_records(rig):
    client, _, _ = rig
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [line for line in r.text.splitlines() if line.startswith("data:")]
    assert any('"content": "hi"' in e for e in events)
    assert events[-1] == "data: [DONE]"
    sample = client.get("/router/samples").json()["recent"][-1]
    assert sample["prompt_tokens"] == 7 and sample["completion_tokens"] == 1
    assert sample["ttft_seconds"] <= sample["total_seconds"]


def test_non_routed_paths_proxy_through(rig):
    client, _, _ = rig
    r = client.get("/server_info")
    assert r.status_code == 200
    assert r.json()["ready"] is True
    # Non-routed traffic never touches the load counters.
    stats = client.get("/router/stats").json()
    assert all(v["inflight_requests"] == 0 for v in stats["replicas"].values())


def test_session_affinity_pins_requests(rig):
    client, record, _ = rig
    client.post("/router/policy", json={"policy": "session-affinity"})
    for _ in range(4):
        client.post(
            "/generate",
            json={"input_ids": [1]},
            headers={"Modal-Session-ID": "sess-7"},
        )
    hosts = {r["host"] for r in record}
    assert len(hosts) == 1  # all four landed on the same replica


def test_policy_admin_endpoints(rig):
    client, _, _ = rig
    r = client.get("/router/policy")
    assert r.json()["policy"] == "least-request"
    assert "gorgo" in r.json()["available"]
    assert client.post("/router/policy", json={"policy": "bogus"}).status_code == 400
    assert client.post("/router/policy", json={"policy": "GORGO"}).status_code == 200
    assert client.get("/router/policy").json()["policy"] == "gorgo"


def test_hyperparameters_admin(rig):
    client, _, _ = rig
    r = client.post("/router/hyperparameters", json={"rtt_weight": 2.5})
    assert r.status_code == 200
    assert r.json()["hyperparameters"]["defaults"]["rtt_weight"] == 2.5
    assert client.post("/router/hyperparameters", json={"nope": 1}).status_code == 400
    r = client.put("/router/hyperparameters", json={})
    assert r.json()["hyperparameters"]["defaults"]["rtt_weight"] == 1.0  # PUT resets


def test_tune_admin_gated_on_policy(rig):
    client, _, _ = rig
    assert client.post("/router/tune", json={}).status_code == 400  # least-request
    client.post("/router/policy", json={"policy": "gorgo"})
    r = client.post("/router/tune", json={"objective_metric": "neg_p95_e2e"})
    assert r.status_code == 200
    assert r.json()["auto_tune"]["enabled"] is True
    assert client.get("/router/tune").json()["auto_tune"]["objective_metric"] == "neg_p95_e2e"


def test_flush_clears_trie_and_samples(rig):
    client, _, _ = rig
    client.post("/generate", json={"input_ids": [1, 2, 3]})
    assert client.get("/router/stats").json()["radix_trie"]["num_sequences"] == 1
    r = client.post("/router/flush")
    assert r.json()["radix_trie_cleared"] is True
    stats = client.get("/router/stats").json()
    assert stats["radix_trie"]["num_sequences"] == 0
    assert stats["buffered_samples"] == 0


def test_no_replicas_returns_503():
    app = create_router_app(
        FakePool([]),
        metrics_interval=3600,
        discovery_interval=3600,
        upstream_transport=httpx.ASGITransport(app=make_upstream([])),
    )
    with TestClient(app) as client:
        r = client.post("/generate", json={"input_ids": [1]})
        assert r.status_code == 503
        assert r.json()["error"]["type"] == "NoReplicas"


def test_health_endpoint(rig):
    client, _, _ = rig
    body = client.get("/health").json()
    assert body["ok"] is True and body["replicas"] == 2


def test_gorgo_routes_to_cached_replica_end_to_end(rig):
    client, record, _ = rig
    client.post("/router/policy", json={"policy": "gorgo"})
    prompt = list(range(64))
    # Seed: first request lands somewhere and tags the trie there.
    client.post("/generate", json={"input_ids": prompt})
    first_host = record[-1]["host"]
    # Repeats must chase the KV cache to the same replica.
    for _ in range(3):
        client.post("/generate", json={"input_ids": prompt})
        assert record[-1]["host"] == first_host
