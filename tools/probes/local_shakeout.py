"""Local end-to-end shakeout — run this before trusting the harness anywhere real.

Assembles the real serving stack with no Modal and no GPUs::

    mock engine (HTTP)  <-  SGLangEngine  <-  Reconciler  <-  create_app   x2 sidecars
                                 ^ local-dir ModalVolumeStore (tmpdir)

then runs the probes against it exactly as they would run against a Modal pool:
``replay_publisher`` re-publishes a synthesized recorded chain (with one empty delta),
``traffic`` drives mixed load with a lag floor, and the ``poller`` scrapes
``/server_info``. Asserts: both replicas converge to the chain head, zero traffic
errors, the 409 constraint path answers correctly, and the summaries are well-formed.

    uv run --extra sglang --extra modal --with uvicorn python -m tools.probes.local_shakeout

No ``from __future__ import annotations`` here — same reason as service.py: the mock
engine's FastAPI handlers are introspected at runtime and their ``Request`` type is a
function-local import; stringized annotations would demote it to a query param (422s).
"""

import asyncio
import json
import socket
import tempfile
import time
from pathlib import Path
from typing import Any

DECODE_S = 0.4  # mock decode time; long enough that some requests straddle commits
CHAIN_LEN = 6
EMPTY_DELTAS = {4}  # exercises the skip-reload path
CADENCE_S = 1.0


# ── the fake pool ─────────────────────────────────────────────────────────────
def mock_engine() -> tuple[Any, dict[str, Any]]:
    """A stand-in sglang: answers the engine control surface + chat completions."""
    from fastapi import FastAPI, Request

    state: dict[str, Any] = {"pulls": [], "reloads": []}
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/pull_weights")
    async def pull_weights(request: Request) -> dict[str, bool]:
        state["pulls"].append((await request.json()).get("target_version"))
        return {"success": True}

    @app.post("/update_weights_from_disk")
    async def update_weights(request: Request) -> dict[str, bool]:
        state["reloads"].append((await request.json()).get("weight_version"))
        return {"success": True}

    @app.get("/flush_cache")
    async def flush_cache() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/pause_generation")
    @app.post("/continue_generation")
    async def pause_resume() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> dict[str, Any]:
        del request
        await asyncio.sleep(DECODE_S)
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    return app, state


class LocalPool:
    """A stitch Pool over plain local URLs (duck-typed; only what the probes use)."""

    def __init__(self, urls: list[str]) -> None:
        self.urls = urls

    def gateway_url(self) -> str:
        return self.urls[0]

    def discover_replicas(self) -> list[str]:
        return list(self.urls)

    def wake(self, replicas: list[str], ref: Any) -> None:
        import httpx

        for url in replicas:
            try:
                httpx.post(f"{url}/wake", timeout=5.0)
            except Exception:  # noqa: BLE001 — best-effort, like the real pool
                pass

    def scale(self, **_: Any) -> None:
        pass


def synth_chain(root: Path, run: str = "recorded", n: int = CHAIN_LEN) -> None:
    """Write a recorded-looking delta chain: weight_v* dirs with real HF-index metadata."""
    for v in range(1, n + 1):
        d = root / run / f"weight_v{v:06d}"
        d.mkdir(parents=True)
        weight_map: dict[str, str] = {} if v in EMPTY_DELTAS else {"layers.0.w": "delta-000.bin"}
        if weight_map:
            (d / "delta-000.bin").write_bytes(b"\x00" * 64)
        meta = {"version": v, "diff": "xor", "base_version": v - 1, "compression": "zstd", "checksum": "xxh3-128"}
        (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": meta, "weight_map": weight_map}))


# ── plumbing ──────────────────────────────────────────────────────────────────
def _free_ports(n: int) -> list[int]:
    ports, socks = [], []
    for _ in range(n):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        socks.append(s)
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


def _server(app: Any, port: int) -> Any:
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    server.install_signal_handlers = lambda: None  # several servers share this loop
    return server


async def _wait_healthy(urls: list[str], timeout: float = 15.0) -> None:
    import httpx

    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=2.0) as client:
        for url in urls:
            while True:
                try:
                    if (await client.get(f"{url}/health")).status_code == 200:
                        break
                except Exception:  # noqa: BLE001
                    if time.time() > deadline:
                        raise
                await asyncio.sleep(0.1)


# ── the shakeout ──────────────────────────────────────────────────────────────
async def main() -> None:
    import httpx

    from stitch.engines.sglang import SGLangEngine
    from stitch.service import create_app
    from stitch.stores.modal_volume import ModalVolumeStore
    from stitch.sync import Reconciler
    from tools.probes import poller, traffic
    from tools.probes.replay_publisher import replay

    tmp = Path(tempfile.mkdtemp(prefix="stitch-shakeout-"))
    root = tmp / "bulletin"
    synth_chain(root)
    engine_port, *sidecar_ports = _free_ports(3)

    engine_app, engine_state = mock_engine()
    servers = [_server(engine_app, engine_port)]
    tasks = [asyncio.create_task(servers[0].serve())]
    await _wait_healthy([f"http://127.0.0.1:{engine_port}"])  # engine first: sidecar startup prefetches from it

    sidecars = []
    for i, port in enumerate(sidecar_ports):
        engine = SGLangEngine(f"http://127.0.0.1:{engine_port}", str(tmp / f"ckpt{i}"))
        rec = Reconciler(store=ModalVolumeStore(root), engine=engine, commit_mode="in_place", reconcile_interval=0.2)
        servers.append(_server(create_app(rec, engine), port))
        sidecars.append(f"http://127.0.0.1:{port}")
    tasks += [asyncio.create_task(s.serve()) for s in servers[1:]]
    await _wait_healthy(sidecars)

    run_id, traffic_summary, _ = await asyncio.gather(
        asyncio.to_thread(
            replay, root=str(root), source_run="recorded", pool=LocalPool(sidecars), cadence_s=CADENCE_S,
        ),
        traffic.run(
            sidecars[0], "mock", shape="mixed", concurrency=6,
            duration=CHAIN_LEN * CADENCE_S + 4.0, lag=1, out_path=str(tmp / "traffic.jsonl"),
        ),
        poller.poll(
            "", replicas=sidecars, interval=0.5, duration=CHAIN_LEN * CADENCE_S + 6.0,
            out_path=str(tmp / "poll.jsonl"),
        ),
    )

    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(50):  # settle: both replicas at the chain head
            infos = [(await client.get(f"{u}/server_info")).json() for u in sidecars]
            if all(i["applied"] == f"{run_id}/weight_v{CHAIN_LEN:06d}" and i["ready"] for i in infos):
                break
            await asyncio.sleep(0.2)
        else:
            raise AssertionError(f"replicas never converged: {infos}")

        # the 409 constraint path, end to end
        resp = await client.post(
            f"{sidecars[0]}/v1/chat/completions",
            json={"model": "mock", "messages": [{"role": "user", "content": "x"}],
                  "max_tokens": 8, "weight_version": {"min_version": 999}},
        )
        assert resp.status_code == 409 and resp.json().get("type") == "WeightVersionNotReady", resp.text

    poll_summary = poller.summarize(str(tmp / "poll.jsonl"))
    assert CHAIN_LEN in poll_summary["versions_seen"], poll_summary
    errors = sum(s.get("errors", 0) for s in traffic_summary.values() if isinstance(s, dict))
    assert errors == 0, traffic_summary
    assert traffic_summary["requests"] > 0
    straddled = sum(s.get("straddled", 0) for s in traffic_summary.values() if isinstance(s, dict))

    for s in servers:
        s.should_exit = True
    await asyncio.gather(*tasks)

    print("traffic:", json.dumps(traffic_summary, indent=2))
    print("poller:", json.dumps(poll_summary, indent=2))
    print(f"engine: {len(engine_state['reloads'])} reloads, {len(engine_state['pulls'])} pulls")
    print(f"straddled requests: {straddled} (in_place, decode {DECODE_S}s vs cadence {CADENCE_S}s)")
    print(f"SHAKEOUT PASS (artifacts in {tmp})")


if __name__ == "__main__":
    asyncio.run(main())
