"""Modal wrapper for rollout traffic and state probes.

The target pool must be deployed in the same environment because
``ModalFlashPool`` resolves names in the caller's environment. Results land on
the ``stitch-probe-results`` Volume under ``/<tag>/``.
"""

from __future__ import annotations

import asyncio
import json

import modal

RESULTS_ROOT = "/probe-results"
MINUTES = 60

app = modal.App("stitch-probes")
results_volume = modal.Volume.from_name("stitch-probe-results", version=2, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx")
    .add_local_python_source("stitch", "tools")
)


@app.function(image=image, volumes={RESULTS_ROOT: results_volume}, timeout=120 * MINUTES)
def poll(pool_app: str, pool_cls: str = "Server", interval: float = 2.0, duration: float = 3600.0, tag: str = "run") -> None:
    from tools.probes import poller

    out = f"{RESULTS_ROOT}/{tag}/server_info.jsonl"
    asyncio.run(poller.poll(pool_app, pool_cls, interval=interval, duration=duration, out_path=out))
    results_volume.commit()
    print(json.dumps(poller.summarize(out), indent=2))


@app.function(image=image, volumes={RESULTS_ROOT: results_volume}, timeout=120 * MINUTES)
def traffic(
    pool_app: str,
    pool_cls: str = "Server",
    model: str = "default",
    shape: str = "mixed",
    concurrency: int = 16,
    duration: float = 600.0,
    lag: int | None = None,
    tag: str = "run",
) -> None:
    from stitch.pools.modal_flash import ModalFlashPool
    from tools.probes import traffic as traffic_mod

    gateway = ModalFlashPool(pool_app, pool_cls).gateway_url()
    out = f"{RESULTS_ROOT}/{tag}/traffic-{shape}.jsonl"
    summary = asyncio.run(traffic_mod.run(
        gateway, model, shape=shape, concurrency=concurrency, duration=duration, lag=lag, out_path=out,
    ))
    results_volume.commit()
    print(json.dumps(summary, indent=2))
