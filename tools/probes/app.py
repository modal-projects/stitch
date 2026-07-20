"""Modal wrapper for the probes — everything runs in the ``stitch-dev`` environment.

The replay publisher needs the delta Volume mounted, so which volume to mount is fixed
at deploy time via ``PROBE_DELTA_VOLUME`` (one probe deploy per target recipe):

    PROBE_DELTA_VOLUME=stitch-delta-glm45-air-fp8 \\
      uv run --extra modal modal deploy -m tools.probes.app -e stitch-dev

The target pool must be deployed in the same environment (ModalFlashPool resolves names
in the caller's environment). Results land on the ``stitch-probe-results`` Volume under
``/<tag>/``; baselines are recorded and human-judged, never CI gates.
"""

from __future__ import annotations

import asyncio
import json
import os

import modal

DELTA_ROOT = "/delta-bulletin"
RESULTS_ROOT = "/probe-results"
MINUTES = 60

app = modal.App("stitch-probes")
delta_volume_name = os.environ.get("PROBE_DELTA_VOLUME", "stitch-probe-scratch")
delta_volume = modal.Volume.from_name(delta_volume_name, version=2, create_if_missing=True)
results_volume = modal.Volume.from_name("stitch-probe-results", version=2, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx")
    # Bake the deploy-time volume choice: the container re-imports this module without the
    # shell's PROBE_DELTA_VOLUME, so without this the store would resolve (and commit) a
    # different volume than the one mounted.
    .env({"PROBE_DELTA_VOLUME": delta_volume_name})
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


@app.function(image=image, volumes={DELTA_ROOT: delta_volume, RESULTS_ROOT: results_volume}, timeout=240 * MINUTES)
def replay(pool_app: str, source_run: str, pool_cls: str = "Server", cadence_s: float = 30.0, limit: int | None = None, tag: str = "run") -> None:
    from tools.probes.replay_publisher import replay as replay_chain

    delta_volume.reload()
    run_id = replay_chain(
        root=DELTA_ROOT, source_run=source_run, app_name=pool_app, cls_name=pool_cls,
        volume_name=delta_volume_name, cadence_s=cadence_s, limit=limit,
    )
    print(f"replay complete: run_id={run_id} tag={tag}")
