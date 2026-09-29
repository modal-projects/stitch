"""Modal wrapper for rollout traffic and state probes.

The target pool must be deployed in the same environment because
``ModalFlashPool`` resolves names in the caller's environment. Results land on
the ``stitch-probe-results`` Volume under ``/<tag>/``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import modal

RESULTS_ROOT = "/probe-results"
MINUTES = 60

app = modal.App("stitch-probes")
results_volume = modal.Volume.from_name(
    "stitch-probe-results", version=2, create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx")
    .add_local_python_source("stitch", "tools")
)


@app.function(
    image=image, volumes={RESULTS_ROOT: results_volume}, timeout=120 * MINUTES
)
def poll(
    pool_app: str,
    pool_cls: str = "Server",
    interval: float = 2.0,
    duration: float = 3600.0,
    tag: str = "run",
) -> None:
    from tools.fleet import poller

    out = f"{RESULTS_ROOT}/{tag}/server_info.jsonl"
    asyncio.run(
        poller.poll(
            pool_app, pool_cls, interval=interval, duration=duration, out_path=out
        )
    )
    results_volume.commit()
    print(json.dumps(poller.summarize(out), indent=2))


@app.function(
    image=image, volumes={RESULTS_ROOT: results_volume}, timeout=120 * MINUTES
)
def traffic(
    pool_app: str,
    pool_cls: str = "Server",
    gateway_function: str | None = None,
    affinity_header: str = "Modal-Session-ID",
    model: str = "default",
    shape: str = "mixed",
    concurrency: int = 16,
    duration: float = 600.0,
    lag: int | None = None,
    context_limit: int = 16384,
    seed: int = 0,
    tag: str = "run",
) -> None:
    from stitch.pools.modal_flash import ModalFlashFleet, ModalFlashPool
    from tools.fleet import traffic as traffic_mod

    pool = (
        ModalFlashFleet(
            pool_app,
            tuple(pool_cls.split(",")),
            gateway_function=gateway_function,
        )
        if gateway_function is not None
        else ModalFlashPool(pool_app, pool_cls)
    )
    gateway = pool.gateway_url()
    out = f"{RESULTS_ROOT}/{tag}/traffic-{shape}.jsonl"
    summary = asyncio.run(
        traffic_mod.run(
            gateway,
            model,
            affinity_header=affinity_header,
            shape=shape,
            concurrency=concurrency,
            duration=duration,
            lag=lag,
            context_limit=context_limit,
            out_path=out,
            seed=seed,
        )
    )
    results_volume.commit()
    print(json.dumps(summary, indent=2))


@app.function(
    image=image, volumes={RESULTS_ROOT: results_volume}, timeout=120 * MINUTES
)
def precision(
    pool_app: str,
    reference_pool_cls: str,
    candidate_pool_cls: str,
    model: str = "default",
    samples: int = 16,
    output_tokens: int = 256,
    seed: int = 0,
    tag: str = "run",
) -> None:
    from stitch.pools.modal_flash import ModalFlashPool
    from tools.fleet import precision as precision_mod

    reference = ModalFlashPool(pool_app, reference_pool_cls).gateway_url()
    candidate = ModalFlashPool(pool_app, candidate_pool_cls).gateway_url()
    result = asyncio.run(
        precision_mod.compare(
            reference,
            candidate,
            model,
            samples=samples,
            output_tokens=output_tokens,
            seed=seed,
        )
    )
    path = f"{RESULTS_ROOT}/{tag}/precision.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    results_volume.commit()
    print(json.dumps(result["summary"], indent=2))
