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
    .pip_install("httpx", "datasets")  # datasets: the rl_replay probe loads the RL dataset's prompts
    # Bake the deploy-time volume choice: containers re-import this module, where the
    # shell's PROBE_DELTA_VOLUME doesn't exist — without this the store would resolve
    # (and commit) a different volume than the one mounted.
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


@app.function(image=image, volumes={RESULTS_ROOT: results_volume}, timeout=120 * MINUTES, region=os.environ.get("PROBE_REGION") or None)
def rl_replay(
    pool_app: str,
    pool_cls: str = "Router",
    model: str = "default",
    dataset: str = "zhuzilin/gsm8k",
    batch_size: int = 64,
    group_size: int = 8,
    max_tokens: int = 16,
    duration: float = 600.0,
    seed: int = 0,
    tag: str = "run",
) -> None:
    """GRPO-shaped replay of the RL dataset's prompts (see traffic.run_rl_replay).
    Deploy with PROBE_REGION set to the router's region so the client leg is constant."""
    from stitch.pools.modal_flash import ModalFlashPool
    from tools.probes import traffic as traffic_mod

    gateway = ModalFlashPool(pool_app, pool_cls).gateway_url()
    out = f"{RESULTS_ROOT}/{tag}/rl_replay.jsonl"
    summary = asyncio.run(traffic_mod.run_rl_replay(
        gateway, model, dataset=dataset, batch_size=batch_size, group_size=group_size,
        max_tokens=max_tokens, duration=duration, seed=seed, out_path=out,
    ))
    results_volume.commit()
    print(json.dumps(summary, indent=2))


@app.function(image=image, volumes={RESULTS_ROOT: results_volume}, timeout=180 * MINUTES)
def bfcl_replay(
    pool_app: str,
    pool_cls: str = "Router",
    model: str = "default",
    trajectories: str = f"{RESULTS_ROOT}/bfcl/bfcl_replay.jsonl",
    concurrency: int = 48,
    max_tokens: int = 256,
    duration: float = 600.0,
    seed: int = 0,
    tag: str = "run",
) -> None:
    """Teacher-forced BFCL multi-turn replay (see traffic.run_bfcl_replay). The
    trajectories file comes from bfcl_prep.py, uploaded to the results volume."""
    from stitch.pools.modal_flash import ModalFlashPool
    from tools.probes import traffic as traffic_mod

    gateway = ModalFlashPool(pool_app, pool_cls).gateway_url()
    out = f"{RESULTS_ROOT}/{tag}/bfcl_replay.jsonl"
    summary = asyncio.run(traffic_mod.run_bfcl_replay(
        gateway, model, trajectories_path=trajectories, concurrency=concurrency,
        max_tokens=max_tokens, duration=duration, seed=seed, out_path=out,
    ))
    results_volume.commit()
    print(json.dumps(summary, indent=2))


@app.function(image=image, volumes={DELTA_ROOT: delta_volume}, timeout=10 * MINUTES)
def claim(pool_app: str, pool_cls: str = "Server", run_id: str = "") -> None:
    """Claim the pool at base (weight_v0) without a trainer — a serving-only bench
    needs replicas 'ready', which requires an applied version. ``pool_cls`` accepts
    a comma-separated list for multi-class (geo) pools. Deploy this app with
    PROBE_DELTA_VOLUME set to the target recipe's delta volume."""
    import uuid

    from stitch.pools.modal_flash import ModalFlashPool
    from stitch.pools.union import UnionPool
    from stitch.publish import claim_run
    from stitch.stores.modal_volume import ModalVolumeStore

    members = [ModalFlashPool(pool_app, c.strip()) for c in pool_cls.split(",") if c.strip()]
    pool = members[0] if len(members) == 1 else UnionPool(members)
    store = ModalVolumeStore(DELTA_ROOT, volume_name=delta_volume_name)
    rid = run_id or f"bench-{uuid.uuid4().hex[:8]}"
    claim_run(store, pool, rid)
    print(f"claimed run_id={rid} on {pool_app}.{pool_cls} (volume={delta_volume_name})")


@app.function(image=image, volumes={DELTA_ROOT: delta_volume, RESULTS_ROOT: results_volume}, timeout=240 * MINUTES)
def replay(pool_app: str, source_run: str, pool_cls: str = "Server", cadence_s: float = 30.0, limit: int | None = None, tag: str = "run") -> None:
    from tools.probes.replay_publisher import replay as replay_chain

    delta_volume.reload()
    run_id = replay_chain(
        root=DELTA_ROOT, source_run=source_run, app_name=pool_app, cls_name=pool_cls,
        volume_name=delta_volume_name, cadence_s=cadence_s, limit=limit,
    )
    print(f"replay complete: run_id={run_id} tag={tag}")
