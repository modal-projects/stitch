"""Weight-sync measurement probes: where does the SGLang reload time go?

Brings up ONE rollout engine identical to the miles_disagg Server cls (same
serving image, same SGLangEndpoint(model_path, tp, SGLANG_SERVER_ARGS), same
volumes + ephemeral disk) and times each phase of the weight-update path,
isolating disk-read throughput from SGLang's load processing:

  1. raw cold-read throughput of a base shard on the /prep Volume,
  2. SGLang cold startup load (reads the base from /prep),
  3. parallel materialize /prep base -> /local-checkpoint (ephemeral NVMe),
  4. raw cold+warm read throughput of a /local-checkpoint shard — the disk the
     steady-state reload actually reads,
  5. reload from /local-checkpoint with a COLD page cache,
  6. immediate second reload (WARM page cache; doubles as the two-consecutive-
     reloads stability gate for the quant postprocess),
  7. optionally, host-side delta apply + reload (pass --delta-run-id/--version).

With the instrumented serving pin, phases 5-7 also emit the engine's
"[reload timing] iter_wait=..s load=..s postprocess=..s total=..s" line, which
splits disk/materialize waits from weight_loader dispatch and the quant
postprocess — read those (and "[disk delta apply] ..." for phase 7) from the
same app logs as this probe's summary.

Run (its own app, no collision with the main deploy):
    EXPERIMENT_CONFIG=kimi_k2_6_nvfp4_disagg MODAL_PROFILE=modal-labs \
      uv run --extra modal modal run --detach -m cookbook.miles_disagg.modal_probes::profile_reload
Read results: modal app logs -e jason-dev miles-kimi-probe
"""

from __future__ import annotations

import glob
import os
import subprocess
import time

import modal

# Reuse the EXACT image / volumes / constants / engine args the Server cls uses.
from cookbook.miles_disagg import modal_train as mt

probe_app = modal.App("miles-kimi-probe")

# The reference NVFP4 checkpoint the anchor probe cold-loads/reloads (must
# already be in the HF cache volume; the probe never downloads it).
DEFAULT_ANCHOR_MODEL = "nvidia/Kimi-K2.6-NVFP4"


def _drop_page_caches() -> bool:
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[diskread] (could not drop page cache: {e})")
        return False


def _dd_read_gbps(path: str, label: str, *, cold: bool = True) -> float:
    """Read a file with dd (optionally dropping the page cache first); return GB/s."""
    if cold:
        _drop_page_caches()
    size = os.path.getsize(path)
    t0 = time.perf_counter()
    subprocess.run(["dd", f"if={path}", "of=/dev/null", "bs=16M"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    gbps = size / 1e9 / dt
    print(f"[diskread] {label}: {size/1e9:.1f} GB in {dt:.1f}s = {gbps:.2f} GB/s  ({path})")
    return gbps


@probe_app.function(
    image=mt.server_image,
    gpu=f"{mt.modal_cfg.gpu}:{mt.miles_cfg.rollout_num_gpus_per_engine}",
    cloud=mt.modal_cfg.cloud,
    region=mt.modal_cfg.region,
    volumes={
        str(mt.HF_CACHE_PATH): mt.hf_cache_volume,
        str(mt.PREP_PATH): mt.prep_volume,
        mt.SGLANG_CACHE_PATH: mt.sglang_cache_volume,
        mt.exp.DELTA_BULLETIN_ROOT: mt.delta_volume,
    },
    ephemeral_disk=mt.modal_cfg.rollout_ephemeral_disk_mib,
    timeout=60 * mt.MINUTES,
)
def profile_reload(delta_run_id: str = "", delta_version: int = 0) -> None:
    import asyncio

    import httpx
    from autoinference_utils.endpoint import SGLangEndpoint

    from cookbook.sidecar import parallel_init_local_checkpoint

    base = mt.MODEL_NAME  # the served base on /prep
    local = mt.LOCAL_CHECKPOINT_PATH  # /local-checkpoint (ephemeral)
    sglang_url = f"http://127.0.0.1:{mt.SGLANG_PORT}"
    R: dict[str, object] = {}  # durable results, printed once at the end

    def base_shards():
        return sorted(glob.glob(f"{base}/*.safetensors"))

    def alive() -> bool:
        try:
            return httpx.get(f"{sglang_url}/health", timeout=10.0).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def _reload(model_path: str) -> float:
        t = time.perf_counter()
        async with httpx.AsyncClient(timeout=None, trust_env=False) as c:
            r = await c.post(f"{sglang_url}/update_weights_from_disk",
                             json={"model_path": model_path, "weight_version": "probe", "flush_cache": False})
            r.raise_for_status()
            body = r.json()
            if body.get("success") is False:
                raise RuntimeError(f"reject: {body}")
            print(f"[reload message] {body.get('message')}")
        return time.perf_counter() - t

    print(f"=== PROBE START — base={base} tp={mt.miles_cfg.rollout_num_gpus_per_engine} ===")
    R["base_shards"] = len(base_shards())
    R["base_total_gb"] = round(sum(os.path.getsize(p) for p in base_shards()) / 1e9, 1)

    # 1. raw read throughput of a base shard (Modal Volume /prep)
    if base_shards():
        R["dd_prep_gbps"] = round(_dd_read_gbps(base_shards()[0], "prep-volume base shard (cold)"), 2)

    # 2. SGLang cold startup load (reads base from /prep Volume) — the cold-load path
    print("[phase] SGLang startup (cold load of base from /prep)...")
    t0 = time.perf_counter()
    endpoint = SGLangEndpoint(
        model_path=base, worker_port=mt.SGLANG_PORT,
        tp=mt.miles_cfg.rollout_num_gpus_per_engine, extra_server_args=mt.SGLANG_SERVER_ARGS,
        health_timeout=mt.SERVER_STARTUP_TIMEOUT, health_poll_interval=10.0,
    )
    endpoint.start()
    R["startup_coldload_from_prep_s"] = round(time.perf_counter() - t0, 1)
    print(f"[phase] startup cold load: {R['startup_coldload_from_prep_s']}s")

    # 3. materialize base -> /local-checkpoint, the disk every steady-state
    #    reload actually reads
    print("[phase] materialize base -> /local-checkpoint (32 workers)...")
    t0 = time.perf_counter()
    parallel_init_local_checkpoint("miles.utils.disk_delta")(local, base)
    R["materialize_s"] = round(time.perf_counter() - t0, 1)
    print(f"[phase] materialize: {R['materialize_s']}s")

    # 4. raw ephemeral-NVMe read rate, cold and warm — the disk-bound-vs-
    #    loader-bound reference points for the reload numbers below
    local_shards = sorted(glob.glob(f"{local}/*.safetensors"))
    if local_shards:
        R["dd_local_cold_gbps"] = round(_dd_read_gbps(local_shards[0], "ephemeral /local-checkpoint shard (cold)"), 2)
        R["dd_local_warm_gbps"] = round(
            _dd_read_gbps(local_shards[0], "ephemeral /local-checkpoint shard (warm)", cold=False), 2
        )

    # 5. reload from /local-checkpoint with a cold page cache (the steady-state
    #    worst case: the checkpoint got evicted during the rollout)
    if alive():
        print("[phase] reload /local-checkpoint (cold page cache)...")
        R["page_cache_dropped"] = _drop_page_caches()
        try:
            R["reload_local_cold_s"] = round(asyncio.run(_reload(local)), 1)
        except Exception as e:  # noqa: BLE001
            R["reload_local_cold"] = f"CRASH: {str(e)[:120]}"
            print(f"[phase] cold reload CRASH: {e}")

    # 6. immediate second reload (warm page cache) — isolates disk reads from
    #    dispatch+postprocess, and gates on the quant postprocess surviving
    #    consecutive reloads in one process
    if alive():
        print("[phase] reload /local-checkpoint again (warm page cache)...")
        try:
            R["reload_local_warm_s"] = round(asyncio.run(_reload(local)), 1)
            R["consecutive_reload"] = "OK"
        except Exception as e:  # noqa: BLE001
            R["consecutive_reload"] = f"CRASH: {str(e)[:120]}"
            print(f"[phase] warm reload CRASH: {e}")

    # 7. optional: host-side delta apply + reload (the full steady-state cycle)
    if delta_run_id and alive():
        from miles.utils.disk_delta import apply_deltas

        delta_root = f"{mt.exp.DELTA_BULLETIN_ROOT}/{delta_run_id}"
        print(f"[phase] apply delta chain {delta_root} up to v{delta_version} + reload...")
        t0 = time.perf_counter()
        R["delta_apply_stats"] = apply_deltas(local, delta_root, delta_version)
        R["delta_apply_s"] = round(time.perf_counter() - t0, 1)
        try:
            R["reload_after_delta_s"] = round(asyncio.run(_reload(local)), 1)
        except Exception as e:  # noqa: BLE001
            R["reload_after_delta"] = f"CRASH: {str(e)[:120]}"

    print("=== PROBE SUMMARY ===")
    for k, v in R.items():
        print(f"  {k}: {v}")
    try:
        endpoint.stop()
    except Exception:  # noqa: BLE001
        pass


@probe_app.function(
    image=mt.server_image,
    gpu=f"{mt.modal_cfg.gpu}:{mt.miles_cfg.rollout_num_gpus_per_engine}",
    cloud=mt.modal_cfg.cloud,
    region=mt.modal_cfg.region,
    volumes={
        str(mt.HF_CACHE_PATH): mt.hf_cache_volume,
        str(mt.PREP_PATH): mt.prep_volume,
        mt.SGLANG_CACHE_PATH: mt.sglang_cache_volume,
    },
    ephemeral_disk=mt.modal_cfg.rollout_ephemeral_disk_mib,
    timeout=60 * mt.MINUTES,
)
def profile_anchor_reload(anchor_model: str = DEFAULT_ANCHOR_MODEL) -> None:
    """Cold-load a reference checkpoint of the same model (e.g. the nvidia NVFP4
    anchor), then reload IT via update_weights_from_disk — the ours-vs-anchor
    comparison that separates checkpoint-layout effects from engine reload-path
    behavior. Record the resolved moe_runner_backend from the server logs with
    every number: anchor and ours may resolve differently."""
    import asyncio

    import httpx
    from autoinference_utils.endpoint import SGLangEndpoint
    from huggingface_hub import snapshot_download

    anchor = snapshot_download(anchor_model, local_files_only=True)
    sglang_url = f"http://127.0.0.1:{mt.SGLANG_PORT}"
    R: dict[str, object] = {"anchor_path": anchor}
    print(f"=== ANCHOR PROBE — {anchor_model} @ {anchor} ===")

    def alive() -> bool:
        try:
            return httpx.get(f"{sglang_url}/health", timeout=10.0).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def _reload(model_path: str) -> float:
        t = time.perf_counter()
        async with httpx.AsyncClient(timeout=None, trust_env=False) as c:
            r = await c.post(f"{sglang_url}/update_weights_from_disk",
                             json={"model_path": model_path, "weight_version": "probe", "flush_cache": False})
            r.raise_for_status()
            body = r.json()
            if body.get("success") is False:
                raise RuntimeError(f"reject: {body}")
            print(f"[reload message] {body.get('message')}")
        return time.perf_counter() - t

    print("[phase] cold-load anchor...")
    t0 = time.perf_counter()
    endpoint = SGLangEndpoint(
        model_path=anchor, worker_port=mt.SGLANG_PORT,
        tp=mt.miles_cfg.rollout_num_gpus_per_engine, extra_server_args=mt.SGLANG_SERVER_ARGS,
        health_timeout=mt.SERVER_STARTUP_TIMEOUT, health_poll_interval=10.0,
    )
    endpoint.start()
    R["anchor_coldload_s"] = round(time.perf_counter() - t0, 1)
    print(f"[phase] anchor cold-load: {R['anchor_coldload_s']}s")

    print("[phase] reload anchor via update_weights_from_disk...")
    if alive():
        try:
            R["anchor_reload_s"] = round(asyncio.run(_reload(anchor)), 1)
            R["anchor_reload"] = "OK"
        except Exception as e:  # noqa: BLE001
            R["anchor_reload"] = f"CRASH: {str(e)[:120]}"
            print(f"[phase] anchor reload CRASH: {e}")
    else:
        R["anchor_reload"] = "SKIP (engine not alive after cold-load)"

    print("=== ANCHOR PROBE SUMMARY ===")
    for k, v in R.items():
        print(f"  {k}: {v}")
    try:
        endpoint.stop()
    except Exception:  # noqa: BLE001
        pass
