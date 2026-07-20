"""FP8-base reload-speed probe (v2). Boots ONE sglang engine on the miles_disagg serving
image — same fork pin, server args, and volumes as the Server cls — and times the
weight-load path, separating disk read from SGLang's load processing:

  1. raw cold-read throughput of a base shard off /prep (disk reference),
  2. cold startup load of the served base (the initial-load number),
  3. WARM reload (page cache primed by startup) — isolates dispatch + quant postprocess,
  4. COLD reload (page cache dropped) — the disk-then-load worst case,
  5. a second warm reload — consecutive-reload stability + steady-state timing.

``load_plan=1`` flips SGLANG_ENABLE_RELOAD_LOAD_PLAN on so reload #3 records the dispatch
plan and #4/#5 replay it — the record/replay-vs-native comparison. The engine's
``[reload timing] ... load=..s postprocess=..s`` line lands in the same app logs.

Run (env holds the prepped GLM-4.5-Air FP8 base):
    EXPERIMENT_CONFIG=glm45_air_fp8 PYTHONPATH=. uv run --extra modal \
      modal run -e nan-dev --detach tools/profiling/reload_probe.py::profile_reload
    modal app logs -e nan-dev weight-sync-probes
"""

from __future__ import annotations

import asyncio
import glob
import os
import subprocess
import time

import modal

import cookbook.miles_disagg.app as mt
from cookbook.common.constants import (
    HF_CACHE_PATH, MINUTES, PREP_PATH, SERVER_STARTUP_TIMEOUT, SGLANG_CACHE_PATH, SGLANG_PORT,
)

app = modal.App(os.environ.get("PROBE_APP", "weight-sync-probes"))


def _drop_page_caches() -> bool:
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[diskread] (could not drop page cache: {e})")
        return False


def _evict(paths: list[str]) -> int:
    """Evict these files' pages from the OS page cache via posix_fadvise(DONTNEED) — the
    per-file, no-root way to force a genuinely COLD read (drop_caches needs privilege the
    Modal container lacks). Sync first so freshly-written (dirty) pages are flushable."""
    subprocess.run(["sync"], check=False)
    n = 0
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
        except Exception:  # noqa: BLE001
            pass
    return n


def _dd_read_gbps(path: str, label: str, *, cold: bool = True) -> float:
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


@app.function(
    image=mt.server_image,
    gpu=f"{mt.modal_cfg.gpu}:{mt.miles_cfg.rollout_num_gpus_per_engine}",
    cloud=mt.modal_cfg.cloud,
    region=mt.modal_cfg.region,
    volumes={
        str(HF_CACHE_PATH): mt.hf_cache_volume,
        str(PREP_PATH): mt.prep_volume,
        SGLANG_CACHE_PATH: mt.sglang_cache_volume,
    },
    ephemeral_disk=mt.modal_cfg.rollout_ephemeral_disk_mib,
    memory=mt.modal_cfg.rollout_memory_mib,
    timeout=90 * MINUTES,
)
def profile_reload(load_plan: int = 0, num_threads: int = 0, disable_mmap: int = 0,
                   prefetch: int = 1, fastsafetensors: int = 0, serve_local: int = 0) -> None:
    import json as _json

    import httpx
    from autoinference_utils.endpoint import SGLangEndpoint

    if fastsafetensors:  # not in the base image; install before the engine imports it
        subprocess.run(["uv", "pip", "install", "--system", "fastsafetensors"], check=True)
        # Validate the (unpushed) nogds fork patch in place: GDS/cuFile is absent under gVisor,
        # so force the loader onto the nogds path (O_DIRECT + host bounce) before the engine imports it.
        _wu = "/sgl-workspace/sglang/python/sglang/srt/model_loader/weight_utils.py"
        _src = open(_wu).read()
        assert "SafeTensorsFileLoader(pg, device)" in _src, "loader construction not found to patch"
        open(_wu, "w").write(_src.replace(
            "SafeTensorsFileLoader(pg, device)", "SafeTensorsFileLoader(pg, device, nogds=True)"))
        print("[probe] patched weight_utils.py -> nogds=True")

    # Match the container the sidecar would launch: the config's SGLANG_ENV first, then the
    # probe's explicit load_plan choice wins so both flag states are measurable.
    for k, v in getattr(mt.exp, "SGLANG_ENV", {}).items():
        os.environ[k] = v
    os.environ["SGLANG_ENABLE_RELOAD_LOAD_PLAN"] = str(load_plan)

    # Loader-config A/B (applies to reloads too — the handler forwards model_loader_extra_config).
    server_args = dict(mt.SGLANG_SERVER_ARGS)
    if num_threads:  # else leave the endpoint's default (64)
        # compact separators: the endpoint splits args on spaces, so the JSON must be one token
        server_args["--model-loader-extra-config"] = _json.dumps(
            {"enable_multithread_load": True, "num_threads": num_threads}, separators=(",", ":")
        )
    if disable_mmap:
        server_args["--weight-loader-disable-mmap"] = ""
    if not prefetch:  # on local NVMe the prefetch pass is a redundant read; test with it off
        server_args.pop("--weight-loader-prefetch-checkpoints", None)
        server_args.pop("--weight-loader-prefetch-num-threads", None)
    if fastsafetensors:  # splits files across ranks (each reads ~1/N) + O_DIRECT disk->GPU
        server_args["--load-format"] = "fastsafetensors"
    R_fmt = "fastsafetensors" if fastsafetensors else ("no-mmap" if disable_mmap else "mmap")

    base = mt.miles_cfg.hf_checkpoint  # /prep/<tag>/fp8 — the served FP8 base (Modal Volume)
    local = mt.exp.LOCAL_CHECKPOINT_PATH  # ephemeral NVMe — the disk production reloads from
    url = f"http://127.0.0.1:{SGLANG_PORT}"
    R: dict[str, object] = {"base": base, "load_plan": load_plan, "fmt": R_fmt,
                            "num_threads": num_threads or 64, "disable_mmap": bool(disable_mmap),
                            "prefetch": bool(prefetch)}

    def base_shards():
        return sorted(glob.glob(f"{base}/*.safetensors"))

    def alive() -> bool:
        try:
            return httpx.get(f"{url}/health", timeout=10.0).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    async def _reload(model_path: str) -> float:
        t = time.perf_counter()
        async with httpx.AsyncClient(timeout=None, trust_env=False) as c:
            r = await c.post(f"{url}/update_weights_from_disk",
                             json={"model_path": model_path, "weight_version": "probe", "flush_cache": False})
            r.raise_for_status()
            body = r.json()
            if body.get("success") is False:
                raise RuntimeError(f"reject: {body}")
            print(f"[reload message] {body.get('message')}")
        return time.perf_counter() - t

    def reload_phase(key: str, model_path: str, *, cold: bool) -> None:
        if not alive():
            R[key] = "SKIP (engine not alive)"
            return
        if cold:  # genuinely evict this checkpoint's shards so the reload reads from disk
            R[f"{key}_evicted_files"] = _evict(sorted(glob.glob(f"{model_path}/*.safetensors")))
        try:
            R[key] = round(asyncio.run(_reload(model_path)), 1)
        except Exception as e:  # noqa: BLE001
            R[key] = f"CRASH: {str(e)[:120]}"
            print(f"[phase] {key} CRASH: {e}")

    print(f"=== FP8 RELOAD PROBE — base={base} tp={mt.miles_cfg.rollout_num_gpus_per_engine} "
          f"load_plan={load_plan} ===")
    shards = base_shards()
    R["base_shards"] = len(shards)
    R["base_total_gb"] = round(sum(os.path.getsize(p) for p in shards) / 1e9, 1)

    # 1. raw cold read of one base shard (disk-bound reference for the reload numbers)
    if shards:
        R["dd_prep_cold_gbps"] = round(_dd_read_gbps(shards[0], "prep base shard (cold)"), 2)

    # 2. optionally pre-seed /local and serve FROM it: fastsafetensors can't O_DIRECT-open the
    #    Volume, so to test it at all we must serve + reload entirely from the NVMe.
    serve_from = base
    if fastsafetensors or serve_local:
        print("[phase] pre-seed /local from /prep base (serving from NVMe)...")
        os.makedirs(local, exist_ok=True)
        t0m = time.perf_counter()
        subprocess.run(f"cp -aL {base}/. {local}/", shell=True, check=True)
        R["materialize_s"] = round(time.perf_counter() - t0m, 1)
        serve_from = local

    print(f"[phase] SGLang startup (cold load from {serve_from})...")
    t0 = time.perf_counter()
    endpoint = SGLangEndpoint(
        model_path=serve_from, worker_port=SGLANG_PORT,
        tp=mt.miles_cfg.rollout_num_gpus_per_engine, extra_server_args=server_args,
        health_timeout=SERVER_STARTUP_TIMEOUT, health_poll_interval=10.0,
    )
    endpoint.start()
    R["startup_coldload_s"] = round(time.perf_counter() - t0, 1)
    print(f"[phase] startup cold load: {R['startup_coldload_s']}s")

    if serve_from == base:
        # control: reload from /prep (Volume, cache-hot from startup), then materialize -> /local
        print("[phase] reload from /prep, warm...")
        reload_phase("reload_prep_warm_s", base, cold=False)
        print("[phase] reload from /prep, cold...")
        reload_phase("reload_prep_cold_s", base, cold=True)
        print("[phase] materialize /prep base -> /local-checkpoint (NVMe)...")
        os.makedirs(local, exist_ok=True)
        t0m = time.perf_counter()
        subprocess.run(f"cp -aL {base}/. {local}/", shell=True, check=True)  # -L: real files
        R["materialize_s"] = round(time.perf_counter() - t0m, 1)

    # reload from /local NVMe — the production steady-state path (and where fastsafetensors runs)
    print("[phase] reload from /local NVMe, warm (production steady state)...")
    reload_phase("reload_local_warm_s", local, cold=False)
    print("[phase] reload from /local NVMe, cold (fadvise-evicted)...")
    reload_phase("reload_local_cold_s", local, cold=True)
    print("[phase] reload from /local NVMe again (stability)...")
    reload_phase("reload_local_warm2_s", local, cold=False)

    print("=== PROBE SUMMARY ===")
    for k, v in R.items():
        print(f"  {k}: {v}")
    try:
        endpoint.stop()
    except Exception:  # noqa: BLE001
        pass
