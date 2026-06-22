"""TEMP profiling probe: why is the SGLang weight reload so slow?

Brings up ONE rollout engine identical to the miles_disagg Server cls (same
serving image, same SGLangEndpoint(model_path=NVFP4 base, tp, SGLANG_SERVER_ARGS),
same volumes + ephemeral disk), then times each phase of the weight-update path
and isolates disk-read throughput from SGLang's load processing:

  1. raw read throughput of the base shards on the /prep Volume,
  2. SGLang cold startup load (reads the base from /prep),
  3. parallel materialize /prep base -> /local-checkpoint (ephemeral NVMe),
  4. raw read throughput of /local-checkpoint shards (ephemeral, cold),
  5. CLEAN reload of the base from /local-checkpoint via update_weights_from_disk
     (no delta -> won't hit the merged-column crash; pure load-perf number),
  6. apply the existing delta (18ec126bbabc/weight_v000001) + reload (the real
     scenario; may crash at the merged-column param — caught + timed).

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

DELTA_RUN_ID = "18ec126bbabc"
DELTA_VERSION = 1


def _dd_read_gbps(path: str, label: str) -> float:
    """Cold-read a file with dd (drop page cache first); return GB/s."""
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3")
    except Exception as e:  # noqa: BLE001
        print(f"[diskread] (could not drop page cache: {e})")
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
def profile_reload(delta_run_id: str = DELTA_RUN_ID, version: int = DELTA_VERSION) -> None:
    import asyncio

    import httpx
    from autoinference_utils.endpoint import SGLangEndpoint

    from miles.utils.disk_delta import apply_deltas
    from stitch.engines.sglang import SGLangDiskDeltaAdapter
    from stitch.protocol import VersionManifest
    from cookbook.miles_disagg.sidecar import _parallel_init_local_checkpoint

    base = mt.MODEL_NAME  # /prep/kimi-k2-6-nvfp4/nvfp4
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
            if r.json().get("success") is False:
                raise RuntimeError(f"reject: {r.json()}")
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

    # 3. KEY TEST 1 — reload /prep DIRECTLY via update_weights_from_disk (the reload path,
    #    complete base, no materialize). Crash here => SGLang reload-path bug (cold-load vs
    #    reload divergence). Success => the crash is materialize/local-checkpoint specific.
    print("[phase] KEY TEST 1: reload /prep directly via update_weights_from_disk...")
    if alive():
        try:
            R["reload_prep_s"] = round(asyncio.run(_reload(base)), 1)
            R["reload_prep"] = "OK"
            print(f"[phase] reload /prep: OK {R['reload_prep_s']}s")
        except Exception as e:  # noqa: BLE001
            R["reload_prep"] = f"CRASH: {str(e)[:120]}"
            print(f"[phase] reload /prep CRASH: {e}")
    else:
        R["reload_prep"] = "SKIP (engine not alive)"

    # 4. materialize + KEY TEST 2 — only if /prep reload didn't kill the engine
    if alive():
        print("[phase] materialize base -> /local-checkpoint (32 workers)...")
        t0 = time.perf_counter()
        _parallel_init_local_checkpoint()(local, base)
        R["materialize_s"] = round(time.perf_counter() - t0, 1)
        print(f"[phase] materialize: {R['materialize_s']}s")
        local_shards = sorted(glob.glob(f"{local}/*.safetensors"))
        if local_shards:
            R["dd_local_gbps"] = round(_dd_read_gbps(local_shards[0], "ephemeral /local-checkpoint shard (cold)"), 2)
        print("[phase] KEY TEST 2: reload /local-checkpoint via update_weights_from_disk...")
        if alive():
            try:
                R["reload_local_s"] = round(asyncio.run(_reload(local)), 1)
                R["reload_local"] = "OK"
            except Exception as e:  # noqa: BLE001
                R["reload_local"] = f"CRASH: {str(e)[:120]}"
                print(f"[phase] reload /local CRASH: {e}")
    else:
        R["materialize_s"] = R["reload_local"] = "SKIP (engine died on /prep reload)"

    print("=== PROBE SUMMARY ===")
    for k, v in R.items():
        print(f"  {k}: {v}")
    try:
        endpoint.stop()
    except Exception:  # noqa: BLE001
        pass


@probe_app.function(
    image=mt.server_image,
    volumes={
        str(mt.PREP_PATH): mt.prep_volume,
        mt.exp.DELTA_BULLETIN_ROOT: mt.delta_volume,
    },
    timeout=15 * mt.MINUTES,
)
def rca_checksum(run_id: str = "ece3aa5c17c0") -> None:
    """CPU RCA: for a few of the checksum-mismatched tensors, reconstruct base XOR delta
    against the CURRENT /prep base and compare to the delta's stored `want` checksum.
      match  => base is consistent with the delta => the SIDECAR's base_local was different
                (volume skew / stale materialize) -> the pool diffed against a different base.
      mismatch => the CURRENT /prep base != the base the trainer snapshotted (trainer-side skew).
    """
    import glob, io, json, struct
    import numpy as np
    import zstandard
    from miles.utils.disk_delta import checksum, make_tensor_reader

    base = mt.MODEL_NAME
    vdir = f"{mt.exp.DELTA_BULLETIN_ROOT}/{run_id}/weight_v000001"
    reader = make_tensor_reader(base)
    meta = json.load(open(f"{vdir}/model.safetensors.index.json"))["metadata"]
    algo = meta["checksum_format"]
    print(f"=== RCA checksum: base={base} delta={vdir} algo={algo} enc={meta['delta_encoding']} ===")

    probe = ["language_model.lm_head.weight", "language_model.model.embed_tokens.weight",
             "language_model.model.layers.3.mlp.experts.0.gate_proj.weight",
             "language_model.model.layers.3.self_attn.o_proj.weight"]
    checked = 0
    for df in sorted(glob.glob(f"{vdir}/*.safetensors")):
        blob = open(df, "rb").read()
        (hlen,) = struct.unpack("<Q", blob[:8])
        header = json.loads(blob[8:8 + hlen]); want = header.get("__metadata__", {})
        for name, info in header.items():
            if name == "__metadata__" or name not in probe:
                continue
            b, e = info["data_offsets"]; ds = 8 + hlen
            diff = np.frombuffer(zstandard.ZstdDecompressor().decompress(blob[ds + b: ds + e]), dtype=np.uint8)
            try:
                base_bytes = reader(name)
            except KeyError:
                print(f"  {name[:60]}: ABSENT from base"); continue
            recon = base_bytes.copy()
            n = min(recon.size, diff.size)
            recon[:n] ^= diff[:n]
            got = checksum(algo, recon)
            ok = got == want.get(name)
            print(f"  {name.split('language_model.')[-1][:50]:50s} base_nbytes={base_bytes.size} diff_nbytes={diff.size} "
                  f"recon==want? {ok}")
            checked += 1
    print(f"checked {checked} tensors")


@probe_app.function(
    image=mt.server_image,
    volumes={
        str(mt.HF_CACHE_PATH): mt.hf_cache_volume,
        str(mt.PREP_PATH): mt.prep_volume,
    },
    timeout=15 * mt.MINUTES,
)
def diff_shapes() -> None:
    """CPU: compare NVFP4 tensor SHAPES (weight + scales) of a routed expert and a
    merged-column param between OUR /prep base and the nvidia anchor — to pinpoint
    the layout divergence the reload's fused-MoE loader rejects."""
    import json
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    ours = mt.MODEL_NAME
    anchor = snapshot_download(mt.exp.ANCHOR_MODEL, local_files_only=True)

    def shapes_for(root: str, names: list[str]) -> dict:
        idx = json.load(open(f"{root}/model.safetensors.index.json")).get("weight_map", {})
        out = {}
        for n in names:
            f = idx.get(n)
            if not f:
                out[n] = "ABSENT"; continue
            with safe_open(f"{root}/{f}", framework="numpy") as sf:
                sl = sf.get_slice(n)
                out[n] = f"{sl.get_shape()} {sl.get_dtype()}"
        return out

    probe_names = []
    for base_mod in ["language_model.model.layers.5.mlp.experts.0.gate_proj",
                     "language_model.model.layers.5.mlp.experts.0.down_proj"]:
        for suf in [".weight", ".weight_scale", ".weight_scale_2", ".input_scale"]:
            probe_names.append(base_mod + suf)
    o = shapes_for(ours, probe_names)
    a = shapes_for(anchor, probe_names)
    print("=== ROUTED-EXPERT NVFP4 SHAPE DIFF (ours vs anchor) ===")
    for n in probe_names:
        mark = "" if o[n] == a[n] else "   <<< DIFFERS"
        print(f"  {n.split('experts.0.')[-1]:28s} ours={o[n]:22s} anchor={a[n]:22s}{mark}")


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
def profile_anchor_reload() -> None:
    """DECISIVE test: is the crash SGLang's reload path (any checkpoint) or OUR
    prepare_checkpoints NVFP4 output specifically?

    Cold-load the KNOWN-GOOD reference nvidia/Kimi-K2.6-NVFP4 (the design anchor),
    then reload IT via update_weights_from_disk:
      anchor reload CRASH  -> SGLang reload-path bug (prepare_checkpoints innocent)
      anchor reload OK      -> our /prep convert_hf_to_nvfp4 output is the culprit
                               (probe v1/v2 showed our /prep reload crashes); the
                               bf16 carve-out (layers 0 & 60, no scales) is the suspect.
    """
    import asyncio

    import httpx
    from huggingface_hub import snapshot_download
    from autoinference_utils.endpoint import SGLangEndpoint

    anchor = snapshot_download(mt.exp.ANCHOR_MODEL, local_files_only=True)
    sglang_url = f"http://127.0.0.1:{mt.SGLANG_PORT}"
    R: dict[str, object] = {"anchor_path": anchor}
    print(f"=== ANCHOR PROBE — {mt.exp.ANCHOR_MODEL} @ {anchor} ===")

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
            if r.json().get("success") is False:
                raise RuntimeError(f"reject: {r.json()}")
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

    print("[phase] KEY: reload anchor via update_weights_from_disk...")
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
