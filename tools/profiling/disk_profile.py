"""Ground-truth ephemeral-disk profile for the rollout pool's container shape.

The steady-state reload reads the checkpoint from /local-checkpoint (the container's
EPHEMERAL scratch, sized by ephemeral_disk), not from the /prep Volume. A reload was
observed crawling at ~0.3 GB/s, which is not NVMe speed — so this measures the raw disk
directly (direct-IO to bypass the page cache, which the container can't drop) and checks
what /local is actually backed by. Matches the pool's ephemeral_disk size, since a large
scratch request may be provisioned on network storage rather than host-local NVMe.

Run:
    EXPERIMENT_CONFIG=glm45_air_fp8 PYTHONPATH=. uv run --extra modal \
      modal run -e nan-dev tools/profiling/disk_profile.py::disk_profile
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import time

import modal

import cookbook.miles_disagg.app as mt
from cookbook.common.constants import MINUTES, PREP_PATH

app = modal.App(os.environ.get("PROBE_APP", "disk-profile"))


def _dd(label: str, args: list[str], nbytes: int, out: dict) -> None:
    t0 = time.perf_counter()
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gbps = nbytes / 1e9 / (time.perf_counter() - t0)
    out[label] = round(gbps, 2)
    print(f"[disk] {label}: {gbps:.2f} GB/s")


@app.function(
    image=mt.server_image,
    gpu=f"{mt.modal_cfg.gpu}:1",  # land on the H200 host class (same local disk as the pool), cheaper than tp4
    cloud=mt.modal_cfg.cloud,
    region=mt.modal_cfg.region,
    volumes={str(PREP_PATH): mt.prep_volume},
    ephemeral_disk=mt.modal_cfg.rollout_ephemeral_disk_mib,  # SAME scratch size as the pool
    memory=mt.modal_cfg.rollout_memory_mib,
    timeout=25 * MINUTES,
)
def disk_profile(write_gb: int = 32) -> dict:
    local = mt.exp.LOCAL_CHECKPOINT_PATH
    os.makedirs(local, exist_ok=True)
    R: dict[str, object] = {
        "placement": {k: v for k, v in os.environ.items() if k.startswith("MODAL_") and len(v) < 80},
    }

    # What is /local actually backed by? (local NVMe vs network scratch)
    for label, path in [("root_/", "/"), ("local_scratch", local)]:
        try:
            df = subprocess.run(["df", "-hT", path], capture_output=True, text=True).stdout.strip().splitlines()
            R[f"df_{label}"] = df[-1] if df else "?"
        except Exception as e:  # noqa: BLE001
            R[f"df_{label}"] = f"err: {e}"
    R["nproc"] = os.cpu_count()

    blocks = max(1, (write_gb << 30) // (16 << 20))
    nbytes = blocks * (16 << 20)
    path = f"{local}/_probe.bin"

    # Direct-IO write/read bypass the page cache -> the disk's true throughput.
    _dd("write_direct_gbps", ["dd", "if=/dev/zero", f"of={path}", "bs=16M", f"count={blocks}",
                              "oflag=direct", "conv=fsync"], nbytes, R)
    _dd("read_direct_gbps", ["dd", f"if={path}", "of=/dev/null", "bs=16M", "iflag=direct"], nbytes, R)
    _dd("read_buffered_gbps", ["dd", f"if={path}", "of=/dev/null", "bs=16M"], nbytes, R)
    _dd("read_cached_gbps", ["dd", f"if={path}", "of=/dev/null", "bs=16M"], nbytes, R)  # 2nd read = page cache
    os.remove(path)

    # A REAL safetensors shard: copy one base shard /prep -> /local, then read it back
    # direct-IO (true cold) and buffered. This is the exact bytes a reload reads.
    shards = sorted(glob.glob(f"{mt.miles_cfg.hf_checkpoint}/*.safetensors"))
    if shards:
        src = shards[0]
        dst = f"{local}/_shard.safetensors"
        sz = os.path.getsize(src)
        t0 = time.perf_counter()
        subprocess.run(["cp", src, dst], check=True)
        R["shard_copy_prep_to_local_gbps"] = round(sz / 1e9 / (time.perf_counter() - t0), 2)
        _dd("shard_read_local_direct_gbps", ["dd", f"if={dst}", "of=/dev/null", "bs=16M", "iflag=direct"], sz, R)
        _dd("shard_read_local_buffered_gbps", ["dd", f"if={dst}", "of=/dev/null", "bs=16M"], sz, R)
        # /prep Volume direct read, for comparison (the boot-load disk)
        _dd("shard_read_prep_direct_gbps", ["dd", f"if={src}", "of=/dev/null", "bs=16M", "iflag=direct"], sz, R)
        os.remove(dst)

    print("=== DISK PROFILE ===")
    print(json.dumps(R, indent=2))
    return R


def _evict(paths: list[str]) -> None:
    subprocess.run(["sync"], check=False)
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except Exception:  # noqa: BLE001
            pass


@app.function(
    image=mt.server_image,
    gpu=f"{mt.modal_cfg.gpu}:1",
    cloud=mt.modal_cfg.cloud,
    region=mt.modal_cfg.region,
    volumes={str(PREP_PATH): mt.prep_volume},
    ephemeral_disk=mt.modal_cfg.rollout_ephemeral_disk_mib,
    memory=mt.modal_cfg.rollout_memory_mib,
    timeout=25 * MINUTES,
)
def cache_probe() -> dict:
    """Answer 'why does /prep re-read hit cache but /local not?' with the loader's OWN access
    pattern: safetensors mmap (get_tensor) and plain sequential read, each COLD (fadvise-
    evicted) then WARM (immediate re-read), on a real shard on /prep (Volume/FUSE) vs /local
    (ephemeral NVMe). If warm>>cold the filesystem caches across reads; if warm≈cold it does
    not. Plus host RAM (does the 112GB checkpoint even fit in page cache?)."""
    import glob as _glob
    import time as _time

    import safetensors.torch

    R: dict[str, object] = {}
    mi = {}
    for line in open("/proc/meminfo"):
        if line.startswith(("MemTotal:", "MemAvailable:")):
            k, v = line.split(":")
            mi[k] = f"{int(v.split()[0]) / 1024 / 1024:.0f} GB"
    R["host_mem"] = mi

    prep_shard = sorted(_glob.glob(f"{mt.miles_cfg.hf_checkpoint}/*.safetensors"))[0]
    local_shard = f"{mt.exp.LOCAL_CHECKPOINT_PATH}/_shard.safetensors"
    os.makedirs(mt.exp.LOCAL_CHECKPOINT_PATH, exist_ok=True)
    subprocess.run(["cp", prep_shard, local_shard], check=True)
    sz = os.path.getsize(prep_shard)

    def read_mmap(path: str) -> float:  # the loader's default path: safe_open + get_tensor
        t = _time.perf_counter()
        with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                f.get_tensor(k)
        return sz / 1e9 / (_time.perf_counter() - t)

    def read_seq(path: str) -> float:  # the disable_mmap path: plain sequential read
        t = _time.perf_counter()
        with open(path, "rb") as f:
            while f.read(16 << 20):
                pass
        return sz / 1e9 / (_time.perf_counter() - t)

    for label, path in [("prep_volume", prep_shard), ("local_nvme", local_shard)]:
        _evict([path]); R[f"{label}_mmap_cold_gbps"] = round(read_mmap(path), 2)
        R[f"{label}_mmap_warm_gbps"] = round(read_mmap(path), 2)  # no evict -> tests caching
        _evict([path]); R[f"{label}_seq_cold_gbps"] = round(read_seq(path), 2)
        R[f"{label}_seq_warm_gbps"] = round(read_seq(path), 2)
        print(f"[cache] {label}: mmap cold={R[f'{label}_mmap_cold_gbps']} warm={R[f'{label}_mmap_warm_gbps']} "
              f"| seq cold={R[f'{label}_seq_cold_gbps']} warm={R[f'{label}_seq_warm_gbps']} GB/s")
    os.remove(local_shard)
    print("=== CACHE PROBE ===")
    print(json.dumps(R, indent=2))
    return R


@app.function(
    image=mt.server_image,
    gpu=f"{mt.modal_cfg.gpu}:1",
    cloud=mt.modal_cfg.cloud,
    region=mt.modal_cfg.region,
    volumes={str(PREP_PATH): mt.prep_volume},
    ephemeral_disk=mt.modal_cfg.rollout_ephemeral_disk_mib,
    memory=mt.modal_cfg.rollout_memory_mib,
    timeout=25 * MINUTES,
)
def cache_probe2() -> dict:
    """Confirm whether /local (ephemeral) retains a page cache across mmap reads, the way the
    loader faults pages. Method: mmap the file and touch one byte per 4KB page (real faults,
    in C via a strided numpy sum) — COLD (fadvise-evicted) then WARM (immediate re-read). If
    warm >> cold it caches (RAM-speed on repeat); if warm ≈ cold it always hits disk. A tmpfs
    (/dev/shm) file is the control — RAM-backed, so it must show RAM-speed both times."""
    import mmap as _mmap
    import time as _time

    import numpy as np

    R: dict[str, object] = {}
    mi = {}
    for line in open("/proc/meminfo"):
        if line.startswith(("MemTotal:", "MemAvailable:", "Cached:")):
            k, v = line.split(":")
            mi[k] = f"{int(v.split()[0]) / 1024 / 1024:.0f} GB"
    R["host_mem"] = mi
    R["df_shm"] = subprocess.run(["df", "-hT", "/dev/shm"], capture_output=True, text=True).stdout.strip().splitlines()[-1]

    prep_shard = sorted(glob.glob(f"{mt.miles_cfg.hf_checkpoint}/*.safetensors"))[0]
    local_shard = f"{mt.exp.LOCAL_CHECKPOINT_PATH}/_c.safetensors"
    shm_shard = "/dev/shm/_c.safetensors"
    os.makedirs(mt.exp.LOCAL_CHECKPOINT_PATH, exist_ok=True)
    subprocess.run(["cp", prep_shard, local_shard], check=True)
    subprocess.run(["cp", prep_shard, shm_shard], check=True)

    def mmap_touch(path: str) -> float:  # fault one byte per 4KB page — the loader's access pattern
        fd = os.open(path, os.O_RDONLY)
        sz = os.fstat(fd).st_size
        mm = _mmap.mmap(fd, 0, prot=_mmap.PROT_READ)
        t = _time.perf_counter()
        int(np.frombuffer(mm, dtype=np.uint8)[::4096].sum(dtype=np.int64))
        dt = _time.perf_counter() - t
        mm.close()
        os.close(fd)
        return round(sz / 1e9 / dt, 2)

    for label, path in [("prep_volume", prep_shard), ("local_nvme", local_shard), ("tmpfs_control", shm_shard)]:
        _evict([path])
        cold = mmap_touch(path)
        warm = mmap_touch(path)  # no evict — RAM-speed here means it cached
        R[f"{label}_mmap_cold_gbps"], R[f"{label}_mmap_warm_gbps"] = cold, warm
        print(f"[cache2] {label}: mmap cold={cold} warm={warm} GB/s  (warm>>cold => caches)")

    os.remove(local_shard)
    os.remove(shm_shard)
    print("=== CACHE PROBE 2 ===")
    print(json.dumps(R, indent=2))
    return R
