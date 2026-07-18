"""Reproduce: on Modal, is reading a large file FASTER from a Volume (network, FUSE) than
from the container's EPHEMERAL local disk? (We saw the Volume win ~3-4x for an mmap read
workload, which is backwards — local disk "should" be faster.)

Fully self-contained: generates the test file itself, needs no model checkpoint, and imports
only `modal`. It writes an N-GB random file to a Volume and to the ephemeral scratch, then
times reading each back via (a) mmap page-faults (touch every byte) and (b) sequential
direct-IO `dd`, both COLD (page cache evicted via posix_fadvise) and WARM.

To keep the "cold" numbers honest, every cold read runs in its OWN fresh, single-use
container (`single_use_containers=True`): the file is written in a throwaway container, and
each read mode (mmap / dd direct / dd buffered) is measured in a separate brand-new container
that never touched the file (cold read, then a warm re-read in the same container). Otherwise
one mode's read would leave the file cached and poison the next mode's "cold" number.

Each mode is repeated over `--trials` fresh containers, and the printed table reports the
mean / std / min / max / median (plus the raw per-trial samples) for every metric.

Run (defaults: 32 GB file, 3 trials, no GPU, 512 GB ephemeral scratch):
    modal run repro_volume_vs_ephemeral.py

Match our setup (H200, big ephemeral, bigger file), 5 trials:
    REPRO_GPU=H200 REPRO_EPHEMERAL_GB=800 modal run repro_volume_vs_ephemeral.py --size-gb 100 --trials 5

Read the printed "=== RESULT ===" table.
"""

from __future__ import annotations

import os

import modal

# GPU / ephemeral size are container-shape knobs (set before `modal run`); the effect may
# depend on the host class and the ephemeral size, so match your real workload's shape.
_GPU = os.environ.get("REPRO_GPU") or None
_EPHEMERAL_GB = int(os.environ.get("REPRO_EPHEMERAL_GB", "512"))
# Pin container memory (request == limit) so both the volume and ephemeral sides get an equal
# budget: memory bounds how much of the file stays in page cache for the "warm" read, so it
# must be held equal to compare fairly. Default exceeds the 32 GB file so a warm read caches.
_MEMORY_MB = int(os.environ.get("REPRO_MEMORY_MB", str(64 * 1024)))

app = modal.App("disk-repro")
volume = modal.Volume.from_name("disk-repro-vol", create_if_missing=True, version=2)
image = modal.Image.debian_slim().pip_install("numpy")

VOL_DIR = "/vol"          # Modal Volume (network, FUSE)
EPH_DIR = "/eph-scratch"  # a dir on the container's ephemeral scratch disk


def _evict(path: str) -> None:
    """Drop this file from the page cache (no root needed), so the next read is truly cold."""
    import subprocess

    subprocess.run(["sync"], check=False)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _write_random(path: str, size_gb: int) -> None:
    """Write `size_gb` of incompressible random data in 1 GB chunks (bounded memory)."""
    import numpy as np

    if os.path.exists(path) and os.path.getsize(path) >= size_gb * (1 << 30):
        print(f"[write] reuse existing {path} ({os.path.getsize(path)/1e9:.0f} GB)")
        return
    print(f"[write] generating {size_gb} GB random file at {path} ...")
    chunk = np.random.randint(0, 256, size=1 << 30, dtype=np.uint8)  # 1 GB incompressible
    with open(path, "wb") as f:
        for _ in range(size_gb):
            f.write(chunk.tobytes())
        f.flush()
        os.fsync(f.fileno())


def _read_mmap_gbps(path: str) -> float:
    """Read the whole file via mmap, touching every byte (numpy sum) — the page-fault access
    pattern an mmap-based loader uses. Returns GB/s."""
    import mmap
    import time

    import numpy as np

    fd = os.open(path, os.O_RDONLY)
    try:
        sz = os.fstat(fd).st_size
        mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        t = time.perf_counter()
        int(np.frombuffer(mm, dtype=np.uint8).sum(dtype=np.uint64))  # forces reading all pages
        dt = time.perf_counter() - t
        mm.close()
    finally:
        os.close(fd)
    return round(sz / 1e9 / dt, 2)


def _read_dd_gbps(path: str, *, direct: bool) -> float:
    """Sequential read via dd (optionally O_DIRECT, bypassing the page cache). Returns GB/s."""
    import subprocess
    import time

    sz = os.path.getsize(path)
    args = ["dd", f"if={path}", "of=/dev/null", "bs=16M"] + (["iflag=direct"] if direct else [])
    t = time.perf_counter()
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return round(sz / 1e9 / (time.perf_counter() - t), 2)


def _env_info(name: str, d: str) -> dict:
    """Host RAM + what `d` is actually backed by, from whichever worker this runs on."""
    import subprocess

    R: dict[str, object] = {}
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            R[f"{name}_host_ram"] = f"{int(line.split()[1]) / 1024 / 1024:.0f} GB"
    # Container memory budget from the cgroup -- this is what actually bounds page-cache
    # retention for the warm read, and what we pin equal across both sides.
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            R[f"{name}_mem_limit"] = "max" if v == "max" else f"{int(v) / (1 << 30):.1f} GB"
            break
        except OSError:
            continue
    os.makedirs(d, exist_ok=True)
    R[f"df_{name}"] = subprocess.run(["df", "-hT", d], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return R


# Each mode runs in its own fresh container (genuinely cold), then a warm re-read there.
READ_MODES = ("mmap", "dd_direct", "dd_buffered")


def _measure_mode(name: str, path: str, mode: str) -> dict:
    """Cold (then warm) read for a single mode, run in a container that hasn't read the file yet.
    Returns a dict of `<name>_<mode>_*_gbps` metrics."""
    R: dict[str, object] = {}
    if mode == "mmap":
        _evict(path); R[f"{name}_mmap_cold_gbps"] = _read_mmap_gbps(path)
        R[f"{name}_mmap_warm_gbps"] = _read_mmap_gbps(path)          # now resident -> cache-fast
    elif mode == "dd_direct":
        # O_DIRECT bypasses the page cache, so there is no meaningful warm variant.
        _evict(path); R[f"{name}_dd_direct_gbps"] = _read_dd_gbps(path, direct=True)
    elif mode == "dd_buffered":
        _evict(path); R[f"{name}_dd_buffered_cold_gbps"] = _read_dd_gbps(path, direct=False)
        R[f"{name}_dd_buffered_warm_gbps"] = _read_dd_gbps(path, direct=False)
    else:
        raise ValueError(f"unknown read mode: {mode!r}")
    print(f"[{name}:{mode}] " + " ".join(f"{k.split(name + '_')[1]}={v}" for k, v in R.items()))
    return R


VOL_FILE = f"{VOL_DIR}/bigfile.bin"
EPH_FILE = f"{EPH_DIR}/bigfile.bin"


@app.function(
    image=image,
    volumes={VOL_DIR: volume},
    ephemeral_disk=_EPHEMERAL_GB * 1024,  # MiB
    memory=(_MEMORY_MB, _MEMORY_MB),  # pin request == limit so both sides get equal memory
    timeout=60 * 60,
    single_use_containers=True,  # throwaway container: the writer must NOT be the one that later reads
)
def write_volume(size_gb: int = 32) -> None:
    """Write & commit the volume test file, then let this container be retired. Reading it back
    from a *different* fresh container is what keeps the volume 'cold' reads honest."""
    os.makedirs(VOL_DIR, exist_ok=True)
    _write_random(VOL_FILE, size_gb)
    volume.commit()


@app.function(
    image=image,
    gpu=_GPU,
    volumes={VOL_DIR: volume},
    ephemeral_disk=_EPHEMERAL_GB * 1024,  # MiB
    memory=(_MEMORY_MB, _MEMORY_MB),  # pin request == limit so both sides get equal memory
    timeout=60 * 60,
    single_use_containers=True,  # fresh container per invocation -> a genuinely cold first read
)
def read_volume(size_gb: int = 32, mode: str = "mmap", *, with_env: bool = False) -> dict:
    """Read the already-committed volume file for a single `mode` in a brand-new container.
    Because this container never wrote the file (and is never reused), the cold read really
    hits the network. Called once per mode so no mode pollutes another's cache."""
    volume.reload()
    R = _env_info("volume", VOL_DIR) if with_env else {}
    R.update(_measure_mode("volume", VOL_FILE, mode))
    return R


@app.function(
    image=image,
    gpu=_GPU,
    ephemeral_disk=_EPHEMERAL_GB * 1024,  # MiB
    memory=(_MEMORY_MB, _MEMORY_MB),  # pin request == limit so both sides get equal memory
    timeout=60 * 60,
    single_use_containers=True,  # fresh container per mode: writes the file, then reads it cold
)
def bench_ephemeral(size_gb: int = 32, mode: str = "mmap", *, with_env: bool = False) -> dict:
    """Ephemeral scratch is container-local, so it must be written and read in the same
    container; the cold read relies on posix_fadvise(DONTNEED) + O_DIRECT to shed the pages the
    write just left in cache. Called once per mode, each in its own fresh container."""
    os.makedirs(EPH_DIR, exist_ok=True)
    _write_random(EPH_FILE, size_gb)
    R = _env_info("ephemeral", EPH_DIR) if with_env else {}
    R.update(_measure_mode("ephemeral", EPH_FILE, mode))
    os.remove(EPH_FILE)
    return R


def _stats(xs: list[float]) -> dict:
    """Summary stats for a list of per-trial GB/s samples."""
    import statistics

    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 3),
        "std": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
        "min": round(min(xs), 3),
        "max": round(max(xs), 3),
        "median": round(statistics.median(xs), 3),
        "trials": [round(x, 3) for x in xs],
    }


@app.local_entrypoint()
def main(size_gb: int = 32, trials: int = 3) -> None:
    import json

    meta: dict[str, object] = {
        "gpu": _GPU or "none",
        "ephemeral_gb": _EPHEMERAL_GB,
        "memory_mb": _MEMORY_MB,
        "file_gb": size_gb,
        "trials": trials,
    }
    samples: dict[str, list[float]] = {}  # metric -> one GB/s value per trial

    def _record(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, (int, float)):
                samples.setdefault(k, []).append(float(v))
            else:
                meta.setdefault(k, v)  # host_ram / df_* -- keep first seen (workers may differ)

    # Write the volume file once; every read runs in its own fresh single-use container. Repeat
    # the whole sweep `trials` times so each trial's "cold" number is genuinely cold.
    write_volume.remote(size_gb=size_gb)
    for t in range(trials):
        for i, mode in enumerate(READ_MODES):
            _record(read_volume.remote(size_gb=size_gb, mode=mode, with_env=(t == 0 and i == 0)))
    for t in range(trials):
        for i, mode in enumerate(READ_MODES):
            _record(bench_ephemeral.remote(size_gb=size_gb, mode=mode, with_env=(t == 0 and i == 0)))

    R = {**meta, "stats": {k: _stats(v) for k, v in samples.items()}}
    print(f"=== RESULT (GB/s, stats over {trials} trials) ===")
    print(json.dumps(R, indent=2))
