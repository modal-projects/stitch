"""O_DIRECT / fastsafetensors capability probe on Modal ephemeral disk (CPU-only, cheap).

Root-causes why `--load-format fastsafetensors` hit "Error opening file" on the reload path,
now that Modal reports `dd iflag=direct = 11.74 GB/s` on ephemeral (so O_DIRECT the syscall
clearly works). Three questions the Modal team would need answered:

  1. What filesystem does the ephemeral checkpoint dir actually sit on (NVMe vs overlay)?
  2. Does gVisor honor a *Python* O_DIRECT open + aligned read there (not just `dd`)?
  3. Is the GDS/cuFile stack fastsafetensors defaults to even present under gVisor?

CPU-only on purpose: the "Error opening file" is at open() time (filesystem-level), so this
reproduces it without a GPU and dodges the B200/H200 capacity wall. Throughput here is
indicative only (CPU-worker local disk may differ from a GPU worker's NVMe).

    PROBE_APP=odirect-probe uv run --extra modal modal run -e nan-dev \
      tools/profiling/odirect_probe.py::probe
"""

from __future__ import annotations

import os
import subprocess
import traceback

import modal

TEST_DIR = "/local-checkpoint"  # match the GLM config's LOCAL_CHECKPOINT_PATH
EPHEMERAL_MIB = 524_288  # 512 GiB (Modal's floor for ephemeral_disk)
FILE_BYTES = 4 * 1024**3  # 4 GiB, a multiple of the 64 MiB read chunk (clean O_DIRECT tail)
CHUNK = 64 * 1024**2

image = (
    modal.Image.debian_slim()
    .apt_install("coreutils", "kmod")
    .pip_install("fastsafetensors", "torch", "numpy", "safetensors")
)
app = modal.App(os.environ.get("PROBE_APP", "odirect-probe"))
_GPU = os.environ.get("PROBE_GPU") or None  # e.g. "H200:1"; unset = CPU worker


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(cmd failed: {e})"


@app.function(image=image, gpu=_GPU, ephemeral_disk=EPHEMERAL_MIB, timeout=20 * 60)
def probe() -> None:
    os.makedirs(TEST_DIR, exist_ok=True)
    path = os.path.join(TEST_DIR, "probe.bin")

    print(f"WORKER GPU = {_GPU or 'CPU-only'}")
    print("=" * 70)
    print("Q1. MOUNT + FILESYSTEM — find where the fast disk (if any) is")
    print("=" * 70)
    print("stat -f", TEST_DIR, ":", _sh(f"stat -f -c 'fstype=%T bsize=%s' {TEST_DIR}"))
    print("all mounts:\n" + _sh("cat /proc/mounts"))
    print("df -hT:\n" + _sh("df -hT"))
    # dd iflag=direct on candidate scratch paths: which, if any, hits the ~11.74 GB/s NVMe?
    for p in (TEST_DIR, "/tmp", "/root", "/dev/shm"):
        try:
            os.makedirs(p, exist_ok=True)
            tf = os.path.join(p, "_ddprobe.bin")
            _sh(f"dd if=/dev/zero of={tf} bs=64M count=32 conv=fsync 2>/dev/null")
            print(f"dd direct read {p}:", _sh(f"dd if={tf} of=/dev/null bs=64M iflag=direct 2>&1 | tail -1"))
            _sh(f"rm -f {tf}")
        except Exception as e:  # noqa: BLE001
            print(f"dd direct read {p}: (skip: {e})")

    print("\n" + "=" * 70)
    print("Q2. PLAIN O_DIRECT open + aligned read (Python syscall path)")
    print("=" * 70)
    import time

    # Write the test file (buffered), then evict its pages so the read is genuinely cold.
    with open(path, "wb") as f:
        buf = os.urandom(CHUNK)
        for _ in range(FILE_BYTES // CHUNK):
            f.write(buf)
    subprocess.run(["sync"], check=False)
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception as e:  # noqa: BLE001
        print("(evict failed:", e, ")")

    # O_DIRECT read into a page-aligned buffer (mmap anon is page-aligned; CHUNK is a 4K multiple).
    import mmap as _mmap

    try:
        odirect = getattr(os, "O_DIRECT", None)
        if odirect is None:
            print("RESULT: os.O_DIRECT not defined in this Python build")
        else:
            aligned = _mmap.mmap(-1, CHUNK)
            fd = os.open(path, os.O_RDONLY | odirect)
            t0 = time.perf_counter()
            total = 0
            while True:
                n = os.readv(fd, [aligned])
                if not n:
                    break
                total += n
            dt = time.perf_counter() - t0
            os.close(fd)
            print(f"RESULT: O_DIRECT read OK — {total/1e9:.2f} GB in {dt:.1f}s = {total/1e9/dt:.2f} GB/s")
    except OSError as e:
        print(f"RESULT: O_DIRECT read FAILED — errno={e.errno} ({os.strerror(e.errno) if e.errno else '?'}): {e}")
    except Exception as e:  # noqa: BLE001
        print(f"RESULT: O_DIRECT read FAILED (non-OSError): {e!r}")

    # dd reference (Modal's own benchmark used this; expect it to work if O_DIRECT is honored).
    print("dd iflag=direct:", _sh(f"dd if={path} of=/dev/null bs=64M iflag=direct 2>&1 | tail -1"))

    print("\n" + "=" * 70)
    print("Q3. GDS / cuFile stack presence (fastsafetensors default path)")
    print("=" * 70)
    print("/dev/nvidia-fs*:", _sh("ls -la /dev/nvidia-fs* 2>&1") or "(none)")
    print("/dev/nvidia*:", _sh("ls /dev/nvidia* 2>&1") or "(none)")
    print("nvidia_fs module:", _sh("lsmod 2>/dev/null | grep -i nvidia_fs") or "(not loaded / lsmod unavailable)")
    print("cufile.json:", _sh("ls -la /etc/cufile.json /usr/local/cuda/gds 2>&1") or "(none)")
    print("libcufile:", _sh("find / -name 'libcufile*' 2>/dev/null | head -3") or "(none)")

    print("\n" + "=" * 70)
    print("Q4. fastsafetensors library: import + open (nogds, CPU)")
    print("=" * 70)
    # A tiny real safetensors file so the loader has valid headers to open.
    sft = os.path.join(TEST_DIR, "probe.safetensors")
    try:
        import torch
        from safetensors.torch import save_file

        save_file({"w": torch.zeros(1024, 1024, dtype=torch.float32)}, sft)
        print("wrote test safetensors:", sft, os.path.getsize(sft), "bytes")
    except Exception as e:  # noqa: BLE001
        print("(could not write test safetensors:", e, ")")

    try:
        import fastsafetensors

        print("fastsafetensors version:", getattr(fastsafetensors, "__version__", "?"))
        from fastsafetensors import SafeTensorsFileLoader, SingleGroup

        # nogds=True → O_DIRECT + CPU bounce buffer (no GDS driver needed). This is the mode
        # that SHOULD work on gVisor if plain O_DIRECT is honored. Report the exact failure.
        loader = SafeTensorsFileLoader(SingleGroup(), device="cpu", nogds=True, debug_log=True)
        loader.add_filenames({0: [sft]})
        fb = loader.copy_files_to_device()
        keys = list(fb.key_to_rank_lidx.keys()) if hasattr(fb, "key_to_rank_lidx") else "?"
        print(f"RESULT: fastsafetensors nogds open OK — tensors={keys}")
        fb.close()
        loader.close()
    except Exception as e:  # noqa: BLE001
        print("RESULT: fastsafetensors nogds FAILED:")
        traceback.print_exc()

    print("\n=== PROBE DONE ===")
