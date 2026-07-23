"""Measure the physical CPU-to-GPU commit ceiling on a four-B300 rollout host.

This probe intentionally does not start SGLang or read a checkpoint.  It answers the
first-principles question underneath a general full-state hot reload: how quickly can
one complete rank-local runtime image move from prepared host memory into four GPUs at
the same time?

Each worker:

* binds itself to the NUMA node local to one GPU before allocating host memory;
* allocates and touches a distinct pinned host working set (150 GB by default);
* streams the whole working set through a reusable GPU window; and
* reports CUDA-event and end-to-end wall bandwidth for several copy granularities.

The reusable destination is deliberate.  It makes the source working set realistic
without requiring a second model-sized HBM allocation.  Reusing a destination does not
reduce traffic over PCIe: every source byte is still transferred and every asynchronous
copy is ordered on the worker's CUDA stream.

Run::

    uv run --extra modal modal run \
      tools/profiling/b300_h2d_physics.py
"""

from __future__ import annotations

import ctypes
import concurrent.futures
import json
import multiprocessing as mp
import os
import queue
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import modal


APP_NAME = "b300-h2d-physics"
GPU_COUNT = 4
DEFAULT_WORKING_SET_GB = 64
DEFAULT_WINDOW_GB = 4
DEFAULT_CHUNK_MIB = "16,64,256,1024"
GIB = 1 << 30
MIB = 1 << 20


app = modal.App(APP_NAME)
image = modal.Image.from_registry("lmsysorg/sglang:v0.5.15.post1").pip_install(
    "nvidia-ml-py"
)


def _command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"command": command, "error": repr(exc)}


def _parse_cpu_list(value: str) -> list[int]:
    cpus: list[int] = []
    for part in value.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _gpu_topology(gpu_index: int) -> dict[str, Any]:
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    pci = pynvml.nvmlDeviceGetPciInfo(handle)
    bus_id = pci.busId
    if isinstance(bus_id, bytes):
        bus_id = bus_id.decode()
    bus_id = str(bus_id).lower()
    sysfs = Path("/sys/bus/pci/devices") / bus_id
    numa_node = -1
    local_cpus: list[int] = []
    try:
        numa_node = int((sysfs / "numa_node").read_text().strip())
    except (FileNotFoundError, ValueError):
        pass
    try:
        local_cpus = _parse_cpu_list((sysfs / "local_cpulist").read_text())
    except FileNotFoundError:
        pass
    return {
        "gpu_index": gpu_index,
        "name": pynvml.nvmlDeviceGetName(handle),
        "pci_bus_id": bus_id,
        "numa_node": numa_node,
        "local_cpus": local_cpus,
    }


def _bind_numa(topology: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested_node": topology["numa_node"],
        "requested_cpus": topology["local_cpus"],
    }
    cpus = topology["local_cpus"]
    if cpus:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError as exc:
            result["affinity_error"] = repr(exc)
    try:
        result["effective_cpus"] = sorted(os.sched_getaffinity(0))
    except OSError as exc:
        result["effective_affinity_error"] = repr(exc)

    node = topology["numa_node"]
    if node >= 0:
        try:
            libnuma = ctypes.CDLL("libnuma.so.1")
            if libnuma.numa_available() >= 0:
                rc = libnuma.numa_run_on_node(node)
                libnuma.numa_set_preferred(node)
                result["numa_run_on_node_rc"] = rc
        except Exception as exc:
            result["numa_library_error"] = repr(exc)
    return result


def _touch_pinned_pages(tensor) -> float:
    """Fault every host page without spending time zeroing the entire allocation."""

    started = time.perf_counter()
    view = tensor.numpy()
    view[::4096] = 1
    if view.size:
        view[-1] = 1
    return time.perf_counter() - started


def _copy_pass(
    source,
    destination,
    stream,
    chunk_bytes: int,
    barrier,
) -> dict[str, float | int]:
    import torch

    total_bytes = source.numel()
    if chunk_bytes > destination.numel():
        raise ValueError(
            f"chunk {chunk_bytes} exceeds destination window {destination.numel()}"
        )

    torch.cuda.synchronize()
    barrier.wait(timeout=10 * 60)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    with torch.cuda.stream(stream):
        start_event.record(stream)
        offset = 0
        copy_count = 0
        while offset < total_bytes:
            size = min(chunk_bytes, total_bytes - offset)
            dst_offset = (copy_count * chunk_bytes) % destination.numel()
            if dst_offset + size > destination.numel():
                dst_offset = 0
            destination[dst_offset : dst_offset + size].copy_(
                source[offset : offset + size], non_blocking=True
            )
            offset += size
            copy_count += 1
        end_event.record(stream)
    end_event.synchronize()
    wall_s = time.perf_counter() - wall_started
    cuda_s = start_event.elapsed_time(end_event) / 1000
    barrier.wait(timeout=10 * 60)
    return {
        "chunk_mib": chunk_bytes // MIB,
        "copy_count": copy_count,
        "bytes": total_bytes,
        "cuda_s": round(cuda_s, 6),
        "wall_s": round(wall_s, 6),
        "cuda_gbps": round(total_bytes / max(cuda_s, 1e-9) / 1e9, 3),
        "wall_gbps": round(total_bytes / max(wall_s, 1e-9) / 1e9, 3),
    }


def _staged_copy_pass(
    source,
    destination,
    pinned_buffers,
    stream,
    barrier,
) -> dict[str, float | int]:
    """Pipeline pageable CPU -> bounded pinned buffers -> GPU."""
    import torch

    chunk_bytes = pinned_buffers[0].numel()
    events = [torch.cuda.Event() for _ in pinned_buffers]
    pending = [False] * len(pinned_buffers)
    cpu_copy_s = 0.0
    copy_count = 0
    offset = 0

    torch.cuda.synchronize()
    barrier.wait(timeout=10 * 60)
    wall_started = time.perf_counter()
    while offset < source.numel():
        size = min(chunk_bytes, source.numel() - offset)
        slot = copy_count % len(pinned_buffers)
        if pending[slot]:
            events[slot].synchronize()

        cpu_started = time.perf_counter()
        pinned_buffers[slot][:size].copy_(source[offset : offset + size])
        cpu_copy_s += time.perf_counter() - cpu_started

        dst_offset = (copy_count * chunk_bytes) % destination.numel()
        if dst_offset + size > destination.numel():
            dst_offset = 0
        with torch.cuda.stream(stream):
            destination[dst_offset : dst_offset + size].copy_(
                pinned_buffers[slot][:size], non_blocking=True
            )
            events[slot].record(stream)
        pending[slot] = True
        offset += size
        copy_count += 1

    for slot, event in enumerate(events):
        if pending[slot]:
            event.synchronize()
    wall_s = time.perf_counter() - wall_started
    barrier.wait(timeout=10 * 60)
    return {
        "chunk_mib": chunk_bytes // MIB,
        "copy_count": copy_count,
        "bytes": source.numel(),
        "cpu_copy_s": round(cpu_copy_s, 6),
        "cpu_copy_gbps": round(source.numel() / max(cpu_copy_s, 1e-9) / 1e9, 3),
        "wall_s": round(wall_s, 6),
        "wall_gbps": round(source.numel() / max(wall_s, 1e-9) / 1e9, 3),
    }


def _parallel_staged_copy_pass(
    source,
    destination,
    pinned_buffers,
    stream,
    barrier,
) -> dict[str, float | int]:
    """Prefill a pinned ring, then refill it with parallel host memcpy calls."""
    import torch

    chunk_bytes = pinned_buffers[0].numel()
    events = [torch.cuda.Event() for _ in pinned_buffers]
    cpu_copy_s = 0.0
    copy_count = 0
    next_offset = 0

    def copy_chunk(slot: int, offset: int, wait_event=None):
        if wait_event is not None:
            wait_event.synchronize()
        size = min(chunk_bytes, source.numel() - offset)
        started = time.perf_counter()
        ctypes.memmove(
            pinned_buffers[slot].data_ptr(),
            source.data_ptr() + offset,
            size,
        )
        return offset, size, time.perf_counter() - started

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(pinned_buffers)
    ) as executor:
        ready = []
        for slot in range(len(pinned_buffers)):
            if next_offset >= source.numel():
                ready.append(None)
                continue
            ready.append(executor.submit(copy_chunk, slot, next_offset))
            next_offset += min(chunk_bytes, source.numel() - next_offset)
        # This is preparation work and intentionally outside the paused interval.
        ready = [future.result() if future is not None else None for future in ready]

        torch.cuda.synchronize()
        barrier.wait(timeout=10 * 60)
        wall_started = time.perf_counter()
        while any(item is not None for item in ready):
            for slot in range(len(pinned_buffers)):
                item = ready[slot]
                if item is None:
                    continue
                if isinstance(item, concurrent.futures.Future):
                    item = item.result()
                offset, size, copy_s = item
                cpu_copy_s += copy_s
                dst_offset = (copy_count * chunk_bytes) % destination.numel()
                if dst_offset + size > destination.numel():
                    dst_offset = 0
                with torch.cuda.stream(stream):
                    destination[dst_offset : dst_offset + size].copy_(
                        pinned_buffers[slot][:size], non_blocking=True
                    )
                    events[slot].record(stream)
                copy_count += 1
                if next_offset < source.numel():
                    ready[slot] = executor.submit(
                        copy_chunk,
                        slot,
                        next_offset,
                        events[slot],
                    )
                    next_offset += min(chunk_bytes, source.numel() - next_offset)
                else:
                    ready[slot] = None
        stream.synchronize()
        wall_s = time.perf_counter() - wall_started
        barrier.wait(timeout=10 * 60)

    return {
        "chunk_mib": chunk_bytes // MIB,
        "copy_count": copy_count,
        "bytes": source.numel(),
        "cpu_copy_thread_s": round(cpu_copy_s, 6),
        "wall_s": round(wall_s, 6),
        "wall_gbps": round(source.numel() / max(wall_s, 1e-9) / 1e9, 3),
    }


def _staged_worker(
    gpu_index: int,
    working_set_bytes: int,
    destination_bytes: int,
    chunk_bytes: int,
    buffer_count: int,
    barrier,
    output_queue,
) -> None:
    try:
        import torch

        topology = _gpu_topology(gpu_index)
        affinity = _bind_numa(topology)
        torch.set_num_threads(max(1, (os.cpu_count() or GPU_COUNT) // GPU_COUNT))
        torch.cuda.set_device(gpu_index)

        allocation_started = time.perf_counter()
        raw_source = torch.empty(working_set_bytes + 4096, dtype=torch.uint8)
        page_offset = (-raw_source.data_ptr()) % 4096
        source = raw_source[page_offset : page_offset + working_set_bytes]
        source_allocation_s = time.perf_counter() - allocation_started
        source_touch_s = _touch_pinned_pages(source)

        allocation_started = time.perf_counter()
        pinned_buffers = [
            torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(buffer_count)
        ]
        pinned_allocation_s = time.perf_counter() - allocation_started
        for pinned in pinned_buffers:
            _touch_pinned_pages(pinned)

        destination = torch.empty(
            destination_bytes, dtype=torch.uint8, device=f"cuda:{gpu_index}"
        )
        stream = torch.cuda.Stream(device=gpu_index)
        print(
            f"[GPU {gpu_index}] staged pass ready: {working_set_bytes / GIB:.1f} GiB "
            f"pageable, {buffer_count}x{chunk_bytes / MIB:.0f} MiB pinned",
            flush=True,
        )
        result = _staged_copy_pass(source, destination, pinned_buffers, stream, barrier)
        print(
            f"[GPU {gpu_index}] staged pass: {result['wall_s']:.3f}s, "
            f"{result['wall_gbps']:.2f} GB/s end-to-end",
            flush=True,
        )
        output_queue.put(
            {
                "ok": True,
                "gpu_index": gpu_index,
                "topology": topology,
                "affinity": affinity,
                "working_set_gb": round(source.numel() / 1e9, 3),
                "source_allocation_s": round(source_allocation_s, 3),
                "source_touch_s": round(source_touch_s, 3),
                "pinned_allocation_s": round(pinned_allocation_s, 3),
                "passes": [result],
            }
        )
    except Exception:
        try:
            barrier.abort()
        except Exception:
            pass
        error = traceback.format_exc()
        print(f"[GPU {gpu_index}] STAGED FAILED\n{error}", flush=True)
        output_queue.put({"ok": False, "gpu_index": gpu_index, "error": error})


def _parallel_staged_worker(
    gpu_index: int,
    working_set_bytes: int,
    destination_bytes: int,
    chunk_bytes: int,
    buffer_count: int,
    barrier,
    output_queue,
) -> None:
    try:
        import torch

        topology = _gpu_topology(gpu_index)
        affinity = _bind_numa(topology)
        torch.cuda.set_device(gpu_index)
        raw_source = torch.empty(working_set_bytes + 4096, dtype=torch.uint8)
        page_offset = (-raw_source.data_ptr()) % 4096
        source = raw_source[page_offset : page_offset + working_set_bytes]
        source_touch_s = _touch_pinned_pages(source)
        pinned_buffers = [
            torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True)
            for _ in range(buffer_count)
        ]
        for pinned in pinned_buffers:
            _touch_pinned_pages(pinned)
        destination = torch.empty(
            destination_bytes, dtype=torch.uint8, device=f"cuda:{gpu_index}"
        )
        stream = torch.cuda.Stream(device=gpu_index)
        print(
            f"[GPU {gpu_index}] parallel staged pass ready: "
            f"{working_set_bytes / GIB:.1f} GiB pageable, "
            f"{buffer_count}x{chunk_bytes / MIB:.0f} MiB pinned",
            flush=True,
        )
        result = _parallel_staged_copy_pass(
            source, destination, pinned_buffers, stream, barrier
        )
        print(
            f"[GPU {gpu_index}] parallel staged pass: {result['wall_s']:.3f}s, "
            f"{result['wall_gbps']:.2f} GB/s end-to-end",
            flush=True,
        )
        output_queue.put(
            {
                "ok": True,
                "gpu_index": gpu_index,
                "topology": topology,
                "affinity": affinity,
                "source_touch_s": round(source_touch_s, 3),
                "passes": [result],
            }
        )
    except Exception:
        try:
            barrier.abort()
        except Exception:
            pass
        error = traceback.format_exc()
        print(f"[GPU {gpu_index}] PARALLEL STAGED FAILED\n{error}", flush=True)
        output_queue.put({"ok": False, "gpu_index": gpu_index, "error": error})


def _cuda_host_call(result, operation: str) -> None:
    """Normalize the cuda-python return conventions exposed by torch.cudart."""
    code = result[0] if isinstance(result, tuple) else result
    if int(code) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {code}")


def _registered_worker(
    gpu_index: int,
    working_set_bytes: int,
    registration_bytes: int,
    destination_bytes: int,
    copy_bytes: int,
    barrier,
    output_queue,
) -> None:
    """Roll a two-chunk host-registration window over a pageable state image."""
    try:
        import torch

        topology = _gpu_topology(gpu_index)
        affinity = _bind_numa(topology)
        torch.cuda.set_device(gpu_index)
        raw_source = torch.empty(working_set_bytes + 4096, dtype=torch.uint8)
        page_offset = (-raw_source.data_ptr()) % 4096
        source = raw_source[page_offset : page_offset + working_set_bytes]
        touch_s = _touch_pinned_pages(source)
        destination = torch.empty(
            destination_bytes, dtype=torch.uint8, device=f"cuda:{gpu_index}"
        )
        stream = torch.cuda.Stream(device=gpu_index)
        cudart = torch.cuda.cudart()
        chunk_count = (source.numel() + registration_bytes - 1) // registration_bytes

        def register(index: int) -> float:
            offset = index * registration_bytes
            size = min(registration_bytes, source.numel() - offset)
            started = time.perf_counter()
            _cuda_host_call(
                cudart.cudaHostRegister(source.data_ptr() + offset, size, 0),
                f"cudaHostRegister(chunk={index})",
            )
            return time.perf_counter() - started

        def unregister(index: int) -> float:
            offset = index * registration_bytes
            started = time.perf_counter()
            _cuda_host_call(
                cudart.cudaHostUnregister(source.data_ptr() + offset),
                f"cudaHostUnregister(chunk={index})",
            )
            return time.perf_counter() - started

        def enqueue(index: int) -> int:
            chunk_offset = index * registration_bytes
            chunk_size = min(registration_bytes, source.numel() - chunk_offset)
            copied = 0
            copy_count = 0
            with torch.cuda.stream(stream):
                while copied < chunk_size:
                    size = min(copy_bytes, chunk_size - copied)
                    dst_offset = (copy_count * copy_bytes) % destination.numel()
                    destination[dst_offset : dst_offset + size].copy_(
                        source[chunk_offset + copied : chunk_offset + copied + size],
                        non_blocking=True,
                    )
                    copied += size
                    copy_count += 1
            return copy_count

        preregister_s = [register(index) for index in range(min(2, chunk_count))]
        print(
            f"[GPU {gpu_index}] registered pass ready: "
            f"{working_set_bytes / GIB:.1f} GiB pageable, "
            f"2x{registration_bytes / GIB:.1f} GiB registered; "
            f"registration={preregister_s}",
            flush=True,
        )
        barrier.wait(timeout=10 * 60)
        wall_started = time.perf_counter()
        register_s = list(preregister_s)
        unregister_s = []
        total_copy_count = 0

        # Consume the first chunk, then each following GPU transfer can overlap
        # registration of the chunk after it without exceeding two live chunks.
        total_copy_count += enqueue(0)
        stream.synchronize()
        unregister_s.append(unregister(0))
        for index in range(1, chunk_count):
            total_copy_count += enqueue(index)
            next_index = index + 1
            if next_index < chunk_count:
                register_s.append(register(next_index))
            stream.synchronize()
            unregister_s.append(unregister(index))
        wall_s = time.perf_counter() - wall_started
        barrier.wait(timeout=10 * 60)
        output_queue.put(
            {
                "ok": True,
                "gpu_index": gpu_index,
                "topology": topology,
                "affinity": affinity,
                "source_touch_s": round(touch_s, 3),
                "registration_gib": registration_bytes / GIB,
                "preregister_s": [round(value, 6) for value in preregister_s],
                "register_s": [round(value, 6) for value in register_s],
                "unregister_s": [round(value, 6) for value in unregister_s],
                "passes": [
                    {
                        "chunk_mib": copy_bytes // MIB,
                        "copy_count": total_copy_count,
                        "bytes": source.numel(),
                        "wall_s": round(wall_s, 6),
                        "wall_gbps": round(source.numel() / wall_s / 1e9, 3),
                    }
                ],
            }
        )
    except Exception:
        try:
            barrier.abort()
        except Exception:
            pass
        error = traceback.format_exc()
        print(f"[GPU {gpu_index}] REGISTERED FAILED\n{error}", flush=True)
        output_queue.put({"ok": False, "gpu_index": gpu_index, "error": error})


def _worker(
    gpu_index: int,
    working_set_bytes: int,
    window_bytes: int,
    chunk_bytes: list[int],
    barrier,
    output_queue,
) -> None:
    try:
        import torch

        print(f"[GPU {gpu_index}] discovering topology", flush=True)
        topology = _gpu_topology(gpu_index)
        affinity = _bind_numa(topology)
        torch.cuda.set_device(gpu_index)
        properties = torch.cuda.get_device_properties(gpu_index)

        print(
            f"[GPU {gpu_index}] allocating {working_set_bytes / GIB:.1f} GiB "
            f"pinned on NUMA node {topology['numa_node']}",
            flush=True,
        )
        allocation_started = time.perf_counter()
        source = torch.empty(working_set_bytes, dtype=torch.uint8, pin_memory=True)
        allocation_s = time.perf_counter() - allocation_started
        print(
            f"[GPU {gpu_index}] pinned allocation ready in {allocation_s:.2f}s; "
            "touching pages",
            flush=True,
        )
        touch_s = _touch_pinned_pages(source)
        print(
            f"[GPU {gpu_index}] pages ready in {touch_s:.2f}s; allocating GPU window",
            flush=True,
        )
        destination = torch.empty(window_bytes, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream(device=gpu_index)

        # Warm the copy engine, CUDA context, and first destination pages before the
        # synchronized full-working-set passes.
        warm_bytes = min(1 * GIB, source.numel(), destination.numel())
        destination[:warm_bytes].copy_(source[:warm_bytes], non_blocking=True)
        torch.cuda.synchronize()

        passes = []
        for size in chunk_bytes:
            print(
                f"[GPU {gpu_index}] waiting for {size / MIB:.0f} MiB copy pass",
                flush=True,
            )
            result = _copy_pass(source, destination, stream, size, barrier)
            passes.append(result)
            print(
                f"[GPU {gpu_index}] {size / MIB:.0f} MiB pass: "
                f"{result['wall_s']:.3f}s, {result['wall_gbps']:.2f} GB/s",
                flush=True,
            )
        output_queue.put(
            {
                "ok": True,
                "gpu_index": gpu_index,
                "topology": topology,
                "affinity": affinity,
                "gpu_total_memory_gb": round(properties.total_memory / 1e9, 3),
                "working_set_gb": round(source.numel() / 1e9, 3),
                "window_gb": round(destination.numel() / 1e9, 3),
                "pinned_allocation_s": round(allocation_s, 3),
                "page_touch_s": round(touch_s, 3),
                "passes": passes,
            }
        )
    except Exception:
        try:
            barrier.abort()
        except Exception:
            pass
        error = traceback.format_exc()
        print(f"[GPU {gpu_index}] FAILED\n{error}", flush=True)
        output_queue.put(
            {
                "ok": False,
                "gpu_index": gpu_index,
                "error": error,
            }
        )


def _aggregate(workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    good = [worker for worker in workers if worker.get("ok")]
    if not good:
        return []
    by_chunk: dict[int, list[dict[str, Any]]] = {}
    for worker in good:
        for result in worker["passes"]:
            by_chunk.setdefault(result["chunk_mib"], []).append(result)

    aggregate = []
    for chunk_mib, results in sorted(by_chunk.items()):
        total_bytes = sum(result["bytes"] for result in results)
        critical_wall_s = max(result["wall_s"] for result in results)
        summary = {
            "chunk_mib": chunk_mib,
            "gpu_count": len(results),
            "total_bytes": total_bytes,
            "critical_wall_s": round(critical_wall_s, 6),
            "aggregate_wall_gbps": round(
                total_bytes / max(critical_wall_s, 1e-9) / 1e9, 3
            ),
            "per_gpu_wall_gbps": [result["wall_gbps"] for result in results],
        }
        if all("cuda_s" in result for result in results):
            critical_cuda_s = max(result["cuda_s"] for result in results)
            summary["critical_cuda_s"] = round(critical_cuda_s, 6)
            summary["aggregate_cuda_gbps"] = round(
                total_bytes / max(critical_cuda_s, 1e-9) / 1e9, 3
            )
        aggregate.append(summary)
    return aggregate


@app.function(
    image=image,
    gpu="B300:4",
    memory=(1024, 2 * 1024 * 1024),
    timeout=60 * 60,
)
def benchmark(
    working_set_gb: int = DEFAULT_WORKING_SET_GB,
    window_gb: int = DEFAULT_WINDOW_GB,
    chunk_mib_csv: str = DEFAULT_CHUNK_MIB,
) -> dict[str, Any]:
    chunk_mib = [int(value) for value in chunk_mib_csv.split(",") if value]
    if not chunk_mib:
        raise ValueError("at least one chunk size is required")
    if max(chunk_mib) * MIB > window_gb * GIB:
        raise ValueError("the destination window must be at least the largest chunk")

    context = mp.get_context("spawn")
    barrier = context.Barrier(GPU_COUNT)
    output_queue = context.Queue()
    workers = [
        context.Process(
            target=_worker,
            args=(
                gpu_index,
                working_set_gb * GIB,
                window_gb * GIB,
                [value * MIB for value in chunk_mib],
                barrier,
                output_queue,
            ),
        )
        for gpu_index in range(GPU_COUNT)
    ]

    started = time.perf_counter()
    for worker in workers:
        worker.start()
    results = []
    deadline = time.monotonic() + 55 * 60
    while len(results) < len(workers):
        try:
            results.append(output_queue.get(timeout=5))
            continue
        except queue.Empty:
            pass
        reported_gpu_indices = {result["gpu_index"] for result in results}
        dead_without_result = [
            worker
            for index, worker in enumerate(workers)
            if worker.exitcode is not None and index not in reported_gpu_indices
        ]
        if dead_without_result or time.monotonic() >= deadline:
            try:
                barrier.abort()
            except Exception:
                pass
            for index, worker in enumerate(workers):
                if not any(result["gpu_index"] == index for result in results):
                    results.append(
                        {
                            "ok": False,
                            "gpu_index": index,
                            "error": (
                                f"worker exited without a report: exitcode={worker.exitcode}"
                            ),
                        }
                    )
            break
    for worker in workers:
        worker.join(timeout=60)
    elapsed_s = time.perf_counter() - started
    results.sort(key=lambda result: result["gpu_index"])

    report = {
        "gpu": "B300:4",
        "working_set_gib_per_gpu": working_set_gb,
        "working_set_gb_total": round(GPU_COUNT * working_set_gb * GIB / 1e9, 3),
        "window_gib_per_gpu": window_gb,
        "chunk_mib": chunk_mib,
        "elapsed_s": round(elapsed_s, 3),
        "host": {
            "cpu_count": os.cpu_count(),
            "nvidia_smi": _command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,pci.bus_id,name,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "nvidia_smi_topology": _command(["nvidia-smi", "topo", "-m"]),
            "numactl": _command(["numactl", "--hardware"]),
            "lscpu": _command(["lscpu"]),
        },
        "workers": results,
        "aggregate": _aggregate(results),
        "status": "passed" if all(result.get("ok") for result in results) else "failed",
    }
    print("=== B300 H2D PHYSICS RESULT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.function(
    image=image,
    gpu="B300:4",
    memory=(1024, 2 * 1024 * 1024),
    timeout=60 * 60,
)
def benchmark_staged(
    working_set_gb: int = DEFAULT_WORKING_SET_GB,
    destination_gb: int = DEFAULT_WINDOW_GB,
    chunk_mib: int = 1024,
    buffer_count: int = 2,
) -> dict[str, Any]:
    """Measure the bounded-pinned-buffer path required by a 605 GB state."""
    if buffer_count < 2:
        raise ValueError("buffer_count must be at least two for overlap")
    if chunk_mib * MIB > destination_gb * GIB:
        raise ValueError("destination must be at least one chunk")

    context = mp.get_context("spawn")
    barrier = context.Barrier(GPU_COUNT)
    output_queue = context.Queue()
    workers = [
        context.Process(
            target=_staged_worker,
            args=(
                gpu_index,
                working_set_gb * GIB,
                destination_gb * GIB,
                chunk_mib * MIB,
                buffer_count,
                barrier,
                output_queue,
            ),
        )
        for gpu_index in range(GPU_COUNT)
    ]
    for worker in workers:
        worker.start()
    results = [output_queue.get(timeout=50 * 60) for _ in workers]
    for worker in workers:
        worker.join(timeout=60)
    results.sort(key=lambda result: result["gpu_index"])
    report = {
        "gpu": "B300:4",
        "path": "pageable_cpu_to_bounded_pinned_to_gpu",
        "working_set_gib_per_gpu": working_set_gb,
        "chunk_mib": chunk_mib,
        "buffer_count": buffer_count,
        "workers": results,
        "aggregate": _aggregate(results),
        "status": "passed" if all(result.get("ok") for result in results) else "failed",
    }
    print("=== B300 STAGED H2D RESULT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.function(
    image=image,
    gpu="B300:4",
    memory=(1024, 2 * 1024 * 1024),
    timeout=60 * 60,
)
def benchmark_staged_parallel(
    working_set_gb: int = 96,
    destination_gb: int = DEFAULT_WINDOW_GB,
    chunk_mib: int = 1024,
    buffer_count: int = 8,
) -> dict[str, Any]:
    """Measure a prefilled, parallel-refill pinned ring for the paused commit."""
    if buffer_count < 2:
        raise ValueError("buffer_count must be at least two for overlap")
    if chunk_mib * MIB > destination_gb * GIB:
        raise ValueError("destination must be at least one chunk")

    context = mp.get_context("spawn")
    barrier = context.Barrier(GPU_COUNT)
    output_queue = context.Queue()
    workers = [
        context.Process(
            target=_parallel_staged_worker,
            args=(
                gpu_index,
                working_set_gb * GIB,
                destination_gb * GIB,
                chunk_mib * MIB,
                buffer_count,
                barrier,
                output_queue,
            ),
        )
        for gpu_index in range(GPU_COUNT)
    ]
    for worker in workers:
        worker.start()
    results = [output_queue.get(timeout=50 * 60) for _ in workers]
    for worker in workers:
        worker.join(timeout=60)
    results.sort(key=lambda result: result["gpu_index"])
    report = {
        "gpu": "B300:4",
        "path": "parallel_pageable_to_prefilled_pinned_ring_to_gpu",
        "working_set_gib_per_gpu": working_set_gb,
        "chunk_mib": chunk_mib,
        "buffer_count": buffer_count,
        "workers": results,
        "aggregate": _aggregate(results),
        "status": "passed" if all(result.get("ok") for result in results) else "failed",
    }
    print("=== B300 PARALLEL STAGED H2D RESULT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.function(
    image=image,
    gpu="B300:4",
    memory=(1024, 2 * 1024 * 1024),
    timeout=60 * 60,
)
def benchmark_registered(
    working_set_gb: int = 96,
    registration_gb: int = 32,
    destination_gb: int = DEFAULT_WINDOW_GB,
    copy_mib: int = 1024,
) -> dict[str, Any]:
    """Measure a rolling two-chunk cudaHostRegister commit path."""
    if working_set_gb < registration_gb * 2:
        raise ValueError("working set must contain at least two registration chunks")
    context = mp.get_context("spawn")
    barrier = context.Barrier(GPU_COUNT)
    output_queue = context.Queue()
    workers = [
        context.Process(
            target=_registered_worker,
            args=(
                gpu_index,
                working_set_gb * GIB,
                registration_gb * GIB,
                destination_gb * GIB,
                copy_mib * MIB,
                barrier,
                output_queue,
            ),
        )
        for gpu_index in range(GPU_COUNT)
    ]
    for worker in workers:
        worker.start()
    results = [output_queue.get(timeout=50 * 60) for _ in workers]
    for worker in workers:
        worker.join(timeout=60)
    results.sort(key=lambda result: result["gpu_index"])
    report = {
        "gpu": "B300:4",
        "path": "rolling_cuda_host_register_to_gpu",
        "working_set_gib_per_gpu": working_set_gb,
        "registration_gib_per_chunk": registration_gb,
        "workers": results,
        "aggregate": _aggregate(results),
        "status": "passed" if all(result.get("ok") for result in results) else "failed",
    }
    print("=== B300 REGISTERED H2D RESULT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


@app.local_entrypoint()
def main(
    working_set_gb: int = DEFAULT_WORKING_SET_GB,
    window_gb: int = DEFAULT_WINDOW_GB,
    chunk_mib: str = DEFAULT_CHUNK_MIB,
) -> None:
    result = benchmark.remote(working_set_gb, window_gb, chunk_mib)
    print(json.dumps(result, indent=2))
