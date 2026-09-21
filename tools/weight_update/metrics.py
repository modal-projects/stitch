"""Host resource measurements for weight-update validation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any


class _MemoryProbe:
    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.samples: list[dict[str, int | str]] = []
        self.thread = threading.Thread(
            target=self._run,
            name="memory-during-weight-stage",
            daemon=True,
        )

    def _run(self) -> None:
        while not self.stop.is_set():
            self.samples.append(_memory_snapshot())
            self.stop.wait(self.interval_s)

    def __enter__(self) -> _MemoryProbe:
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop.set()
        self.thread.join(timeout=max(5.0, 2 * self.interval_s))

    def summary(self) -> dict[str, int | str | None]:
        numeric_keys = {
            key
            for sample in self.samples
            for key, value in sample.items()
            if isinstance(value, int)
        }
        result: dict[str, int | str | None] = {"samples": len(self.samples)}
        for key in sorted(numeric_keys):
            values = [
                value
                for sample in self.samples
                if isinstance(value := sample.get(key), int)
            ]
            if key == "MemAvailable_bytes":
                result[f"{key}_min"] = min(values)
            else:
                result[f"{key}_max"] = max(values)
        return result


def _memory_snapshot() -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    cgroup_files = {
        "memory_current": (
            Path("/sys/fs/cgroup/memory.current"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
        "memory_peak": (
            Path("/sys/fs/cgroup/memory.peak"),
            Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
        ),
        "memory_max": (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ),
    }
    for key, candidates in cgroup_files.items():
        path = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if path is not None:
            value = path.read_text().strip()
            result[key] = value if value == "max" else int(value)
    wanted = {"MemTotal", "MemAvailable", "Cached", "AnonPages", "Shmem"}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator and key in wanted:
            result[f"{key}_bytes"] = int(value.split()[0]) * 1024
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu_used_mib = [
            int(line.strip()) for line in completed.stdout.splitlines() if line.strip()
        ]
        if gpu_used_mib:
            result["gpu_memory_used_total_bytes"] = sum(gpu_used_mib) << 20
            result["gpu_memory_used_max_rank_bytes"] = max(gpu_used_mib) << 20
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return result


def _local_storage_snapshot(paths: dict[str, str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for label, root in paths.items():
        total = 0
        if os.path.isdir(root):
            for directory, _, filenames in os.walk(root):
                for filename in filenames:
                    try:
                        total += os.path.getsize(os.path.join(directory, filename))
                    except FileNotFoundError:
                        pass
        result[f"{label}_bytes"] = total
    usage = shutil.disk_usage("/")
    result["filesystem_used_bytes"] = usage.used
    result["filesystem_free_bytes"] = usage.free
    return result


def _cgroup_cpu_usage_s() -> float | None:
    cgroup_v2 = Path("/sys/fs/cgroup/cpu.stat")
    if cgroup_v2.is_file():
        for line in cgroup_v2.read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "usage_usec":
                return int(value) / 1_000_000
    cgroup_v1 = Path("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    if cgroup_v1.is_file():
        return int(cgroup_v1.read_text()) / 1_000_000_000
    return None


def _cgroup_io_snapshot() -> dict[str, int] | None:
    path = Path("/sys/fs/cgroup/io.stat")
    if not path.is_file():
        return None
    result: dict[str, int] = {}
    for line in path.read_text().splitlines():
        for field in line.split()[1:]:
            key, separator, value = field.partition("=")
            if separator:
                result[key] = result.get(key, 0) + int(value)
    return result


def _io_usage_delta(
    before: dict[str, int] | None,
    after: dict[str, int] | None,
) -> dict[str, int] | None:
    if before is None or after is None:
        return None
    return {
        key: max(0, after.get(key, 0) - before.get(key, 0))
        for key in sorted(before.keys() | after.keys())
    }


def _resource_snapshot() -> dict[str, Any]:
    gpu_memory_bytes = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in completed.stdout.splitlines():
            gpu_memory_bytes.append(int(line.strip()) << 20)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return {
        "allowed_cpu_count": len(os.sched_getaffinity(0)),
        "numa_node_count": sum(
            1
            for path in Path("/sys/devices/system/node").glob("node[0-9]*")
            if path.is_dir()
        ),
        "gpu_count": len(gpu_memory_bytes),
        "gpu_memory_bytes": gpu_memory_bytes,
        "memory": _memory_snapshot(),
    }


def _cpu_usage_delta(
    before: float | None,
    after: float | None,
    wall_s: float,
) -> dict[str, float] | None:
    if before is None or after is None:
        return None
    cpu_s = max(0.0, after - before)
    return {
        "cpu_s": round(cpu_s, 6),
        "average_cores": round(cpu_s / max(wall_s, 1e-9), 3),
    }


def _print_profile_summary(results: dict[str, Any]) -> None:
    summary = {
        key: results.get(key)
        for key in (
            "model",
            "runtime",
            "update_mode",
            "canonical_storage",
            "sample_id",
            "status",
            "served_version",
            "failure_version",
            "startup_ready_s",
            "startup_cpu",
            "memory_during_startup",
        )
        if key in results
    }
    summary["updates"] = [
        {
            key: update.get(key)
            for key in (
                "from_version",
                "target_version",
                "delta_count",
                "prepare_s",
                "prepare_cpu",
                "commit_rpc_s",
                "commit_cpu",
                "prepare_through_commit_s",
                "generation_during_prepare",
                "critical_rank_prepare_s",
                "critical_rank_commit_s",
                "correctness",
            )
        }
        for update in results.get("updates", [])
    ]
    for key in ("preparation_failure", "post_mutation_failure"):
        if key in results:
            summary[key] = results[key]
    print(f"PROFILE_SUMMARY={json.dumps(summary, sort_keys=True)}", flush=True)
