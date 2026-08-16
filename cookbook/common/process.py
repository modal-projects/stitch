"""Subprocess + runtime helpers shared by every recipe: launch the sidecar beside
sglang, wait on HTTP liveness, terminate cleanly, monitor host RAM, apply git patches.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from stitch.sidecar import SidecarConfig

# The sidecar entrypoint lives in stitch core; the recipe launches it beside the
# recipe-specific sglang server, pointing --store-factory at the cookbook's
# backend dispatch so core never has to know this repo's Store backends.
SIDECAR_MODULE = "stitch.sidecar"
STORE_FACTORY = "cookbook.common.storage:create_store"


def start_sidecar(
    *,
    sidecar_port: int,
    sglang_port: int,
    bulletin_root: str,
    base_checkpoint_dir: str,
    local_checkpoint_dir: str | None,
    delta_update_mode: str,
    disk_load_format: str,
    store_backend: str,
    volume_name: str,
    s3_root: str | None,
    s3_endpoint_url: str | None,
    run_id: str,
    boot_version: int,
    commit_mode: str,
    flush_cache_on_commit: bool = False,
    debug_requests: bool = False,
) -> subprocess.Popen:
    """Launch the versioned rollout proxy (the shared sidecar) beside sglang."""
    # Empty settings normalize to unset: only set options reach the factory.
    store_options = {"backend": store_backend}
    if volume_name:
        store_options["volume_name"] = volume_name
    if s3_root:
        store_options["s3_root"] = s3_root
    if s3_endpoint_url:
        store_options["s3_endpoint_url"] = s3_endpoint_url
    config = SidecarConfig(
        host="0.0.0.0",
        port=sidecar_port,
        upstream=f"http://127.0.0.1:{sglang_port}",
        bulletin_root=bulletin_root,
        base_checkpoint_dir=base_checkpoint_dir,
        local_checkpoint_dir=local_checkpoint_dir,
        delta_update_mode=delta_update_mode,
        disk_load_format=disk_load_format,
        store_factory=STORE_FACTORY,
        store_options=store_options,
        commit_mode=commit_mode,
        flush_cache_on_commit=flush_cache_on_commit,
        run_id=run_id,
        boot_version=boot_version,
        debug_requests=debug_requests,
    )
    cmd = [
        "python3",
        "-m",
        SIDECAR_MODULE,
        *config.to_argv(),
    ]
    print("Starting sidecar:", " ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def wait_http(url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"process exited while waiting for {url}: code={process.returncode}"
            )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for {url}; last error: {last_error}")


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def apply_git_patches(patch_paths: list[str], repo_dir: str, label: str) -> None:
    """Apply git patches to a runtime checkout, tolerating an already-applied patch
    (idempotent across container restarts)."""
    for patch_path in patch_paths:
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"{label} not found: {patch_path}")
        check = subprocess.run(
            ["git", "-C", repo_dir, "apply", "--check", patch_path],
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            subprocess.run(["git", "-C", repo_dir, "apply", patch_path], check=True)
            print(f"[{label}] applied {patch_path}", flush=True)
            continue
        reverse = subprocess.run(
            ["git", "-C", repo_dir, "apply", "--reverse", "--check", patch_path],
            capture_output=True,
            text=True,
        )
        if reverse.returncode == 0:
            print(f"[{label}] already applied {patch_path}", flush=True)
            continue
        raise RuntimeError(
            f"cannot apply {label} {patch_path}\ncheck: {check.stderr}\nreverse: {reverse.stderr}"
        )


def start_host_mem_monitor(interval_s: int = 20) -> None:
    """Trace this node's host-RAM from a daemon thread. Modal exposes no host-RAM metric, so
    this log line is the only signal for the OOM peak (the publish weight-gather). Best-effort."""
    host = socket.gethostname()

    def _meminfo() -> tuple[float, float]:
        total = avail = 0.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) / 1024 / 1024
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) / 1024 / 1024
        except Exception:  # noqa: BLE001
            pass
        return total, avail

    def _loop() -> None:
        # Multi-node runs emit one line per node per tick, which floods the aggregate log. Stay quiet
        # unless host RAM is climbing toward a host-OOM (avail < 500 GiB) or a sparse 10-min heartbeat —
        # the OOM danger zone is the only signal worth surfacing; normal operation is silent.
        heartbeat = max(1, 600 // interval_s)
        i = 0
        while True:
            total, avail = _meminfo()
            if i == 0 or avail < 500 or i % heartbeat == 0:
                print(
                    f"[hostmem] {host} used={total - avail:.0f}GiB avail={avail:.0f}GiB total={total:.0f}GiB",
                    flush=True,
                )
            i += 1
            time.sleep(interval_s)

    threading.Thread(target=_loop, daemon=True, name="host-mem-monitor").start()


def dist_rank() -> int | None:
    """This process's torch.distributed rank, or None off the distributed path (single-process dev)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001
        return None
    return None


def dist_barrier() -> None:
    """Wait for all ranks; a no-op off the distributed path."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def dist_all_gather_object(value: Any) -> list[Any]:
    """Gather one small control-plane value from every distributed rank."""
    try:
        import torch.distributed as dist
    except ImportError:
        return [value]

    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def dist_is_container_leader() -> bool:
    """Return whether this rank owns container-scoped side effects.

    Distributed trainer ranks are separate processes, but ranks hosted by one
    Modal container share its mounted Volumes. Elect the lowest global rank in
    each container so a container-scoped operation runs once per mount.
    """
    rank = dist_rank()
    if rank is None:
        return True
    container_id = os.environ.get("MODAL_TASK_ID")
    container_ids = dist_all_gather_object(container_id)
    if not all(container_ids):
        # Preserve the previous one-commit-per-rank behavior when the runtime
        # cannot prove which processes share a mount.
        return True
    return rank == container_ids.index(container_id)
