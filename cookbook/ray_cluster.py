"""Ray cluster bring-up helpers for multi-node Modal training.

Shared across cookbook trainers (slime, miles, etc.). Originally adapted from
https://github.com/modal-projects/multinode-training-guide.
"""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import Any


RAY_START_TIMEOUT = 240
RAY_WORKER_JOIN_TIMEOUT = 180


def training_nodes(cfg: Any) -> int:
    nodes = int(getattr(cfg, "actor_num_nodes", 1))
    if getattr(cfg, "use_critic", False) or getattr(cfg, "advantage_estimator", None) == "ppo":
        nodes += int(getattr(cfg, "critic_num_nodes", nodes))
    return nodes


def get_modal_cluster_context(n_nodes: int) -> tuple[int, str, str]:
    """Return (rank, master_addr, my_ip) for the current Modal cluster."""
    import modal.experimental

    try:
        info = modal.experimental.get_cluster_info()
    except Exception:  # noqa: BLE001
        if n_nodes == 1:
            ip = _get_local_ip()
            return 0, ip, ip
        raise
    actual_nodes = len(info.container_ipv4_ips)
    if actual_nodes == 0 and n_nodes == 1:
        ip = _get_local_ip()
        return 0, ip, ip
    if actual_nodes != n_nodes:
        raise RuntimeError(f"cluster size mismatch: expected {n_nodes} node(s), got {actual_nodes}")
    return info.rank, info.container_ipv4_ips[0], info.container_ipv4_ips[info.rank]


def _get_local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def start_ray_head(my_ip: str, n_nodes: int, *, ray_port: int) -> None:
    """Start the Ray head node and wait for all workers to join."""
    import ray

    start_cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={my_ip}",
        f"--port={ray_port}",
        "--disable-usage-stats",
        "--include-dashboard=false",
    ]
    try:
        subprocess.run(start_cmd, check=True, timeout=RAY_START_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        _print_ray_logs()
        raise RuntimeError(f"Ray head node did not finish startup within {RAY_START_TIMEOUT}s") from exc
    except subprocess.CalledProcessError as exc:
        _print_ray_logs()
        raise RuntimeError(f"Ray head node failed to start with exit code {exc.returncode}") from exc

    last_error = ""
    for _ in range(RAY_START_TIMEOUT):
        try:
            ray.init(address=f"{my_ip}:{ray_port}")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
    else:
        _print_ray_logs()
        raise RuntimeError(f"Ray head node failed to start before timeout: {last_error}")

    for _ in range(RAY_WORKER_JOIN_TIMEOUT):
        alive = [n for n in ray.nodes() if n["Alive"]]
        print(f"Waiting for workers: {len(alive)}/{n_nodes} alive")
        if len(alive) == n_nodes:
            break
        time.sleep(1)
    else:
        _print_ray_logs()
        raise RuntimeError(f"Timed out waiting for all {n_nodes} Ray nodes to join")


def start_ray_worker(my_ip: str, master_addr: str, *, ray_port: int) -> None:
    """Join this container to the Ray cluster as a worker node."""
    subprocess.run(
        [
            "ray",
            "start",
            f"--node-ip-address={my_ip}",
            "--address",
            f"{master_addr}:{ray_port}",
            "--disable-usage-stats",
        ],
        check=True,
        timeout=RAY_START_TIMEOUT,
    )


def _print_ray_logs() -> None:
    log_dir = Path("/tmp/ray/session_latest/logs")
    for name in (
        "dashboard.log",
        "dashboard.err",
        "gcs_server.out",
        "gcs_server.err",
        "raylet.out",
        "raylet.err",
        "monitor.out",
        "monitor.err",
    ):
        path = log_dir / name
        if not path.exists():
            continue
        print(f"===== {path} =====")
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            print(f"could not read {path}: {exc}")
            continue
        for line in lines[-80:]:
            print(line)
