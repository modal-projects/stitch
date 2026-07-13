"""Ray cluster bring-up for multi-node Modal training — framework-agnostic."""

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
    """(rank, master_addr, my_ip) for the current Modal cluster (single-node safe)."""
    import modal.experimental

    try:
        info = modal.experimental.get_cluster_info()
    except Exception:  # noqa: BLE001
        if n_nodes == 1:
            ip = _local_ip()
            return 0, ip, ip
        raise
    actual = len(info.container_ipv4_ips)
    if actual == 0 and n_nodes == 1:
        ip = _local_ip()
        return 0, ip, ip
    if actual != n_nodes:
        raise RuntimeError(f"cluster size mismatch: expected {n_nodes} node(s), got {actual}")
    return info.rank, info.container_ipv4_ips[0], info.container_ipv4_ips[info.rank]


def _local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def start_ray_head(my_ip: str, n_nodes: int, *, ray_port: int) -> None:
    import ray

    try:
        subprocess.run(
            ["ray", "start", "--head", f"--node-ip-address={my_ip}", f"--port={ray_port}",
             "--disable-usage-stats", "--include-dashboard=false"],
            check=True, timeout=RAY_START_TIMEOUT,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        _print_ray_logs()
        raise RuntimeError(f"Ray head failed to start: {exc}") from exc

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
        raise RuntimeError(f"Ray head failed to start before timeout: {last_error}")

    for _ in range(RAY_WORKER_JOIN_TIMEOUT):
        alive = [n for n in ray.nodes() if n["Alive"]]
        print(f"Waiting for workers: {len(alive)}/{n_nodes} alive")
        if len(alive) == n_nodes:
            return
        time.sleep(1)
    _print_ray_logs()
    raise RuntimeError(f"Timed out waiting for all {n_nodes} Ray nodes to join")


def start_ray_worker(my_ip: str, master_addr: str, *, ray_port: int) -> None:
    subprocess.run(
        ["ray", "start", f"--node-ip-address={my_ip}", "--address", f"{master_addr}:{ray_port}",
         "--disable-usage-stats"],
        check=True, timeout=RAY_START_TIMEOUT,
    )


def _print_ray_logs() -> None:
    log_dir = Path("/tmp/ray/session_latest/logs")
    for name in ("gcs_server.out", "gcs_server.err", "raylet.out", "raylet.err", "monitor.err"):
        path = log_dir / name
        if not path.exists():
            continue
        print(f"===== {path} =====")
        try:
            for line in path.read_text(errors="replace").splitlines()[-80:]:
                print(line)
        except OSError as exc:
            print(f"could not read {path}: {exc}")
