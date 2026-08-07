"""Ray cluster bring-up for multi-node Modal training — framework-agnostic."""

from __future__ import annotations

import fcntl
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

RAY_START_TIMEOUT = 240
RAY_WORKER_JOIN_TIMEOUT = 180
_SIOCGIFADDR = 0x8915


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
        raise RuntimeError(
            f"cluster size mismatch: expected {n_nodes} node(s), got {actual}"
        )
    return info.rank, info.container_ipv4_ips[0], info.container_ipv4_ips[info.rank]


def _local_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def _ipv4_interfaces() -> list[tuple[str, str]]:
    interfaces: list[tuple[str, str]] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            request = struct.pack("256s", name.encode()[:15])
            try:
                address = socket.inet_ntoa(
                    fcntl.ioctl(sock.fileno(), _SIOCGIFADDR, request)[20:24]
                )
            except OSError:
                continue
            interfaces.append((name, address))
    return interfaces


def _interface_for_ip(ip: str) -> str:
    for name, address in _ipv4_interfaces():
        if address == ip:
            return name
    raise RuntimeError(f"no network interface owns Modal cluster IP {ip}")


def _ray_start_command(my_ip: str, *args: str) -> list[str]:
    # NVSHMEM uses the hostname to identify node-local peers. Modal containers
    # share a hostname, so Ray and its workers need a per-node UTS hostname.
    node_hostname = f"stitch-{my_ip.replace('.', '-')}"
    return [
        "unshare",
        "--user",
        "--map-root-user",
        "--uts",
        "bash",
        "-c",
        'hostname "$1" && shift && exec "$@"',
        "stitch-ray-node",
        node_hostname,
        "ray",
        "start",
        *args,
    ]


def start_ray_head(my_ip: str, n_nodes: int, *, ray_port: int) -> None:
    import ray

    try:
        subprocess.run(
            _ray_start_command(
                my_ip,
                "--head",
                f"--node-ip-address={my_ip}",
                f"--port={ray_port}",
                "--disable-usage-stats",
                "--include-dashboard=false",
            ),
            check=True,
            timeout=RAY_START_TIMEOUT,
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
        _ray_start_command(
            my_ip,
            f"--node-ip-address={my_ip}",
            "--address",
            f"{master_addr}:{ray_port}",
            "--disable-usage-stats",
        ),
        check=True,
        timeout=RAY_START_TIMEOUT,
    )


def start_ray_node(
    rank: int,
    master_addr: str,
    my_ip: str,
    *,
    n_nodes: int,
    ray_port: int,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Set this node's Ray/NCCL env, then bring Ray up (head on rank 0, worker otherwise).
    ``extra_env`` overlays the framework-specific vars a recipe adds — its own HOST_IP alias, a
    PYTHONPATH, its training ``environment``."""
    cluster_interface = _interface_for_ip(my_ip)
    print(f"Modal cluster network: {my_ip} via {cluster_interface}")
    os.environ.update(
        {
            "SGLANG_HOST_IP": my_ip,
            "HOST_IP": my_ip,
            "MASTER_ADDR": master_addr,
            "RAY_ADDRESS": f"{master_addr}:{ray_port}",
            "no_proxy": f"127.0.0.1,{master_addr},{my_ip}",
            "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
            # NVSHMEM's UID bootstrap otherwise auto-selects an interface that
            # can be container-local and unreachable from another Modal node.
            "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": f"={cluster_interface}",
            **(extra_env or {}),
        }
    )
    if rank == 0:
        start_ray_head(my_ip, n_nodes, ray_port=ray_port)
    else:
        start_ray_worker(my_ip, master_addr, ray_port=ray_port)


def _print_ray_logs() -> None:
    log_dir = Path("/tmp/ray/session_latest/logs")
    for name in (
        "gcs_server.out",
        "gcs_server.err",
        "raylet.out",
        "raylet.err",
        "monitor.err",
    ):
        path = log_dir / name
        if not path.exists():
            continue
        print(f"===== {path} =====")
        try:
            for line in path.read_text(errors="replace").splitlines()[-80:]:
                print(line)
        except OSError as exc:
            print(f"could not read {path}: {exc}")
