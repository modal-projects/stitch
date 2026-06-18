"""Ray cluster, sidecar process, and smoke-check helpers for the example.

Ray helpers were originally adapted from
https://github.com/modal-projects/multinode-training-guide.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from stitch.protocol import write_latest
from stitch.providers.modal import discover_flash_targets, resolve_flash_gateway_url

RAY_START_TIMEOUT = 240
RAY_WORKER_JOIN_TIMEOUT = 180


# ── Ray cluster ───────────────────────────────────────────────────────────────


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


# ── SLIME launch ──────────────────────────────────────────────────────────────


def prepare_slime_config(slime_cfg: Any, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    from huggingface_hub import snapshot_download
    import yaml

    from cookbook.slime_disagg.configs.base import YAML_CONFIG_FIELDS

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(slime_cfg, attr, None)) and not str(val).startswith("/"):
            setattr(slime_cfg, attr, snapshot_download(val, local_files_only=True))

    for field in YAML_CONFIG_FIELDS:
        if isinstance(val := getattr(slime_cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            setattr(slime_cfg, field, path)


def build_train_cmd(slime_cfg: Any, slime_root: str) -> str:
    """Build the training command, sourcing model arch args if needed."""
    train_script = f"{slime_root}/{'train_async.py' if slime_cfg.async_mode else 'train.py'}"
    if slime_cfg.slime_model_script:
        inner = (
            f"source {slime_root}/{slime_cfg.slime_model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(slime_cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(slime_cfg.cli_args())}"


# ── Sidecar process ───────────────────────────────────────────────────────────


def start_sglang_sidecar(
    *,
    sidecar_port: int,
    sglang_port: int,
    bulletin_root: str,
    volume_name: str,
    commit_mode: str,
    debug_requests: bool = False,
) -> subprocess.Popen:
    cmd = [
        "python3",
        "-m",
        "cookbook.slime_disagg.sidecar",
        "--host",
        "0.0.0.0",
        "--port",
        str(sidecar_port),
        "--upstream-url",
        f"http://127.0.0.1:{sglang_port}",
        "--bulletin-root",
        bulletin_root,
        "--volume-name",
        volume_name,
        "--commit-mode",
        commit_mode,
    ]
    if debug_requests:
        cmd.append("--debug-requests")
    print("Starting sidecar:", " ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def wait_http(url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"process exited while waiting for {url}: code={process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}; last error: {last_error}")


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


# ── Flash pool smoke check ────────────────────────────────────────────────────


class VersionAheadError(RuntimeError):
    """Raised when a monotonic rollout pool has already advanced past a smoke version."""


def smoke_flash_pool(
    *,
    app_name: str,
    cls_name: str,
    model_name: str,
    weight_version: int,
    expect_min_containers: int,
    timeout_seconds: int,
) -> None:
    """Poll the Flash gateway and direct container URLs until the pool serves
    completions at the expected weight version."""
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while True:
        gateway = resolve_flash_gateway_url(app_name, cls_name)
        targets = discover_flash_targets(app_name, cls_name)
        if len(targets) < expect_min_containers:
            last_error = f"expected at least {expect_min_containers} containers, found {len(targets)}: {targets}"
        else:
            try:
                _check_flash_pool_once(gateway, targets, model_name, weight_version)
                return
            except VersionAheadError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
        if time.time() >= deadline:
            raise TimeoutError(f"Flash pool smoke did not pass before timeout: {last_error}")
        print(f"Waiting for Flash pool readiness: {last_error}")
        time.sleep(10)


def _check_flash_pool_once(gateway: str, targets: list[str], model_name: str, expected: int) -> None:
    print(f"Gateway URL: {gateway}")
    print(f"Direct container URLs ({len(targets)}):")
    for target in targets:
        print(f"  {target}")

    for target in [gateway, *targets]:
        info = _get_json(f"{target}/server_info", timeout=30)
        print(f"{target} server_info={info}")
        current = int(info["current_version"])
        if current > expected:
            raise VersionAheadError(f"{target} current_version={current} already passed expected {expected}")
        if current != expected:
            raise RuntimeError(f"{target} current_version={current} expected {expected}")

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "weight_version": {"exact_version": expected},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = _post_json(f"{gateway}/v1/chat/completions", payload, timeout=180)
    print(f"Gateway completion: {data}")
    if int(data.get("weight_version_start", -1)) != expected or int(data.get("weight_version_end", -1)) != expected:
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")


def _get_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.load(resp)


# ── Bulletin board ────────────────────────────────────────────────────────────


def reset_bulletin_board(root: str | Path, volume: Any, *, confirm: bool = False) -> None:
    if not confirm:
        raise ValueError("Pass --confirm to clear retained sparse-delta versions.")

    import shutil

    root = Path(root)
    shutil.rmtree(root / "versions", ignore_errors=True)
    (root / "versions").mkdir(parents=True, exist_ok=True)
    write_latest(root, 0)
    volume.commit()
