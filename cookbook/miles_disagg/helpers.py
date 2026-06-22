"""Ray cluster, sidecar process, and smoke-check helpers for the miles example.

The miles twin of cookbook/slime_disagg/helpers.py. The Ray bring-up, sidecar
launch, and Flash-pool smoke check are trainer-agnostic and identical; only
``prepare_miles_config`` / ``build_train_cmd`` (which source the miles model
script and run miles' train.py/train_async.py) and the sidecar module path
differ. Ray helpers were originally adapted from
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


# ── miles launch ──────────────────────────────────────────────────────────────


def prepare_miles_config(miles_cfg: Any, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs.

    hf_checkpoint / ref_load already point at prepared absolute paths (the served
    NVFP4 base and bf16 masters), so the ``startswith("/")`` guard skips them; a
    repo-id-shaped value (if any) is snapshot-downloaded from the HF cache.
    """
    from huggingface_hub import snapshot_download
    import yaml

    from cookbook.miles_disagg.configs.base import YAML_CONFIG_FIELDS

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(miles_cfg, attr, None)) and not str(val).startswith("/"):
            setattr(miles_cfg, attr, snapshot_download(val, local_files_only=True))

    for field in YAML_CONFIG_FIELDS:
        if isinstance(val := getattr(miles_cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            setattr(miles_cfg, field, path)


def materialize_node_local_yaml(miles_cfg: Any, field: str, dest_dir: str = "/root/.miles_node_yaml") -> None:
    """Materialize a per-actor-read YAML config to a deterministic node-local path.

    Some config files (notably ``te_precision_config_file``, which
    ``load_quantization_recipe`` re-reads on every Ray actor during model build)
    are read independently on each trainer node — not just parsed once on the head.
    ``prepare_miles_config`` writes them under ``tempfile.mkdtemp()`` on the head
    only, so on a multi-node cluster the other containers can't see that path.

    Call this on EVERY node (SPMD train()), before the rank-0 gate: each node
    writes identical content (from the shared payload) to the same fixed path, so
    the path the head embeds in the args resolves locally on all actors. No volume
    commit/reload race — Ray actors are long-lived and wouldn't see post-start
    volume writes anyway.
    """
    import yaml

    if isinstance(val := getattr(miles_cfg, field, None), dict):
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{field}.yaml")
        with open(path, "w") as f:
            yaml.dump(val, f)
        setattr(miles_cfg, field, path)


def build_train_cmd(miles_cfg: Any, miles_root: str) -> str:
    """Build the training command, sourcing model arch args if needed.

    miles' train.py / train_async.py live at the repo root and consume the
    ``MODEL_ARGS`` bash array defined by the sourced model script, exactly like
    slime's launcher.
    """
    train_script = f"{miles_root}/{'train_async.py' if miles_cfg.async_mode else 'train.py'}"
    if miles_cfg.miles_model_script:
        inner = (
            f"source {miles_root}/{miles_cfg.miles_model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(miles_cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(miles_cfg.cli_args())}"


# ── Sidecar process ───────────────────────────────────────────────────────────


def start_sglang_sidecar(
    *,
    sidecar_port: int,
    sglang_port: int,
    bulletin_root: str,
    local_checkpoint_dir: str,
    base_checkpoint_dir: str,
    volume_name: str,
    commit_mode: str,
    debug_requests: bool = False,
) -> subprocess.Popen:
    cmd = [
        "python3",
        "-m",
        "cookbook.miles_disagg.sidecar",
        "--host",
        "0.0.0.0",
        "--port",
        str(sidecar_port),
        "--upstream-url",
        f"http://127.0.0.1:{sglang_port}",
        "--bulletin-root",
        bulletin_root,
        "--local-checkpoint-dir",
        local_checkpoint_dir,
        "--base-checkpoint-dir",
        base_checkpoint_dir,
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
    """Wake the elastic pool on demand and confirm it serves at the expected
    weight version.

    The pool has no warm floor (min_containers=0), so a completion sent to the
    Flash gateway is what scales it 0->1; Flash holds the request during the
    container's cold start (model load + FP4 kernel tuning), so the warmup uses a
    generous timeout. ``expect_min_containers`` is advisory only — a value > 0
    just means "also confirm at least one direct container reports the version."
    """
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while True:
        try:
            _check_flash_pool_once(app_name, cls_name, model_name, weight_version)
            return
        except VersionAheadError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if time.time() >= deadline:
            raise TimeoutError(f"Flash pool smoke did not pass before timeout: {last_error}")
        print(f"Waiting for Flash pool to wake/serve: {last_error}")
        time.sleep(10)


def _check_flash_pool_once(app_name: str, cls_name: str, model_name: str, expected: int) -> None:
    gateway = resolve_flash_gateway_url(app_name, cls_name)
    print(f"Gateway URL: {gateway}")

    # Wake the (scaled-to-zero) pool and wait for a container to serve. Flash
    # holds the request through the cold start, so the timeout must exceed it.
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "weight_version": {"exact_version": expected},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = _post_json(f"{gateway}/v1/chat/completions", payload, timeout=900)
    print(f"Gateway completion: {data}")
    start, end = int(data.get("weight_version_start", -1)), int(data.get("weight_version_end", -1))
    if start > expected or end > expected:
        raise VersionAheadError(f"gateway served version {start}->{end}, already past expected {expected}")
    if start != expected or end != expected:
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")

    # The pool is warm now; confirm each live container reports the version.
    targets = discover_flash_targets(app_name, cls_name)
    print(f"Direct container URLs ({len(targets)}):")
    for target in [gateway, *targets]:
        info = _get_json(f"{target}/server_info", timeout=30)
        print(f"{target} server_info={info}")
        current = int(info["current_version"])
        if current > expected:
            raise VersionAheadError(f"{target} current_version={current} already passed expected {expected}")
        if current != expected:
            raise RuntimeError(f"{target} current_version={current} expected {expected}")


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
