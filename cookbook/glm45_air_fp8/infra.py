"""Non-stitch deployment plumbing for the glm45_air_fp8 example: Ray cluster
bring-up, runtime git patches, a host-RAM monitor, config materialization, the
miles train command, the sidecar subprocess, and the Flash-pool smoke.

None of this is stitch's concern — it is the miles/Megatron/Modal/Ray glue any
disaggregated miles deployment needs. The stitch-facing wiring is in hooks.py
(publish/request) and sidecar.py (serve). Ported from the shared cookbook spine.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from stitch.pools.modal_flash import ModalFlashPool
from stitch.versions import VersionRef

RAY_START_TIMEOUT = 240
RAY_WORKER_JOIN_TIMEOUT = 180


# ── Ray cluster ──────────────────────────────────────────────────────────────
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


# ── Runtime git patches ──────────────────────────────────────────────────────
def apply_git_patches(patch_paths: list[str], repo_dir: str, label: str) -> None:
    """Apply git patches to a runtime checkout, tolerating an already-applied patch
    (idempotent across container restarts)."""
    for patch_path in patch_paths:
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"{label} not found: {patch_path}")
        check = subprocess.run(["git", "-C", repo_dir, "apply", "--check", patch_path], capture_output=True, text=True)
        if check.returncode == 0:
            subprocess.run(["git", "-C", repo_dir, "apply", patch_path], check=True)
            print(f"[{label}] applied {patch_path}", flush=True)
            continue
        reverse = subprocess.run(
            ["git", "-C", repo_dir, "apply", "--reverse", "--check", patch_path], capture_output=True, text=True
        )
        if reverse.returncode == 0:
            print(f"[{label}] already applied {patch_path}", flush=True)
            continue
        raise RuntimeError(f"cannot apply {label} {patch_path}\ncheck: {check.stderr}\nreverse: {reverse.stderr}")


# ── Config materialization ───────────────────────────────────────────────────
def prepare_config(cfg: Any, tmpdir: str, yaml_config_fields: tuple[str, ...]) -> None:
    """Resolve HF repo-id checkpoint fields to local paths and materialize inline
    YAML config dicts to files the trainer reads. Absolute paths are left untouched."""
    from huggingface_hub import snapshot_download
    import yaml

    for attr in ("hf_checkpoint", "load", "ref_load", "critic_load"):
        if (val := getattr(cfg, attr, None)) and not str(val).startswith("/"):
            setattr(cfg, attr, snapshot_download(val, local_files_only=True))
    for field in yaml_config_fields:
        if isinstance(val := getattr(cfg, field, None), dict):
            path = os.path.join(tmpdir, f"{field}.yaml")
            with open(path, "w") as f:
                yaml.dump(val, f)
            setattr(cfg, field, path)


def materialize_node_local_yaml(cfg: Any, field: str, dest_dir: str = "/root/.miles_node_yaml") -> None:
    """Write an inline YAML config to a deterministic node-local path on EVERY node.

    Fields like ``te_precision_config_file`` are re-read on each Ray actor during
    model build, so they must resolve to identical content at an identical path on
    all nodes — ``prepare_config`` only writes them on the head's temp dir."""
    import yaml

    if isinstance(val := getattr(cfg, field, None), dict):
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{field}.yaml")
        with open(path, "w") as f:
            yaml.dump(val, f)
        setattr(cfg, field, path)


def start_host_mem_monitor(interval_s: int = 20) -> None:
    """Log this node's host-RAM trajectory from a daemon thread. The trainer can
    OOM-kill at host-RAM exhaustion (the publish weight-gather is the peak) and Modal
    leaves no peak behind, so this makes `modal app logs -f` show the blow. Best-effort."""
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
        while True:
            total, avail = _meminfo()
            print(f"[hostmem] {host} used={total - avail:.0f}GiB avail={avail:.0f}GiB total={total:.0f}GiB", flush=True)
            time.sleep(interval_s)

    threading.Thread(target=_loop, daemon=True, name="host-mem-monitor").start()


# ── Train command ────────────────────────────────────────────────────────────
def build_train_cmd(cfg: Any, miles_root: str) -> str:
    """The miles train command, sourcing the model-arch MODEL_ARGS script if set.
    ``train_async.py`` / ``train.py`` live at the miles root and consume MODEL_ARGS."""
    train_script = f"{miles_root}/{'train_async.py' if cfg.async_mode else 'train.py'}"
    model_script = getattr(cfg, "miles_model_script", "")
    if model_script:
        inner = (
            f"source {miles_root}/{model_script} && "
            f"python3 {train_script} ${{MODEL_ARGS[@]}} {shlex.join(cfg.cli_args())}"
        )
        return f"bash -c {shlex.quote(inner)}"
    return f"python3 {train_script} {shlex.join(cfg.cli_args())}"


# ── Sidecar subprocess ───────────────────────────────────────────────────────
def start_sidecar(
    *, sidecar_port: int, sglang_port: int, bulletin_root: str, local_checkpoint_dir: str,
    volume_name: str, commit_mode: str, debug_requests: bool = False,
) -> subprocess.Popen:
    """Launch the versioned rollout proxy (cookbook.glm45_air_fp8.sidecar) beside sglang."""
    cmd = [
        "python3", "-m", "cookbook.glm45_air_fp8.sidecar",
        "--host", "0.0.0.0", "--port", str(sidecar_port),
        "--upstream", f"http://127.0.0.1:{sglang_port}",
        "--bulletin-root", bulletin_root,
        "--local-checkpoint-dir", local_checkpoint_dir,
        "--volume-name", volume_name,
        "--commit-mode", commit_mode,
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


# ── Flash-pool smoke ─────────────────────────────────────────────────────────
class VersionAheadError(RuntimeError):
    """A monotonic pool has already advanced past the smoke's expected version."""


def smoke_flash_pool(*, app_name: str, cls_name: str, model_name: str, weight_version: int, timeout_seconds: int) -> None:
    """Poll until the pool serves completions at ``weight_version`` — through the
    gateway (which also wakes a scaled-down pool; Flash holds the request through the
    cold start) and then each live replica's ``/server_info``."""
    pool = ModalFlashPool(app_name, cls_name)
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while True:
        try:
            gateway = pool.gateway_url()
            print(f"Gateway URL: {gateway}")
            data = _post_json(f"{gateway}/v1/chat/completions", _completion(model_name, weight_version), timeout=900)
            print(f"Gateway completion: {data}")
            _check_completion(data, weight_version)
            for target in pool.discover_replicas():
                info = _get_json(f"{target}/server_info", timeout=30)
                print(f"{target} server_info={info}")
                _check_version(_applied_version(info), weight_version, target)
            return
        except VersionAheadError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if time.time() >= deadline:
            raise TimeoutError(f"Flash pool smoke did not pass before timeout: {last_error}")
        print(f"Waiting for Flash pool readiness: {last_error}")
        time.sleep(10)


def _applied_version(info: dict) -> int:
    applied = info.get("applied")
    return VersionRef.parse(applied).version if applied else -1


def _check_version(current: int, expected: int, target: str) -> None:
    if current > expected:
        raise VersionAheadError(f"{target} applied={current} already past expected {expected}")
    if current != expected:
        raise RuntimeError(f"{target} applied={current}, expected {expected}")


def _check_completion(data: dict, expected: int) -> None:
    start, end = int(data.get("weight_version_start", -1)), int(data.get("weight_version_end", -1))
    if start > expected or end > expected:
        raise VersionAheadError(f"gateway served {start}->{end}, already past expected {expected}")
    if start != expected or end != expected:
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")


def _completion(model_name: str, expected: int) -> dict:
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 8,
        "temperature": 0,
        "weight_version": {"exact_version": expected},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _get_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.load(resp)
