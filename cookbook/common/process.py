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

SIDECAR_MODULE = "cookbook.common.sidecar"  # the one shared serve() entrypoint


def start_sidecar(
    *, sidecar_port: int, sglang_port: int, bulletin_root: str, local_checkpoint_dir: str,
    volume_name: str, commit_mode: str, debug_requests: bool = False,
) -> subprocess.Popen:
    """Launch the versioned rollout proxy (the shared sidecar) beside sglang."""
    cmd = [
        "python3", "-m", SIDECAR_MODULE,
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


def start_host_mem_monitor(interval_s: int = 20) -> None:
    """Trace this node's host-RAM from a daemon thread. Host-RAM exhaustion OOM-kills the
    trainer (the peak is the publish weight-gather) and Modal exposes no host-RAM metric, so
    this log line is the only signal. Best-effort."""
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
