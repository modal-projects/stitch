"""Modal Flash helpers for the sparse-delta SLIME example."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from stitch.protocol import write_latest


class VersionAheadError(RuntimeError):
    """Raised when a monotonic rollout pool has already advanced past a smoke version."""


def training_nodes(cfg: Any) -> int:
    nodes = int(getattr(cfg, "actor_num_nodes", 1))
    if getattr(cfg, "use_critic", False) or getattr(cfg, "advantage_estimator", None) == "ppo":
        nodes += int(getattr(cfg, "critic_num_nodes", nodes))
    return nodes


def ensure_pythonpath(cfg: Any, *paths: str) -> None:
    current = str(cfg.environment.get("PYTHONPATH", ""))
    parts = [*paths]
    parts.extend(p for p in current.split(":") if p)
    cfg.environment["PYTHONPATH"] = ":".join(dict.fromkeys(parts))


def redact_command_for_log(command: str) -> str:
    return re.sub(r"(--wandb-key(?:=|\s+))('[^']*'|\"[^\"]*\"|\S+)", r"\1<redacted>", command)


def reset_bulletin_board(root: str | Path, volume: Any, *, confirm: bool = False) -> None:
    if not confirm:
        raise ValueError("Pass --confirm to clear retained sparse-delta versions.")

    import shutil

    root = Path(root)
    shutil.rmtree(root / "versions", ignore_errors=True)
    (root / "versions").mkdir(parents=True, exist_ok=True)
    write_latest(root, 0)
    volume.commit()


def spawn_train_from_deployed_or_ephemeral(
    *,
    deployed_app_name: str,
    experiment: str,
    fallback_function: Any,
    function_name: str = "train",
    environment_name: str | None = None,
) -> Any:
    """Spawn the deployed train function, falling back to the current app."""
    import modal
    from modal.exception import NotFoundError

    try:
        deployed_function = modal.Function.from_name(
            deployed_app_name,
            function_name,
            environment_name=environment_name,
        )
        call = deployed_function.spawn(experiment)
        _print_spawned_call("deployed", deployed_app_name, function_name, experiment, call)
        return call
    except NotFoundError:
        print(
            f"Deployed function {deployed_app_name}.{function_name} was not found; "
            "falling back to this modal run's ephemeral app."
        )

    call = fallback_function.spawn(experiment)
    _print_spawned_call("ephemeral", "current app", function_name, experiment, call)
    return call


def _print_spawned_call(
    source: str,
    app_name: str,
    function_name: str,
    experiment: str,
    call: Any,
) -> None:
    object_id = getattr(call, "object_id", None) or getattr(call, "call_id", None)
    suffix = f" call_id={object_id}" if object_id else ""
    print(f"Spawned {source} {app_name}.{function_name}({experiment!r}){suffix}")


def run_flash_pool_smoke(
    *,
    gateway_resolver: Callable[[], str],
    target_discoverer: Callable[[], list[str]],
    model_name: str,
    weight_version: int = 0,
    expect_min_containers: int = 1,
    timeout_seconds: int = 60,
) -> None:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while True:
        gateway = gateway_resolver()
        targets = target_discoverer()
        if len(targets) < expect_min_containers:
            last_error = f"expected at least {expect_min_containers} containers, found {len(targets)}: {targets}"
        else:
            try:
                check_flash_pool_once(gateway, targets, model_name, weight_version)
                return
            except VersionAheadError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
        if time.time() >= deadline:
            raise TimeoutError(f"Flash pool smoke did not pass before timeout: {last_error}")
        print(f"Waiting for Flash pool readiness: {last_error}")
        time.sleep(10)


def check_flash_pool_once(gateway: str, targets: list[str], model_name: str, weight_version: int) -> None:
    import requests

    expected = int(weight_version)
    print(f"Gateway URL: {gateway}")
    print(f"Direct container URLs ({len(targets)}):")
    for target in targets:
        print(f"  {target}")

    for target in [gateway, *targets]:
        info = requests.get(f"{target}/server_info", timeout=30).json()
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
    response = requests.post(f"{gateway}/v1/chat/completions", json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    print(f"Gateway completion: {data}")
    if int(data.get("weight_version_start", -1)) != expected or int(data.get("weight_version_end", -1)) != expected:
        raise RuntimeError(f"unexpected gateway weight metadata: {data}")


def start_sglang_sidecar(
    *,
    sidecar_port: int,
    sglang_port: int,
    bulletin_root: str,
    volume_name: str,
    pythonpath_root: str = "/root",
) -> subprocess.Popen:
    env = {
        **os.environ,
        "DELTA_BULLETIN_ROOT": bulletin_root,
        "DELTA_VOLUME_NAME": volume_name,
        "PYTHONPATH": prepend_pythonpath(os.environ.get("PYTHONPATH", ""), pythonpath_root),
    }
    cmd = [
        "python3",
        "-m",
        "stitch.servers.sglang",
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
    ]
    print("Starting sidecar:", " ".join(cmd))
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def prepend_pythonpath(current: str, *paths: str) -> str:
    parts = [*paths, *(p for p in current.split(":") if p)]
    return ":".join(dict.fromkeys(parts))


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
