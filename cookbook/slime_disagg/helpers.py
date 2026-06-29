"""Trainer-specific helpers for the slime_disagg example.

Ray cluster, sidecar, and process helpers are shared across trainers via
:mod:`cookbook.ray_cluster` and :mod:`cookbook.sidecar_process`. This module
provides slime-specific config preparation, train command building, and the
Flash pool smoke check.
"""

from __future__ import annotations

import json
import os
import shlex
import time
import urllib.error
import urllib.request
from typing import Any

from stitch.providers.modal import discover_flash_targets, resolve_flash_gateway_url

# Re-export shared helpers so existing callers (modal_train.py) don't break.
from cookbook.ray_cluster import (  # noqa: F401
    RAY_START_TIMEOUT,
    RAY_WORKER_JOIN_TIMEOUT,
    get_modal_cluster_context,
    start_ray_head,
    start_ray_worker,
    training_nodes,
)
from cookbook.sidecar_process import (  # noqa: F401
    terminate_process,
    wait_http,
)


SIDECAR_MODULE = "cookbook.slime_disagg.sidecar"


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
) -> "subprocess.Popen":
    import subprocess

    from cookbook.sidecar_process import start_sglang_sidecar as _start

    return _start(
        sidecar_module=SIDECAR_MODULE,
        sidecar_port=sidecar_port,
        sglang_port=sglang_port,
        bulletin_root=bulletin_root,
        local_checkpoint_dir=local_checkpoint_dir,
        base_checkpoint_dir=base_checkpoint_dir,
        volume_name=volume_name,
        commit_mode=commit_mode,
        debug_requests=debug_requests,
    )


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
