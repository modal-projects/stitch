"""Trainer-specific helpers for the miles_disagg example.

Ray cluster, sidecar, and process helpers are shared across trainers via
:mod:`cookbook.ray_cluster` and :mod:`cookbook.sidecar_process`. This module
provides miles-specific config preparation, train command building, and the
Flash pool smoke check.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
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


SIDECAR_MODULE = "cookbook.miles_disagg.sidecar"


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


def start_host_mem_monitor(interval_s: int = 20) -> None:
    """Log this node's host-RAM trajectory to stdout from a daemon thread.

    The trainer can OOM-kill on host-RAM exhaustion (the publish/update_weights
    full-model gather is the peak consumer), but Megatron only reports GPU memory
    and the kill leaves no durable peak behind. This logs MemTotal/MemAvailable +
    the container cgroup usage every ``interval_s`` so a live ``modal app logs -f``
    shows exactly which phase blows the ~1.95 TiB B200:8 node and how high it peaks.
    Runs on EVERY node (called from the SPMD enter()), so whichever rank OOMs has
    its own trace. Best-effort: never raises."""
    import threading

    host = socket.gethostname()

    def _meminfo() -> tuple[float, float]:
        total = avail = 0.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) / 1024 / 1024  # GiB
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) / 1024 / 1024
        except Exception:  # noqa: BLE001
            pass
        return total, avail

    def _cgroup_used_gib() -> float:
        for path in ("/sys/fs/cgroup/memory.current",  # cgroup v2
                     "/sys/fs/cgroup/memory/memory.usage_in_bytes"):  # v1
            try:
                with open(path) as f:
                    return int(f.read().strip()) / 1024**3
            except Exception:  # noqa: BLE001
                continue
        return -1.0

    def _loop() -> None:
        while True:
            total, avail = _meminfo()
            used = total - avail
            cg = _cgroup_used_gib()
            print(
                f"[hostmem] {host} used={used:.0f}GiB avail={avail:.0f}GiB "
                f"total={total:.0f}GiB cgroup_used={cg:.0f}GiB",
                flush=True,
            )
            time.sleep(interval_s)

    threading.Thread(target=_loop, daemon=True, name="host-mem-monitor").start()
