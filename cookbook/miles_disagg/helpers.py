"""Miles-specific wiring for the shared disagg launch spine.

The constants below are the axes where miles differs from slime; everything
else is re-exported unchanged from the shared cookbook modules. This module
also owns the two helpers only miles needs: the node-local YAML materializer
(for ``te_precision_config_file``) and the host-RAM monitor.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from typing import Any

from cookbook.miles_disagg.configs.base import YAML_CONFIG_FIELDS

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
    start_sglang_sidecar as _start_sidecar,
    terminate_process,
    wait_http,
)
from cookbook.trainer_helpers import (  # noqa: F401
    VersionAheadError,
    build_train_cmd as _build_train_cmd,
    prepare_config,
    smoke_flash_pool as _smoke_flash_pool,
)


SIDECAR_MODULE = "cookbook.sidecar"  # `python3 -m` entry on each rollout replica
MODEL_SCRIPT_ATTR = "miles_model_script"  # config attr naming the sourced MODEL_ARGS script
WAKE_ON_DEMAND = True  # scale-from-zero rollout pool (the smoke completion wakes it)


def start_sglang_sidecar(**kwargs: Any) -> subprocess.Popen:
    """Launch the sidecar (`python3 -m` on this recipe's sidecar module)."""
    return _start_sidecar(sidecar_module=SIDECAR_MODULE, **kwargs)


def prepare_miles_config(miles_cfg: Any, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs.

    hf_checkpoint / ref_load already point at prepared absolute paths (the served
    NVFP4 base and bf16 masters), so the ``startswith("/")`` guard skips them; a
    repo-id-shaped value (if any) is snapshot-downloaded from the HF cache.
    """
    prepare_config(miles_cfg, tmpdir, YAML_CONFIG_FIELDS)


def build_train_cmd(miles_cfg: Any, miles_root: str) -> str:
    """Build the training command, sourcing miles' model arch args if needed."""
    return _build_train_cmd(miles_cfg, miles_root, model_script_attr=MODEL_SCRIPT_ATTR)


def smoke_flash_pool(**kwargs: Any) -> None:
    """Smoke the scale-from-zero miles rollout pool: the completion wakes it,
    then each warmed container is confirmed at the version."""
    _smoke_flash_pool(wake_on_demand=WAKE_ON_DEMAND, **kwargs)


def _apply_git_patches(patch_paths: list[str], repo_dir: str, label: str, runtime_name: str) -> None:
    for patch_path in patch_paths:
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"{runtime_name} not found: {patch_path}")

        check = subprocess.run(
            ["git", "-C", repo_dir, "apply", "--check", patch_path],
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            subprocess.run(["git", "-C", repo_dir, "apply", patch_path], check=True)
            print(f"[{label}] applied {patch_path}", flush=True)
            continue

        reverse = subprocess.run(
            ["git", "-C", repo_dir, "apply", "--reverse", "--check", patch_path],
            capture_output=True,
            text=True,
        )
        if reverse.returncode == 0:
            print(f"[{label}] already applied {patch_path}", flush=True)
            continue

        raise RuntimeError(
            f"Cannot apply {runtime_name} {patch_path}\n"
            f"apply --check stderr:\n{check.stderr}\n"
            f"reverse --check stderr:\n{reverse.stderr}"
        )


def apply_sglang_runtime_patches(patch_paths: list[str], repo_dir: str = "/sgl-workspace/sglang") -> None:
    """Apply git patches to the runtime SGLang checkout before server start."""
    _apply_git_patches(patch_paths, repo_dir, "SGLang patch", "SGLang runtime patch")


def apply_megatron_runtime_patches(patch_paths: list[str], repo_dir: str = "/root/Megatron-LM") -> None:
    """Apply git patches to the runtime Megatron checkout before trainer start."""
    _apply_git_patches(patch_paths, repo_dir, "Megatron patch", "Megatron runtime patch")


def materialize_node_local_yaml(cfg: Any, field: str, dest_dir: str = "/root/.miles_node_yaml") -> None:
    """Materialize a per-actor-read YAML config to a deterministic node-local path.

    Some config files (notably ``te_precision_config_file``, which
    ``load_quantization_recipe`` re-reads on every Ray actor during model build)
    are read independently on each trainer node — not just parsed once on the head.
    ``prepare_config`` writes them under ``tempfile.mkdtemp()`` on the head only,
    so on a multi-node cluster the other containers can't see that path.

    Call this on EVERY node (SPMD train()), before the rank-0 gate: each node
    writes identical content (from the shared payload) to the same fixed path, so
    the path the head embeds in the args resolves locally on all actors. No volume
    commit/reload race — Ray actors are long-lived and wouldn't see post-start
    volume writes anyway.
    """
    import yaml

    if isinstance(val := getattr(cfg, field, None), dict):
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{field}.yaml")
        with open(path, "w") as f:
            yaml.dump(val, f)
        setattr(cfg, field, path)


def start_host_mem_monitor(interval_s: int = 20) -> None:
    """Log this node's host-RAM trajectory to stdout from a daemon thread.

    The trainer can OOM-kill on host-RAM exhaustion (the publish/update_weights
    full-model gather is the peak consumer), but Megatron only reports GPU memory
    and the kill leaves no durable peak behind. This logs MemTotal/MemAvailable +
    the container cgroup usage every ``interval_s`` so a live ``modal app logs -f``
    shows exactly which phase blows a big node and how high it peaks. Runs on EVERY
    node (called from the SPMD enter()), so whichever rank OOMs has its own trace.
    Best-effort: never raises."""
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
