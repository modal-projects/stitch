"""Trainer-specific helpers for the slime_disagg example.

Thin wrappers over the shared launch spine: Ray-cluster/sidecar/process helpers
come from :mod:`cookbook.ray_cluster` / :mod:`cookbook.sidecar_process`, and
config-prep / train-command / smoke-check come from :mod:`cookbook.trainer_helpers`.
This module only supplies the slime-specific axes: the sidecar module path, the
config-field tuple to materialize, the model-script attribute, and that the
rollout pool runs with a warm floor (not scale-from-zero).
"""

from __future__ import annotations

import subprocess
from typing import Any

from cookbook.slime_disagg.configs.base import YAML_CONFIG_FIELDS

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
from cookbook.trainer_helpers import (  # noqa: F401
    VersionAheadError,
    build_train_cmd as _build_train_cmd,
    prepare_config,
    smoke_flash_pool as _smoke_flash_pool,
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


def prepare_slime_config(slime_cfg: Any, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    prepare_config(slime_cfg, tmpdir, YAML_CONFIG_FIELDS)


def build_train_cmd(slime_cfg: Any, slime_root: str) -> str:
    """Build the training command, sourcing slime's model arch args if needed."""
    return _build_train_cmd(slime_cfg, slime_root, model_script_attr="slime_model_script")


def smoke_flash_pool(
    *,
    app_name: str,
    cls_name: str,
    model_name: str,
    weight_version: int,
    expect_min_containers: int,
    timeout_seconds: int,
) -> None:
    """Smoke the warm-floor slime rollout pool (min_containers > 0)."""
    _smoke_flash_pool(
        app_name=app_name,
        cls_name=cls_name,
        model_name=model_name,
        weight_version=weight_version,
        expect_min_containers=expect_min_containers,
        timeout_seconds=timeout_seconds,
        wake_on_demand=False,
    )
