"""Slime-specific wiring for the shared disagg launch spine.

The constants below are the axes where slime differs from miles; everything
else is re-exported unchanged from the shared cookbook modules.
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
MODEL_SCRIPT_ATTR = "slime_model_script"  # config attr naming the sourced MODEL_ARGS script
WAKE_ON_DEMAND = False  # warm-floor rollout pool (min_containers > 0)


def start_sglang_sidecar(**kwargs: Any) -> subprocess.Popen:
    """Launch the sidecar (`python3 -m` on this recipe's sidecar module)."""
    return _start_sidecar(sidecar_module=SIDECAR_MODULE, **kwargs)


def prepare_slime_config(slime_cfg: Any, tmpdir: str) -> None:
    """Resolve HF repo IDs to local paths and materialize inline YAML configs."""
    prepare_config(slime_cfg, tmpdir, YAML_CONFIG_FIELDS)


def build_train_cmd(slime_cfg: Any, slime_root: str) -> str:
    """Build the training command, sourcing slime's model arch args if needed."""
    return _build_train_cmd(slime_cfg, slime_root, model_script_attr=MODEL_SCRIPT_ATTR)


def smoke_flash_pool(**kwargs: Any) -> None:
    """Smoke the warm-floor slime rollout pool."""
    _smoke_flash_pool(wake_on_demand=WAKE_ON_DEMAND, **kwargs)
