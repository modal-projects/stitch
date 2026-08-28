"""Standalone Qwen3-0.6B rollout pool: a small open model for cheap end-to-end runs.

Three pinned replicas behind the offline-evals router (version-pinned placement
plus the registry-driven rollout control loop), cpu delta updates so a publish
is a small host-RAM staging step rather than a full reload. Use this config to
exercise the whole launch/publish/eval flow without waiting on a large model.
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import CHECKPOINTS_PATH

APP_NAME = "stitch-standalone-qwen3-0p6b"
EXPERIMENT_VOLUME_NAME = "stitch-standalone-qwen3-0p6b"
CHECKPOINT_VOLUME_NAME = "inference-checkpoints"
LOCAL_CHECKPOINT_PATH = None

SOURCE_MODEL = "Qwen/Qwen3-0.6B"
SOURCE_REVISION = "main"
BASE_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-0p6b"

ROLLOUT_GPUS_PER_ENGINE = 1
ROLLOUT_INPUTS_PER_ENGINE = 8

SIDECAR_COMMIT_MODE = "in_place"
SIDECAR_FLUSH_CACHE_ON_COMMIT = True
# Opt into the offline-evals router wiring (cookbook/standalone/offline_evals/).
OFFLINE_EVALS = True
# The registry owns reconciliation timing: replicas never self-reconcile on the
# sidecar's poll, only on the registry's wake.
SIDECAR_RECONCILE_INTERVAL = 365 * 24 * 3600.0
SGLANG_DELTA_UPDATE_MODE = "cpu"

SGLANG_SERVER_ARGS = {
    "--tp": str(ROLLOUT_GPUS_PER_ENGINE),
    "--load-format": "auto",
    "--dtype": "auto",
    "--context-length": "8192",
    # cpu delta updates stage incoming weights in host RAM.
    "--enable-cpu-weight-cache": "",
    "--cpu-weight-cache-max-compile-group-gb": "8",
}

modal = ModalConfig(
    gpu="L40S",
    routing_region="us-west",
    rollout_min_containers=3,
    rollout_max_containers=3,
    # Doubles as the consolidation capacity unit: the router consolidates
    # replicas only while the survivors can absorb this many inputs each.
    rollout_target_inputs=ROLLOUT_INPUTS_PER_ENGINE,
    rollout_ephemeral_disk_mib=512 * 1024,
    rollout_memory_mib=(64 * 1024, 128 * 1024),
    router_registry_min_containers=1,
    router_min_containers=1,
)
