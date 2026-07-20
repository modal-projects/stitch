"""Qwen3-4B sparse-delta run with GORGO-routed rollouts.

Same training run as ``qwen3_4b_delta_flash``, but rollout traffic goes through
the stitch Router (prefix-cache/load/RTT-aware gorgo policy, online-ES tuned on
p95 E2E) instead of the Flash gateway. A/B against the base config — or against
``router_policy="session-affinity"`` to isolate the policy from the extra hop.
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.slime_disagg.configs import qwen3_4b_delta_flash as _base


APP_NAME = "stitch-qwen3-4b-gorgo"
DELTA_VOLUME_NAME = "stitch-delta-qwen3-4b-gorgo"
DELTA_BULLETIN_ROOT = _base.DELTA_BULLETIN_ROOT
LOCAL_CHECKPOINT_PATH = _base.LOCAL_CHECKPOINT_PATH
SIDECAR_COMMIT_MODE = _base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = _base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_SERVER_ARGS = _base.SGLANG_SERVER_ARGS

modal = ModalConfig(
    gpu="H200",
    router_enabled=True,
    router_policy="gorgo",
    # The sidecar buffers responses, so router-observed TTFT == E2E; tune on E2E.
    router_tuner={"enabled": True, "objective_metric": "neg_p95_e2e"},
)

slime = _base.slime
