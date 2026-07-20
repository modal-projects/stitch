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
    # Pinned fleet: the A/B benchmark must not have Flash autoscaling change the
    # replica count mid-run.
    rollout_min_containers=2,
    rollout_max_containers=2,
    router_enabled=True,
    # The bench driver (tools/probes/router_bench.py) owns the policy schedule and
    # the tuner segment; start deterministic. (Tune on E2E when enabling: the
    # sidecar buffers responses, so router-observed TTFT == E2E.)
    router_policy="session-affinity",
    router_tuner=None,
)


class _Slime(_base._Slime):
    # The base recipe is a 3-step smoke; the benchmark needs rollouts flowing for
    # the whole ~50-min block schedule.
    num_rollout = 30


slime = _Slime()
