"""Qwen3-4B rollout fleet split across two regions, GORGO-routed — the serving-only
geo benchmark (Part A).

Same serving stack as the training recipe (image, sglang args, concurrency), but the
fleet is one H200 replica per region (Server -> us-east, ServerB -> us-west, both
pinned min=max=1) with the Router co-located in us-east. No trainer runs against this
app — traffic comes from ``tools/probes`` replaying GSM8K prompts in GRPO shape while
``tools/probes/router_bench.py`` alternates the routing policy. The RTT asymmetry
(router -> us-east ~0, router -> us-west ~60-80ms) is what exercises gorgo's rtt term.
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.slime_disagg.configs import qwen3_4b_delta_flash as _base


APP_NAME = "stitch-qwen3-4b-geo"
DELTA_VOLUME_NAME = "stitch-delta-qwen3-4b-geo"
DELTA_BULLETIN_ROOT = _base.DELTA_BULLETIN_ROOT
LOCAL_CHECKPOINT_PATH = _base.LOCAL_CHECKPOINT_PATH
SIDECAR_COMMIT_MODE = _base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = _base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_SERVER_ARGS = _base.SGLANG_SERVER_ARGS

modal = ModalConfig(
    gpu="H200",
    rollout_regions=["us-east", "us-west"],
    rollout_min_containers=1,  # per region class -> fleet of exactly 2, one per region
    rollout_max_containers=1,
    router_enabled=True,
    # The bench driver flips policies / enables the tuner on its schedule; start
    # deterministic: affinity routing, tuner off, default gorgo weights.
    router_policy="session-affinity",
    router_tuner=None,
)

slime = _base.slime
