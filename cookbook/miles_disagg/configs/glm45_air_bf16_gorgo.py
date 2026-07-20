"""GLM-4.5-Air bf16 with GORGO-routed rollouts.

Same training run as ``glm45_air_bf16``, but rollout traffic goes through the
stitch Router (prefix-cache/load/RTT-aware gorgo policy, online-ES tuned on
p95 E2E) instead of the Flash gateway. A/B against the base config — or against
``router_policy="session-affinity"`` to isolate the policy from the extra hop.
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.miles_disagg.configs import glm45_air_bf16 as _base


APP_NAME = "stitch-glm45-air-bf16-gorgo"
DELTA_VOLUME_NAME = "stitch-delta-glm45-air-bf16-gorgo"
DELTA_BULLETIN_ROOT = _base.DELTA_BULLETIN_ROOT
LOCAL_CHECKPOINT_PATH = _base.LOCAL_CHECKPOINT_PATH
SIDECAR_COMMIT_MODE = _base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = _base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_SERVER_ARGS = _base.SGLANG_SERVER_ARGS

modal = ModalConfig(
    gpu="H200",
    region="us",
    memory=1_048_576,
    rollout_min_containers=2,  # load balancing needs >1 replica to matter
    rollout_target_inputs=32,
    proxy_regions=["us-west"],
    rollout_ephemeral_disk_mib=819_200,
    torch_dist_prep_nodes=4,
    torch_dist_prep_gpus_per_node=8,
    torch_dist_convert_extra_args=(
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 4 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--decoder-last-pipeline-num-layers 10"
    ),
    torch_dist_prep_ephemeral_disk_mib=819_200,
    router_enabled=True,
    router_policy="gorgo",
    # The sidecar buffers responses, so router-observed TTFT == E2E; tune on E2E.
    router_tuner={"enabled": True, "objective_metric": "neg_p95_e2e"},
)

miles = _base.miles
