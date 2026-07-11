"""Moonlight-16B-A3B rollout provider config for the hot-load API shim.

Moonlight is a small DeepSeek-V3-architecture (MLA + DeepSeek-MoE) model — the
Kimi K2.6 family at a size that fits one H200. The pool serves rollouts and emits
per-token routed experts so the trainer can replay them (routing replay).
"""

from __future__ import annotations

from pathlib import Path


APP_NAME = "stitch-moonlight-api-shim"
MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
# The checkpoint the pool boots from and seeds every delta onto. An HF repo id
# resolves from the prewarmed hub cache; an absolute path is an S3-mounted or
# prep-volume base the customer pre-uploaded. Here it is the public base model;
# a customer running a finetune of the same arch/quant points this at their own
# base dir instead. MODEL_NAME stays the served-model label clients send.
BASE_CHECKPOINT = MODEL_NAME

HF_SECRET_NAME = "huggingface-secret"
SHIM_SECRET_NAME = "stitch-api-shim-provider"
HF_CACHE_VOLUME_NAME = "huggingface-cache"

HF_CACHE_PATH = Path("/root/.cache/huggingface")
# Ephemeral host-local full HF checkpoint the sidecar patches in place per delta
# (seeded from the base; rebuilt on a cold container).
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
# in_place pauses/applies/continues without flushing; stale-version KV is isolated
# by the sidecar's extra_key stamping and drains as its in-flight requests finish.
COMMIT_MODE = "in_place"
S3_TRANSPORT_BUCKET_NAME = "modal-stitch-s3-transport"
S3_TRANSPORT_KEY_PREFIX = "standalone-rollouts/moonlight"
S3_TRANSPORT_MOUNT_PATH = Path("/mnt/stitch-s3-transport")
S3_TRANSPORT_REGION = None
S3_TRANSPORT_OIDC_AUTH_ROLE_ARN = (
    "arn:aws:iam::459781239556:role/modal-buckets/stitch-s3-transport-role"
)

GPU = "H200"
CLOUD = None
# Pin the rollout pool (and the front door) to the US so they stay co-located.
REGION = "us"
# Region inputs are routed through. The front door's `routing_region` and the
# rollout pool's Flash `proxy_regions` are kept identical so customer traffic and
# the pool share one entry region.
ROUTING_REGION = "us-east"
PROXY_REGIONS = [ROUTING_REGION]
ROLLOUT_MIN_CONTAINERS = 4
ROLLOUT_NUM_GPUS_PER_ENGINE = 1  # Moonlight (~32 GB bf16, tiny MLA KV) fits 1xH200
ROLLOUT_CONCURRENCY = 32

# Moonlight is DeepSeek-V3-arch (no qwen reasoning parser). Routing replay needs
# the pool to emit per-token routed experts: the trainer launches no engine in
# publish-only mode, so --enable-return-routed-experts must be set here, not by
# slime's sglang_engine. mem-fraction-static is a starting point — measure it.
SGLANG_SERVER_ARGS = {
    "--context-length": "8192",
    "--mem-fraction-static": "0.85",
    "--enable-return-routed-experts": "",
}
