"""Qwen3-4B rollout provider config for the hot-load API shim."""

from __future__ import annotations

from pathlib import Path


APP_NAME = "stitch-qwen3-4b-api-shim"
MODEL_NAME = "Qwen/Qwen3-4B"

HF_SECRET_NAME = "huggingface-secret"
SHIM_SECRET_NAME = "stitch-api-shim-provider"
HF_CACHE_VOLUME_NAME = "huggingface-cache"

HF_CACHE_PATH = Path("/root/.cache/huggingface")
# Ephemeral host-local full HF checkpoint the sidecar patches in place per delta
# (seeded from the base; rebuilt on a cold container).
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
# How the rollout sidecar applies versions. in_place pauses/applies/continues
# without flushing (relies on the engine overlap-drain fix); quiesce is the safe
# fallback. See docs/kv-version-namespace-design.md.
COMMIT_MODE = "in_place"
S3_TRANSPORT_BUCKET_NAME = "modal-stitch-s3-transport"
S3_TRANSPORT_KEY_PREFIX = "standalone-rollouts/qwen3-4b"
S3_TRANSPORT_MOUNT_PATH = Path("/mnt/stitch-s3-transport")
S3_TRANSPORT_REGION = None
S3_TRANSPORT_OIDC_AUTH_ROLE_ARN = (
    "arn:aws:iam::459781239556:role/modal-buckets/stitch-s3-transport-role"
)

GPU = "H200"
CLOUD = None
REGION = None
PROXY_REGIONS = ["us-east"]
ROLLOUT_MIN_CONTAINERS = 2
ROLLOUT_NUM_GPUS_PER_ENGINE = 1
ROLLOUT_CONCURRENCY = 64

# The disk-delta-weight-sync branch applies deltas host-side and reloads via the
# ordinary update_weights_from_disk path, so the old engine-side delta server
# args (--update-weight-delta-chunk-bytes/--update-weight-delta-read-workers) no
# longer exist and must not be passed.
SGLANG_SERVER_ARGS = {
    "--reasoning-parser": "qwen3",
    "--context-length": "16384",
    "--mem-fraction-static": "0.84",
    "--chunked-prefill-size": "4096",
    "--max-prefill-tokens": "4096",
}
