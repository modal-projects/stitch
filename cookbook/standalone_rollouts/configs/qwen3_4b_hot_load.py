"""Qwen3-4B rollout provider config for the hot-load API shim."""

from __future__ import annotations

from pathlib import Path


APP_NAME = "stitch-qwen3-4b-api-shim"
MODEL_NAME = "Qwen/Qwen3-4B"

HF_SECRET_NAME = "huggingface-secret"
SHIM_SECRET_NAME = "stitch-api-shim-provider"
HF_CACHE_VOLUME_NAME = "huggingface-cache"
STATE_DICT_NAME = "stitch-api-shim-qwen3-4b-state"

HF_CACHE_PATH = Path("/root/.cache/huggingface")
SNAPSHOT_ROOT = "/snapshots"
BASE_SNAPSHOT_IDENTITY = "base"
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

SGLANG_UPDATE_WEIGHT_DELTA_CHUNK_BYTES = 1024 * 1024 * 1024
SGLANG_UPDATE_WEIGHT_DELTA_READ_WORKERS = 8
SGLANG_SERVER_ARGS = {
    "--reasoning-parser": "qwen3",
    "--context-length": "16384",
    "--mem-fraction-static": "0.84",
    "--chunked-prefill-size": "4096",
    "--max-prefill-tokens": "4096",
}
