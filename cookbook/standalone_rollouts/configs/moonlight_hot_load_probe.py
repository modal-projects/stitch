"""Scratch probe variant of moonlight_hot_load (not for check-in): scratch app
name + transport prefix, one replica. Deploy with PROVIDER_CONFIG=moonlight_hot_load_probe."""

from cookbook.standalone_rollouts.configs.moonlight_hot_load import *  # noqa: F401,F403

APP_NAME = "stitch-moonlight-api-shim-probe"
S3_TRANSPORT_KEY_PREFIX = "standalone-rollouts/_probe-review-fixes-live"
ROLLOUT_MIN_CONTAINERS = 1
