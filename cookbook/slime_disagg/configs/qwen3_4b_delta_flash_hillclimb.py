"""Longer Qwen3-4B sparse-delta run for reward hillclimb validation."""

from __future__ import annotations

from cookbook.slime_disagg.configs import qwen3_4b_delta_flash as _base


APP_NAME = "stitch-qwen3-4b-hillclimb"
DELTA_VOLUME_NAME = "stitch-delta-qwen3-4b-hillclimb"
DELTA_BULLETIN_ROOT = _base.DELTA_BULLETIN_ROOT
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
SIDECAR_COMMIT_MODE = _base.SIDECAR_COMMIT_MODE
SGLANG_SERVER_ARGS = _base.SGLANG_SERVER_ARGS

modal = _base.modal


class _Slime(_base._Slime):
    # Longer than the protocol smoke, but still bounded enough to use as a
    # repeatable disaggregated reward-hillclimb check.
    num_rollout = 120
    eval_interval = 20
    log_passrate = True
    # update_weight_disk_dir is inherited from the base config (same Volume mount
    # path); only DELTA_VOLUME_NAME differs, so this app owns its own bulletin.


slime = _Slime()
