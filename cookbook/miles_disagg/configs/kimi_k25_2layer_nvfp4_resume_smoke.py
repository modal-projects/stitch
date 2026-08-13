"""Temporary same-run resume validation with one abandoned weight version."""

from __future__ import annotations

import copy

from cookbook.miles_disagg.configs import kimi_k25_2layer_nvfp4 as _base

APP_NAME = "stitch-k25-resume-smoke"
EXPERIMENT_VOLUME_NAME = _base.EXPERIMENT_VOLUME_NAME
LOCAL_CHECKPOINT_PATH = _base.LOCAL_CHECKPOINT_PATH
SIDECAR_COMMIT_MODE = _base.SIDECAR_COMMIT_MODE
SIDECAR_FLUSH_CACHE_ON_COMMIT = _base.SIDECAR_FLUSH_CACHE_ON_COMMIT
SGLANG_DELTA_UPDATE_MODE = _base.SGLANG_DELTA_UPDATE_MODE
SGLANG_SERVER_ARGS = {
    **_base.SGLANG_SERVER_ARGS,
    "--cuda-graph-backend-prefill": "disabled",
}
MEGATRON_RUNTIME_PATCHES = _base.MEGATRON_RUNTIME_PATCHES
modal = copy.copy(_base.modal)
miles = copy.copy(_base.miles)

# Current Miles loads K2.5 directly through Megatron Bridge; no offline torch_dist
# checkpoint is part of the model's native recipe.
miles.ref_load = str(_base.BF16_CHECKPOINT_PATH)
miles.megatron_to_hf_mode = "bridge"
miles.attention_backend = "flash"

# The first trainer saves rollout 1 (weight v2), publishes through v3, then exits.
# Resuming the same run must restore v2 and replace the abandoned v3 directory.
miles.num_rollout = 4
miles.save_interval = 2
miles.debug_exit_after_rollout = 3
miles.rollout_batch_size = 2
miles.rollout_max_response_len = 64
miles.n_samples_per_prompt = 1
miles.global_batch_size = 2
miles.sglang_server_concurrency = 1
