"""Kill-and-resume e2e: the Qwen3-30B NVFP4 recipe with a fast checkpoint cadence.

Reuses the base recipe's prepared checkpoints and dataset; this derivation only
sets the checkpoint policy resume needs, a light rollout load so steps take
minutes on the single-replica floor, and a run-scoped app/volume identity. Kill
the trainer after the first complete pair; its Modal retry must resume from it,
rewinding the live replica in place.

Known Miles caveat: the qwen3moe bridge-mode ``save_hf`` export writes an
incomplete weight map, so a replica that boots from an export cannot apply
later deltas (its forward staging fails and retries while it keeps serving).
"""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.miles_disagg.configs.qwen3_30b_a3b_nvfp4_46 import *  # noqa: F401,F403
from cookbook.miles_disagg.configs.qwen3_30b_a3b_nvfp4_46 import (
    _Miles,
)
from cookbook.miles_disagg.configs.qwen3_30b_a3b_nvfp4_46 import (
    modal as _base_modal,
)

APP_NAME = "stitch-qwen3-30b-nvfp4-e2e"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-30b-nvfp4-e2e"

# Megatron saves spill through the trainer's ephemeral disk (see the base
# recipe's save_interval note); the e2e writes several. Host RAM must cover the
# CPU-offloaded optimizer plus the saves' write-back cache — the 1 TiB default
# OOMed at the second save.
modal = ModalConfig(
    **vars(_base_modal),
    trainer_ephemeral_disk_mib=1024 * 1024,
    trainer_memory_mib=(1024 * 1024, 1792 * 1024),
)


class _E2EMiles(_Miles):
    num_rollout = 7
    save_interval = 3
    save_hf = "hf_checkpoints/weight_v{rollout_id:06d}"

    # The pinned Miles routes an external rollout endpoint through the v2
    # session server (v1 assumes a Miles-managed router); the default v2
    # picker/postprocessor cover single-turn math prompts.
    use_session_server = "v2"
    session_server_port = 21000
    session_server_startup_timeout_seconds = 180
    # The default generate builds Miles-router URLs; the hub's single_turn
    # honors rollout_endpoint_url (the runtime Miles patch). The GLM recipes
    # instead route through custom session-agent functions.
    custom_generate_function_path = "miles.rollout.generate_hub.single_turn.generate"
    pause_generation_mode = "in_place"  # overlap weight sync, as the GLM recipes do
    # The default rollout loop aborts stragglers through the Miles router; the
    # fully-async loop is the external-fleet path the GLM recipes run.
    fully_async = True
    async_max_concurrent_samples = 16
    async_data_buffer_capacity_factor = 3.0
    async_unused_samples_handler = "drop"

    rollout_batch_size = 8
    n_samples_per_prompt = 4
    global_batch_size = 32
    rollout_max_response_len = 3072


miles = _E2EMiles()
