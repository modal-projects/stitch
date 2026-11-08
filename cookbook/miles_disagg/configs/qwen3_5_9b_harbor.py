"""Qwen3.5-9B on our Harbor tasks: a smaller, weaker test model (smoke run and base recipe).

The settings of the Qwen3.8-27B Harbor recipes (score centering, Terminus-2, 128K tokens per rollout, 16K reply
cap, one hour per agent), with the model changed. Qwen3.5-9B has the same hybrid architecture as Qwen3.8-27B, with
32 layers instead of 64. Its chat template family is qwen35: the Hub template with preserve_thinking. That template
has no reasoning effort, so this recipe drops `reasoning_effort`.

Tasks: mixture-9b-v1, the tasks the 9B solved 1 to 3 of 4 times in its difficulty filter (no cyber: the 9B is at
its floor there). As is, this is the 5-step smoke run (`*-smoke` Volumes are not auto-evaluated); the learning-rate
runs subclass it with their own names, length and learning rate.
"""

import dataclasses

import modal as modal_sdk

from cookbook.common.constants import CHECKPOINTS_PATH
from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2 as v2
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2 import *  # noqa: F403

APP_NAME = "stitch-qwen3-5-9b-harbor-smoke"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-5-9b-harbor-smoke"
SOURCE_MODEL = "Qwen/Qwen3.5-9B"
SOURCE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
BF16_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-5-9b-c2022362-bf16"
ROLLOUT_CHECKPOINT_PATH = BF16_CHECKPOINT_PATH
TORCH_DIST_CHECKPOINT_PATH = CHECKPOINTS_PATH / "qwen3-5-9b-c2022362-torch-dist-tp2"
# Tasks the 9B solved 1-3 of 4 in its difficulty filter (PLAN.md section 0.10).
TASKS_ROOT = "/harbor-envs/mixture-9b-v1"
# Harbor at the 27B recipes' pin plus one parser fix (modal-projects/harbor, branch anthonyhe/terminus-blank-tag):
# a reply with blank brackets such as "< >" no longer crashes Terminus-2 and ends the trial.
HARBOR_REF = "611da466c821938e33d63c12a906642b1774b3d7"
TRAINER_IMAGE_RUN_COMMANDS = (
    "uv pip install --python /opt/sglang/bin/python "
    f'"harbor[modal] @ git+https://github.com/modal-projects/harbor@{HARBOR_REF}" "modal==1.5.3"',
    "uv pip uninstall --python /opt/sglang/bin/python mooncake-transfer-engine-cuda13",
)
TRAINER_EXTRA_VOLUMES = {
    "/harbor-envs": v2.TRAINER_EXTRA_VOLUMES["/harbor-envs"],
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-qwen3-5-9b-smoke-trials", create_if_missing=True, version=2
    ),
}

# One 8 x H200 trainer (TP 2 x CP 2 x DP 2, the 27B's B300 split) and 16 rollout servers with 16 agent sessions
# each, 256 agents as in the 27B runs. Per token the 9B needs half the 27B's KV cache, and its weights are a third
# as large, so an H200 holds more of its sessions.
modal = dataclasses.replace(
    v2.modal,
    gpu="H200",
    rollout_min_containers=16,
    rollout_max_containers=16,
    rollout_target_inputs=16,
)


class _Qwen35_9B(v2._MixV2):
    megatron_model_type = "qwen3.5-9B"
    hf_checkpoint = str(ROLLOUT_CHECKPOINT_PATH)
    ref_load = str(TORCH_DIST_CHECKPOINT_PATH)
    tito_model = "qwen35"
    num_rollout = 5  # the smoke run; the learning-rate runs set their own
    wandb_group = "qwen3-5-9b-harbor-smoke"
    prometheus_run_name = wandb_group
    environment = {
        **v2._MixV2.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-5-9b-harbor-smoke-sandboxes",
            max_seq_len=v2._MixV2.max_seq_len,
            max_output_tokens=v2._MixV2.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


_arguments = harbor_config.arguments(TASKS_ROOT)
# Qwen3.5's template has no reasoning effort
_arguments.pop("apply_chat_template_kwargs")
miles = _Qwen35_9B(**_arguments, **harbor_config.SCORE_CENTERING)
