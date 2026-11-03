"""Mixture v3a on two 8 x H200 machines: qwen3_8_27b_harbor_mix_v2_h200x2 on the task mix `mixture-v3a`.

mixture-v3a keeps the 1,885 mixture-v2 tasks that still gave a learning signal late in the v2 run, and adds 328
tasks that the base model solved 1 to 7 of 8 times (TermGrade, MiMo terminal, mixture-v2 drops). Only the tasks
and names change, so the scores compare with v2 from the same start. 55 steps are about one epoch. Own app,
Volume, trials Volume, sandbox app and W&B group.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2_h200x2 as h200
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2_h200x2 import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v3a"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v3a"
TASKS_ROOT = "/harbor-envs/mixture-v3a"
TRAINER_EXTRA_VOLUMES = {
    **h200.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v3a-trials", create_if_missing=True, version=2
    ),
}


class _MixV3a(h200._MixV2H200x2):
    wandb_group = "qwen3-8-27b-harbor-mix-v3a"
    prometheus_run_name = wandb_group
    environment = {
        **h200._MixV2H200x2.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-harbor-mix-v3a-sandboxes",
            max_seq_len=h200._MixV2H200x2.max_seq_len,
            max_output_tokens=h200._MixV2H200x2.rollout_max_response_len,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
    }


miles = _MixV3a(**harbor_config.arguments(TASKS_ROOT), **harbor_config.SCORE_CENTERING)
