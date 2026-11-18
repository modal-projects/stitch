"""20 RL steps under OpenCode with the evaluation's harness, from step 20 of the first practice run (weights only), on
practice tasks graded with vLLM's decoding rules, as PostTrainBench's final evaluation decodes a shipped model.

Tasks: `ptb-practice-g-oc` on the harbor-envs Volume (eval repo envs/posttrain_practice.py --agent opencode, with the
vLLM-rule grader): decoding (long answers, so an unset temperature and the base model's end-of-text-only stop cost
reward), integrity (dev questions shown item by item; training on data built around them halves the reward), and
graded and repair tasks, each validated by its oracle. Trainer: one 8 x H200 machine (TP 4 x CP 2), as
qwen3_8_27b_harbor_ptb_practice_oc_smoke; batch, steps and saves as qwen3_8_27b_harbor_ptb_practice.
"""

import dataclasses

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_ptb_practice_e as e
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_ptb_practice_e import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-ptb-practice-g-oc"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-ptb-practice-g-oc"
TASKS_ROOT = "/harbor-envs/ptb-practice-g-oc"
HARBOR_AGENT = harbor_config.OPENCODE
AGENT_CONCURRENCY = 128
modal = dataclasses.replace(e.modal, gpu="H200", rollout_min_containers=16, rollout_max_containers=16)
TRAINER_EXTRA_VOLUMES = {
    **e.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-ptb-practice-g-oc-trials",
        environment_name=e.first.s2.EXPERIMENT_VOLUME_ENV,
        create_if_missing=True,
        version=2,
    ),
}


class _PtbPracticeGOc(e._PtbPracticeE):
    tensor_model_parallel_size = 4
    context_parallel_size = 2
    sglang_server_concurrency = AGENT_CONCURRENCY
    async_max_concurrent_samples = AGENT_CONCURRENCY
    wandb_group = "qwen3-8-27b-ptb-practice-g-oc"
    prometheus_run_name = wandb_group
    environment = {
        **e._PtbPracticeE.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-ptb-practice-g-oc-sandboxes",
            max_seq_len=e.first.s2.b300._MixV2.max_seq_len,
            max_output_tokens=e.first.s2.b300._MixV2.rollout_max_response_len,
            agent_timeout=960,  # x 2.0 = 32 minutes: the 30-minute tasks fit
            trial_timeout=3600,
            max_turns=200,
        ),
    }


miles = _PtbPracticeGOc(
    **harbor_config.arguments(TASKS_ROOT, agent=HARBOR_AGENT), **harbor_config.SCORE_CENTERING
)
