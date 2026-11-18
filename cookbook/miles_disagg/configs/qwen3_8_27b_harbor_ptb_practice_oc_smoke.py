"""Smoke test of RL rollouts under OpenCode: one step on 4 practice tasks x 2 attempts.

The agent is OpenCode with the evaluation harness (opencode_agent.py), inside the task sandbox; it reaches the session
servers through public tunnels (session_tunnel.py). Tasks: `ptb-practice-oc-smoke` on the harbor-envs Volume, built
by the eval repo's `envs/posttrain_practice.py --agent opencode` (OpenCode in the image, rows name opencode). Start:
the checkpoint that the last practice runs (qwen3_8_27b_harbor_ptb_practice_e and _f_h200x2) started from (step 20 of
the first practice run, weights only). The trainer is one 8 x H200 machine (TP 4 x CP 2, as
qwen3_8_27b_harbor_mix_v2b_h200). Groups with no reward spread stay, so the step does not wait for signal; nothing is
saved. The step's samples and conversations go to the trials Volume (debug/) for the gate checks.
"""

import dataclasses

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_ptb_practice_e as e
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_ptb_practice_e import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-ptb-practice-oc-smoke"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-ptb-practice-oc-smoke"
TASKS_ROOT = "/harbor-envs/ptb-practice-oc-smoke"
HARBOR_AGENT = harbor_config.OPENCODE
AGENT_CONCURRENCY = 8
modal = dataclasses.replace(e.modal, gpu="H200", rollout_min_containers=2, rollout_max_containers=2)
TRAINER_EXTRA_VOLUMES = {
    **e.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-ptb-practice-oc-smoke-trials",
        environment_name=e.first.s2.EXPERIMENT_VOLUME_ENV,
        create_if_missing=True,
        version=2,
    ),
}


class _PtbPracticeOcSmoke(e._PtbPracticeE):
    tensor_model_parallel_size = 4
    context_parallel_size = 2
    rollout_batch_size = 4
    n_samples_per_prompt = 2
    global_batch_size = 4 * 2
    num_rollout = 1
    save_interval = None
    save_retain_interval = None
    dynamic_sampling_filter_path = None
    sglang_server_concurrency = AGENT_CONCURRENCY
    async_max_concurrent_samples = AGENT_CONCURRENCY
    save_debug_rollout_data = "/harbor-trials/debug/rollout_{rollout_id}.pt"
    save_debug_trajectory_data = "/harbor-trials/debug/trajectory_{rollout_id}.jsonl"
    wandb_group = "qwen3-8-27b-ptb-practice-oc-smoke"
    prometheus_run_name = wandb_group
    environment = {
        **e._PtbPracticeE.environment,
        **harbor_config.environment(
            tasks_root=TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app="qwen3-8-27b-ptb-practice-oc-smoke-sandboxes",
            max_seq_len=e.first.s2.b300._MixV2.max_seq_len,
            max_output_tokens=e.first.s2.b300._MixV2.rollout_max_response_len,
            agent_timeout=960,  # x 2.0 = 32 minutes: the 30-minute tasks fit
            trial_timeout=3600,
            max_turns=200,
        ),
    }


miles = _PtbPracticeOcSmoke(
    **harbor_config.arguments(TASKS_ROOT, agent=HARBOR_AGENT), **harbor_config.SCORE_CENTERING
)
