"""Mixture v2 on two 8 x H200 machines with 12,288 tokens per reply in training instead of 16,384, 35 steps.

A reply over the limit ends its attempt with score 0, so a tighter limit in training pushes replies shorter.
Tests whether that cuts the eval attempts that end on a too-long reply; evals keep 16,384 tokens per reply,
so the scores compare with qwen3_8_27b_harbor_mix_v2_h200x2. Own app, Volume, trials Volume, sandbox app and
W&B group, so it runs next to other v2 runs.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_8_27b_harbor_mix_v2_h200x2 as h200
from cookbook.miles_disagg.configs.qwen3_8_27b_harbor_mix_v2_h200x2 import *  # noqa: F403

APP_NAME = "stitch-qwen3-8-27b-harbor-mix-v2-reply12k"
EXPERIMENT_VOLUME_NAME = "stitch-miles-qwen3-8-27b-harbor-mix-v2-reply12k"
TRAINER_EXTRA_VOLUMES = {
    **h200.TRAINER_EXTRA_VOLUMES,
    "/harbor-trials": modal_sdk.Volume.from_name(
        "harbor-mix-v2-reply12k-trials", create_if_missing=True, version=2
    ),
}
REPLY_TOKENS = 12_288


def _environment(sandbox_app: str) -> dict[str, str]:
    """v2's rollout environment with this reply limit. The input budget stays v2's (context minus 16,384), so
    only the reply limit changes."""
    v2 = h200._MixV2H200x2.environment
    return {
        **v2,
        **harbor_config.environment(
            tasks_root=h200.TASKS_ROOT,
            trials_dir="/harbor-trials",
            sandbox_app=sandbox_app,
            max_seq_len=h200._MixV2H200x2.max_seq_len,
            max_output_tokens=REPLY_TOKENS,
            agent_timeout=1800,
            trial_timeout=7200,
            max_turns=200,
        ),
        "AGENT_MAX_INPUT_TOKENS": v2["AGENT_MAX_INPUT_TOKENS"],
    }


class _MixV2Reply12k(h200._MixV2H200x2):
    rollout_max_response_len = REPLY_TOKENS
    num_rollout = 35
    wandb_group = "qwen3-8-27b-harbor-mix-v2-reply12k"
    prometheus_run_name = wandb_group
    environment = _environment("qwen3-8-27b-harbor-mix-v2-reply12k-sandboxes")


miles = _MixV2Reply12k(**harbor_config.arguments(h200.TASKS_ROOT), **harbor_config.SCORE_CENTERING)
