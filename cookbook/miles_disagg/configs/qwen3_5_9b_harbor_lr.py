"""Qwen3.5-9B learning-rate test: the 9B recipe (qwen3_5_9b_harbor) on mixture-9b-v2 for 40 steps, with one learning
rate and rollout seed per run. The run modules qwen3_5_9b_harbor_lr<k>x_s<seed> only pick them; each run gets its
own app, Volume, trials Volume, sandbox app and W&B group.

Tasks: mixture-9b-v2, the 1,095 tasks the 9B solved 1 to 3 of 4 times in its difficulty filter (terminal, code and
bug fixes; no cyber), about 1.5 passes per run.
"""

import modal as modal_sdk

from cookbook.miles_disagg import harbor_config
from cookbook.miles_disagg.configs import qwen3_5_9b_harbor as smoke
from cookbook.miles_disagg.configs.qwen3_5_9b_harbor import *  # noqa: F403

TASKS_ROOT = "/harbor-envs/mixture-9b-v2"


def run(name: str, lr: float, seed: int) -> dict:
    """The module globals of one run, e.g. run("lr1x-s42", 1e-6, 42)."""

    class _Run(smoke._Qwen35_9B):
        num_rollout = 40
        wandb_group = f"qwen3-5-9b-harbor-{name}"
        prometheus_run_name = wandb_group
        environment = {
            **smoke._Qwen35_9B.environment,
            **harbor_config.environment(
                tasks_root=TASKS_ROOT,
                trials_dir="/harbor-trials",
                sandbox_app=f"qwen3-5-9b-harbor-{name}-sandboxes",
                max_seq_len=smoke._Qwen35_9B.max_seq_len,
                max_output_tokens=smoke._Qwen35_9B.rollout_max_response_len,
                agent_timeout=1800,
                trial_timeout=7200,
                max_turns=200,
            ),
        }

    _Run.lr = lr
    _Run.rollout_seed = seed
    arguments = harbor_config.arguments(TASKS_ROOT)
    # Qwen3.5's template has no reasoning effort
    arguments.pop("apply_chat_template_kwargs")
    return {
        "APP_NAME": f"stitch-qwen3-5-9b-harbor-{name}",
        "EXPERIMENT_VOLUME_NAME": f"stitch-miles-qwen3-5-9b-harbor-{name}",
        "TASKS_ROOT": TASKS_ROOT,
        "TRAINER_EXTRA_VOLUMES": {
            **smoke.TRAINER_EXTRA_VOLUMES,
            "/harbor-trials": modal_sdk.Volume.from_name(
                f"harbor-qwen3-5-9b-{name}-trials", create_if_missing=True, version=2
            ),
        },
        "miles": _Run(**arguments, **harbor_config.SCORE_CENTERING),
    }
