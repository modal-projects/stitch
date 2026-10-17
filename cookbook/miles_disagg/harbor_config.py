"""Shared settings for Miles' Harbor path; recipes own model, tasks and capacity.

Tasks are prepared outside Stitch as Harbor task folders plus one train.jsonl on a Volume.
"""

import base64
import json
import os

from cookbook.common.serving_image import SGLangRuntime
from cookbook.miles_disagg import swebench_config

# Environment of the recipe's task Volume and judge Secret, when it is not the run's environment.
DATA_ENV = os.environ.get("STITCH_DATA_ENV") or None

# The stack that runs score centering on feat/heterogeneous-swe-e2e. Other recipes keep
# Stitch's default pins. This Miles retries a rollout request's 409 from its own settings
# and lacks stitch-miles' delta-history cleanup on resume (#3875).
MILES_IMAGE_TAG = "radixark/miles:dev-202609290439"
MILES_REPO_REF = "0fe16fcb5ed5fba9bd3aef43f7b03f60ee774bd0"  # stitch-miles-hetero
SGLANG_RUNTIME = SGLangRuntime(
    image="lmsysorg/sglang:v0.5.20",
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-hetero",  # returns the sampler's candidates for score centering
    commit="25d7c62b2b8f7eae7f3a0b45807d36ea8fbaac9c",
)
# Long turns get their response started and kept alive, as on feat/heterogeneous-swe-e2e.
SIDECAR_RESPONSE_KEEPALIVE_AFTER = 60.0

# terminus-2 needs Harbor's Miles branch (truncation policy for TITO); pip cannot resolve it.
# Install into the interpreter that runs the trainer and Ray (the image's /opt/sglang venv).
HARBOR_REF = "917b3c1be7cf6553a7a9fe25de8adb6f8e639287"  # harbor-miles-v0.20.0
TRAINER_EXTRA_PIP_PACKAGES = ("uv",)
# Miles skips the dynamic global batch size for a batch with compact rollouts (an agent session that yielded
# several samples, e.g. after terminus-2 summarized its context), then fails its check
# `use_dynamic_global_batch_size == (x is not None)` after the rollout. Size such a batch by rollouts. Applied
# to the pinned Miles at image build; the build fails if the line is gone.
COMPACT_BATCH_PATCH = """
import sys
path = sys.argv[1]
text = open(path).read()
old = "    if not args.disable_rollout_trim_samples and not is_compact:\\n"
new = (
    "    if not args.disable_rollout_trim_samples and is_compact and args.use_dynamic_global_batch_size:\\n"
    "        rollouts = len({s.rollout_id if s.rollout_id is not None else s.index for s in data})\\n"
    "        metadata['dynamic_global_batch_size'] = _compute_dynamic_global_batch_size(\\n"
    "            args, train_parallel_config=train_parallel_config, num_samples=rollouts\\n"
    "        )\\n" + old
)
if text.count(old) != 1 or "is_compact and args.use_dynamic_global_batch_size" in text:
    sys.exit(f"compact batch patch: {path} lacks the line or has the patch already")
open(path, "w").write(text.replace(old, new))
"""
_PATCHED_FILE = "/root/miles/miles/ray/rollout/rollout_data_conversion.py"
TRAINER_IMAGE_RUN_COMMANDS = (
    "uv pip install --python /opt/sglang/bin/python "
    f'"harbor[modal] @ git+https://github.com/harbor-framework/harbor@{HARBOR_REF}" "modal==1.5.3"',
    # Miles imports Mooncake in every Ray worker (object_store.py) though these recipes use the Ray object
    # store. Its Go library (libetcd_wrapper.so) put Go's signal handler in front of every crash, which hid
    # the real stack (see RAY_USE_UVLOOP in environment()). Without the package Miles falls back
    # (_MOONCAKE_AVAILABLE = False).
    "uv pip uninstall --python /opt/sglang/bin/python mooncake-transfer-engine-cuda13",
    f"echo {base64.b64encode(COMPACT_BATCH_PATCH.encode()).decode()} | base64 -d | python3 - {_PATCHED_FILE}",
)


# Score centering (arXiv:2609.20807) in place of #267's IcePop, as in the recipe
# qwen3_6_35b_a3b_hetero_score_centering_mis_top_p on feat/heterogeneous-swe-e2e (score
# centering, masked importance sampling, top-p sampling with support replay). Score centering
# alone, or IcePop's masking alone, collapsed there.
SCORE_CENTERING = {
    "loss_type": "score_centering",
    "score_centering_top_k": 128,
    # Each token's score is weighted by its trainer / sampler probability ratio inside
    # [0.5, 5] and by zero outside it.
    "score_centering_is": "mis",
    "score_centering_mis_low": 0.5,
    "score_centering_mis_high": 5.0,
    "disable_grpo_std_normalization": True,  # group-centered rewards, as in the paper
    "calculate_per_token_loss": True,
    "use_rollout_logprobs": True,  # the loss scores the trainer against the sampler
    "skip_actor_forward_only": False,
    "use_tis": False,
    "custom_tis_function_path": None,
    # The score-centering loss logs its own train-rollout mismatch (train_rollout_kl);
    # this flag adds the policy-loss path's metrics and needs a TIS function there.
    "get_mismatch_metrics": None,
    "tis_clip_low": None,
    "tis_clip": None,
    # Samplers draw from the top-p 0.97 nucleus, at most 64 tokens. Miles then replays that
    # support in the trainer, and the 128 recorded candidates hold all of it, so centering
    # is exact.
    "rollout_top_p": 0.97,
    "rollout_top_k": 64,
    # No DFlash: this Miles pin has no score centering for speculative decoding.
    "sglang_speculative_algorithm": None,
    # This Miles pin retries a stale-version 409 itself, from these settings.
    "rollout_request_max_attempts": 1200,
    "rollout_request_retry_interval": 1.0,
}


def without_speculative_decoding(server_args: dict[str, str]) -> dict[str, str]:
    """SGLang server arguments with the speculative-decoding flags removed."""
    return {k: v for k, v in server_args.items() if not k.startswith("--speculative-")}


def arguments(tasks_root: str) -> dict:
    """Miles workload settings: the Harbor agent function and its reward."""
    args = swebench_config.arguments()
    for key in (
        "custom_rollout_log_function_path",
        "session_sample_picker_path",
        "session_sample_postprocessor_path",
    ):
        args.pop(key)
    return args | {
        "prompt_data": f"{tasks_root}/train.jsonl",
        # Miles' function, minus the env that broke terminus on old tmux (harbor_agent.py).
        "custom_agent_function_path": "cookbook.miles_disagg.harbor_agent.run",
        "custom_rm_path": "generate.reward_func",
        # Qwen3.8's template defaults to xhigh effort; the effort ablations favor medium.
        "apply_chat_template_kwargs": {"reasoning_effort": "medium"},
    }


def environment(
    *,
    tasks_root: str,
    trials_dir: str,
    sandbox_app: str,
    max_seq_len: int,
    max_output_tokens: int,
    agent_timeout: int,
    trial_timeout: int,
    max_turns: int,
) -> dict[str, str]:
    """Rollout-worker environment for harbor_agent_function."""
    return {
        "PYTHONPATH": (
            "/root/Megatron-LM:/root/miles:/root/miles/examples/experimental/harbor"
            ":/root/miles/examples/swe-agent-harbor-docker"
        ),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "RAY_health_check_timeout_ms": "60000",
        "RAY_health_check_failure_threshold": "30",
        # Ray installs uvloop in every worker, so each event loop of the rollout worker (Ray's, Miles',
        # Modal's) is a uvloop loop. With 256 agents that worker segfaulted every 20-60 min in libuv's
        # timer heap (heap_remove in uv_timer_stop). asyncio's own loop has no C timer heap to corrupt.
        "RAY_USE_UVLOOP": "0",
        "LITELLM_LOG": "ERROR",
        "HARBOR_ENV_TYPE": "modal",
        "HARBOR_TASKS_DIR": f"{tasks_root}/tasks",
        "HARBOR_TRIALS_DIR": trials_dir,
        # Harbor's default sandbox lifetime is 24 h, so a sandbox of a stopped run keeps running.
        # Let each sandbox live only a little longer than its trial.
        "HARBOR_ENV_KWARGS": json.dumps(
            {"app_name": sandbox_app, "sandbox_timeout_secs": trial_timeout + 600}
        ),
        "AGENT_MODEL_NAME": "model",
        "AGENT_TIMEOUT": str(agent_timeout),
        "AGENT_TRIAL_TIMEOUT": str(trial_timeout),
        "AGENT_MAX_INPUT_TOKENS": str(max_seq_len - max_output_tokens),
        "AGENT_MAX_OUTPUT_TOKENS": str(max_output_tokens),
        "HARBOR_MAX_SEQ_LEN": str(max_seq_len),
        "HARBOR_AGENT_MAX_ITERATIONS": str(max_turns),
    }
