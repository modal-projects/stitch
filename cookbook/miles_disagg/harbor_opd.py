"""On-policy distillation (OPD) from a teacher server for Harbor rollouts: Miles' SGLang OPD with the Harbor reward kept.

Miles scores samples for SGLang OPD in its reward function, which it calls only for samples without a reward, and its
stock post-process sets every task reward to 0. Session server v2 gives every Harbor sample its agent's reward. So:
- `session_postprocess` (--session-sample-postprocessor-path) keeps that reward in the sample's metadata instead of
  assigning it, so Miles calls the reward function;
- `teacher_reward` (--custom-rm-path) scores the sample's tokens with the teacher and returns the Harbor reward with the
  teacher's log-probs of the response tokens; the recipe sets reward_key="harbor", so filters and metrics read the
  Harbor reward;
- `post_process_rewards` (--custom-reward-post-process-path) stores those log-probs on each sample and returns the
  Harbor rewards as Miles' default post-process does.
The teacher is a Modal Endpoint at --rm-url (SGLang's native /generate), reached with the trainer's proxy token.
"""

import asyncio
import logging
import os

import aiohttp
import torch
from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout
from miles.rollout.session.v2.postprocessor_hub.default_postprocess import default_postprocess

logger = logging.getLogger(__name__)
ATTEMPTS = 4
TIMEOUT_S = 600


def session_postprocess(leaf_samples: list, session_metadata: dict) -> list:
    samples = default_postprocess(leaf_samples, session_metadata)
    reward = (session_metadata.get("agent") or {}).get("reward")
    for sample in samples:
        sample.metadata["harbor_reward"] = float(reward or 0.0)
        sample.reward = None  # so Miles calls teacher_reward for it
    return samples


async def teacher_reward(args, sample, **kwargs) -> dict:
    if not args.rm_url:
        raise ValueError("--rm-url is empty: set the teacher's /generate URL (OPD_TEACHER_URL)")
    n = sample.response_length
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    headers = {
        "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
        "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
    }
    for attempt in range(ATTEMPTS):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as session:
                async with session.post(args.rm_url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    entries = (await response.json())["meta_info"]["input_token_logprobs"]
            # SGLang's first input log-prob is a placeholder; the response is the last n tokens
            teacher = [entry[0] for entry in entries[1:][-n:]] if n > 0 else []
            if len(teacher) != n:
                raise ValueError(f"teacher scored {len(teacher)} of {n} response tokens")
            return {"harbor": sample.metadata["harbor_reward"], "teacher_log_probs": teacher}
        except Exception as exc:  # noqa: BLE001 - retried; the last failure falls back below
            logger.warning("teacher scoring failed (attempt %d of %d): %s", attempt + 1, ATTEMPTS, exc)
            await asyncio.sleep(2**attempt)
    # No teacher: the sampler's own log-probs, so this sample gets no distillation signal
    return {
        "harbor": sample.metadata["harbor_reward"],
        "teacher_log_probs": list(sample.rollout_log_probs),
        "teacher_failed": True,
    }


def post_process_rewards(args, samples: list, **kwargs) -> tuple[list[float], list[float]]:
    for sample in samples:
        sample.teacher_log_probs = torch.tensor(sample.reward["teacher_log_probs"], dtype=torch.float32)
    if failed := sum(bool(sample.reward.get("teacher_failed")) for sample in samples):
        logger.warning("teacher scoring failed for %d of %d samples: no OPD signal for them", failed, len(samples))
    raw = [sample.get_reward_value(args) for sample in samples]
    if args.advantage_estimator in ("grpo", "gspo", "reinforce_plus_plus_baseline") and args.rewards_normalization:
        return raw, _normalize_rewards_by_rollout(args, samples, raw, None)
    return raw, raw
