"""Miles hooks an offline eval sets on its eval dataset.

Runs inside Miles' eval, so it imports Miles; nothing in training references it.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def generate(input: Any) -> Any:
    """The recipe's generate function, rerun while an episode aborts.

    An aborted episode failed in the sandbox or the API before the policy got a
    verdict, and Miles' eval would score it 0. Each attempt starts from the original
    sample. Every returned sample records its attempts, why each earlier attempt
    aborted, and its sample index, which the results writer reads.
    """
    from miles.rollout.inference_rollout.compatibility import load_generate_function
    from miles.utils.types import Sample

    inner = load_generate_function(input.args.custom_generate_function_path)
    retries = max(0, int(getattr(input.args, "eval_infra_retries", 0)))
    original = input.sample
    attempt = 0
    aborts: list[dict[str, Any]] = []
    while True:
        attempt += 1
        output = await inner(dataclasses.replace(input, sample=copy.deepcopy(original)))
        samples = (
            output.samples if isinstance(output.samples, list) else [output.samples]
        )
        finished = bool(samples) and all(
            sample.status != Sample.Status.ABORTED for sample in samples
        )
        if not finished:
            aborts.append(_abort_reason(attempt, samples))
        for sample in samples:
            sample.metadata = {
                **(sample.metadata or {}),
                "eval_attempts": attempt,
                "eval_aborts": list(aborts),
                "eval_sample_index": original.index,
            }
        if finished or attempt > retries:
            return output
        logger.warning(
            "eval sample %s aborted (attempt %d of %d)",
            original.index,
            attempt,
            retries + 1,
        )


# What the agent records about a failed episode (``modal_swe.agent._failure``).
_ABORT_FIELDS = ("exit_status", "failure_phase", "agent_error", "root_error")


def _abort_reason(attempt: int, samples: list[Any]) -> dict[str, Any]:
    """Why an attempt aborted, as its first sample records it. An attempt that
    aborts before any model call returns a bare copy of the input sample, so its
    fields are absent."""
    metadata = (samples[0].metadata or {}) if samples else {}
    return {"attempt": attempt, **{key: metadata.get(key) for key in _ABORT_FIELDS}}
