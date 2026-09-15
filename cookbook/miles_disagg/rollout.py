"""Bounded recovery of completed SWE episodes that fail during model interaction."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput

logger = logging.getLogger(__name__)


def _agent_generator():
    from miles.rollout.generate_hub.agentic_tool_call import generate

    return generate


def _samples(result):
    return result.samples if isinstance(result.samples, list) else [result.samples]


def _completed_interaction_failure(result) -> bool:
    samples = _samples(result)
    if len(samples) != 1:
        return False
    sample = samples[0]
    metadata = sample.metadata or {}
    return (
        sample.status.value == "aborted"
        and metadata.get("exit_status") == "agent_error"
        and metadata.get("failure_phase") == "interaction"
        and (metadata.get("agent_metrics") or {}).get("infra_error") == 1
    )


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Retry one completed infrastructure failure in a fresh session and sandbox.

    The agent has returned through its cleanup before this failure is reported.
    Session creation/collection failures and cancellation do not establish that
    contract, so they are left to the existing complete-group filter.
    """
    original = deepcopy(input.sample)
    generate_agent = _agent_generator()
    result = await generate_agent(input)
    if input.evaluation:
        return result

    retried = _completed_interaction_failure(result)
    if retried:
        original.reset_for_retry()
        logger.warning(
            "Retrying completed infrastructure-failed SWE interaction for sample %s",
            original.index,
        )
        result = await generate_agent(replace(input, sample=original))
    samples = _samples(result)
    recovered = (
        retried and bool(samples) and all(s.status.value != "aborted" for s in samples)
    )
    for sample in samples:
        metrics = dict((sample.metadata or {}).get("agent_metrics") or {})
        metrics.update(
            infra_retry_count=int(retried), infra_retry_recovered=int(recovered)
        )
        sample.metadata = {**(sample.metadata or {}), "agent_metrics": metrics}
    return result


def _add_arguments(parser):
    _agent_generator().add_arguments(parser)


generate.add_arguments = _add_arguments
