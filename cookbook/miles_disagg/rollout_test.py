from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest

from cookbook.miles_disagg import rollout


class Status(Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class Sample:
    index: int = 7
    group_index: int = 3
    rollout_id: int = 7
    prompt: str = "repair task"
    metadata: dict = field(default_factory=lambda: {"task": "original"})
    tokens: list[int] = field(default_factory=list)
    reward: float | None = None
    rollout_log_probs: list[float] | None = None
    rollout_routed_experts: list | None = None
    status: Status = Status.COMPLETED

    def reset_for_retry(self):
        self.tokens = []
        self.reward = self.rollout_log_probs = self.rollout_routed_experts = None
        self.status = Status.ABORTED


@dataclass(frozen=True)
class Input:
    sample: Sample
    evaluation: bool = False
    state: object = field(default_factory=object)
    sampling_params: dict = field(default_factory=lambda: {"temperature": 1})


def failure(sample, **metadata):
    result = deepcopy(sample)
    result.status = Status.ABORTED
    result.tokens = [91, 92]
    result.reward = -1
    result.rollout_log_probs = [-0.4]
    result.rollout_routed_experts = [[9]]
    result.metadata.update(
        exit_status="agent_error",
        failure_phase="interaction",
        agent_metrics={"infra_error": 1},
    )
    result.metadata.update(metadata)
    return SimpleNamespace(samples=[result])


def test_retry_uses_clean_original_prompt_and_preserves_group_identity(monkeypatch):
    request = Input(Sample())
    original = deepcopy(request.sample)
    calls = []

    async def agent(input):
        calls.append(input)
        if len(calls) == 1:
            input.sample.metadata["failed_attempt"] = True
            return failure(input.sample)
        sample = input.sample
        assert sample is not request.sample
        assert (sample.index, sample.group_index, sample.rollout_id, sample.prompt) == (
            original.index,
            original.group_index,
            original.rollout_id,
            original.prompt,
        )
        assert input.state is request.state
        assert input.sampling_params is request.sampling_params
        assert sample.metadata == original.metadata
        assert sample.tokens == []
        assert (
            sample.reward
            is sample.rollout_log_probs
            is sample.rollout_routed_experts
            is None
        )
        sample.status = Status.COMPLETED
        sample.tokens = [11, 12]
        sample.reward = 1
        sample.metadata["agent_metrics"] = {"infra_error": 0}
        return SimpleNamespace(samples=[sample])

    monkeypatch.setattr(rollout, "_agent_generator", lambda: agent)
    result = asyncio.run(rollout.generate(request))
    assert len(calls) == 2
    assert result.samples[0].tokens == [11, 12]
    assert result.samples[0].reward == 1
    assert result.samples[0].metadata["agent_metrics"] == {
        "infra_error": 0,
        "infra_retry_count": 1,
        "infra_retry_recovered": 1,
    }


@pytest.mark.parametrize(
    ("evaluation", "metadata"),
    [
        (True, {}),
        (False, {"exit_status": "sandbox_infra_error"}),
        (False, {"exit_status": "session_collect_error"}),
        (False, {"failure_phase": "startup"}),
        (False, {"agent_metrics": {"infra_error": 0}}),
    ],
)
def test_uncertain_failures_and_evaluation_are_not_retried(
    monkeypatch, evaluation, metadata
):
    request = Input(Sample(), evaluation=evaluation)
    result = failure(request.sample, **metadata)
    calls = []

    async def agent(input):
        calls.append(input)
        return result

    monkeypatch.setattr(rollout, "_agent_generator", lambda: agent)
    assert asyncio.run(rollout.generate(request)) is result
    assert len(calls) == 1
    metrics = result.samples[0].metadata["agent_metrics"]
    if evaluation:
        assert "infra_retry_count" not in metrics
        assert "infra_retry_recovered" not in metrics
    else:
        assert metrics["infra_retry_count"] == 0
        assert metrics["infra_retry_recovered"] == 0


def test_second_failure_remains_aborted_without_a_third_attempt(monkeypatch):
    calls = []

    async def agent(input):
        calls.append(input)
        return failure(input.sample)

    monkeypatch.setattr(rollout, "_agent_generator", lambda: agent)
    result = asyncio.run(rollout.generate(Input(Sample())))
    assert len(calls) == 2
    assert result.samples[0].status is Status.ABORTED
    assert result.samples[0].metadata["agent_metrics"]["infra_retry_recovered"] == 0


def test_only_failed_member_is_regenerated_and_eight_members_complete(monkeypatch):
    calls = [0] * 8

    async def agent(input):
        index = input.sample.index
        calls[index] += 1
        if index == 7 and calls[index] == 1:
            return failure(input.sample)
        input.sample.status = Status.COMPLETED
        return SimpleNamespace(samples=[input.sample])

    requests = [Input(Sample(index=i, rollout_id=i)) for i in range(8)]
    monkeypatch.setattr(rollout, "_agent_generator", lambda: agent)

    async def run():
        return await asyncio.gather(
            *(rollout.generate(request) for request in requests)
        )

    results = asyncio.run(run())
    assert calls == [1] * 7 + [2]
    assert len(results) == 8
    assert all(result.samples[0].status is Status.COMPLETED for result in results)
    assert all(results[i].samples[0] is requests[i].sample for i in range(7))
    metrics = [result.samples[0].metadata["agent_metrics"] for result in results]
    assert [item["infra_retry_count"] for item in metrics] == [0] * 7 + [1]
    assert [item["infra_retry_recovered"] for item in metrics] == [0] * 7 + [1]


def test_cancellation_does_not_start_a_replacement(monkeypatch):
    calls = []

    async def agent(input):
        calls.append(input)
        raise asyncio.CancelledError

    monkeypatch.setattr(rollout, "_agent_generator", lambda: agent)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(rollout.generate(Input(Sample())))
    assert len(calls) == 1


def test_generator_arguments_are_forwarded(monkeypatch):
    observed = []
    monkeypatch.setattr(
        rollout,
        "_agent_generator",
        lambda: SimpleNamespace(add_arguments=observed.append),
    )
    parser = object()
    rollout.generate.add_arguments(parser)
    assert observed == [parser]
