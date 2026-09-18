"""External-rollout cleanup contracts, exercised inside the Miles trainer image."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def rollout(monkeypatch):
    pytest.importorskip("miles")
    from miles.rollout.inference_rollout import inference_rollout_train

    monkeypatch.setattr(inference_rollout_train, "call_agent_abort_hook", AsyncMock())
    return inference_rollout_train


def test_external_cleanup_does_not_discover_or_abort_fleet_workers(
    rollout, monkeypatch
):
    monkeypatch.setattr(
        rollout, "get_worker_urls", AsyncMock(side_effect=AssertionError("no router"))
    )
    monkeypatch.setattr(
        rollout, "post", AsyncMock(side_effect=AssertionError("no fleet-wide abort"))
    )
    args = SimpleNamespace(
        rollout_endpoint_url="https://fleet.example", partial_rollout=False
    )
    state = SimpleNamespace(args=args, aborted=False)

    assert asyncio.run(rollout.abort(state, set(), 3)) == []

    assert state.aborted
    rollout.call_agent_abort_hook.assert_awaited_once_with(args)


@pytest.mark.parametrize("partial_rollout", [False, True])
def test_external_cleanup_drains_agent_tasks_and_preserves_partial_samples(
    rollout, monkeypatch, partial_rollout
):
    monkeypatch.setattr(
        rollout, "get_worker_urls", AsyncMock(side_effect=AssertionError("no router"))
    )
    args = SimpleNamespace(
        rollout_endpoint_url="https://fleet.example", partial_rollout=partial_rollout
    )
    state = SimpleNamespace(args=args, aborted=False)
    sample = SimpleNamespace(response="partial answer", metadata={})

    async def run():
        released = asyncio.Event()

        async def pending_agent():
            await released.wait()
            return [sample]

        async def abort_agent(_args):
            assert state.aborted
            released.set()

        monkeypatch.setattr(rollout, "call_agent_abort_hook", abort_agent)
        task = asyncio.create_task(pending_agent())
        result = await asyncio.wait_for(rollout.abort(state, {task}, 3), timeout=1)
        assert task.done() and not task.cancelled()
        return result

    assert asyncio.run(run()) == ([[sample]] if partial_rollout else [])
    assert sample.metadata == ({"start_rollout_id": 3} if partial_rollout else {})


def test_managed_cleanup_still_aborts_each_worker(rollout, monkeypatch):
    args = SimpleNamespace(rollout_endpoint_url=None, partial_rollout=False)
    state = SimpleNamespace(args=args, aborted=False)
    workers = ["http://worker-a", "http://worker-b"]
    monkeypatch.setattr(rollout, "get_worker_urls", AsyncMock(return_value=workers))
    monkeypatch.setattr(rollout, "post", AsyncMock())

    assert asyncio.run(rollout.abort(state, set(), 3)) == []

    rollout.get_worker_urls.assert_awaited_once_with(args)
    assert rollout.post.await_count == 2
    for worker in workers:
        rollout.post.assert_any_await(f"{worker}/abort_request", {"abort_all": True})
    rollout.call_agent_abort_hook.assert_awaited_once_with(args)
