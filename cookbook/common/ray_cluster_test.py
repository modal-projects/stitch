from __future__ import annotations

import sys
from types import SimpleNamespace

import modal
import pytest

from cookbook.common import ray_cluster


@pytest.fixture
def ray(monkeypatch):
    ray = SimpleNamespace(
        node="head", participants=["head", "worker"], seen=[], get_error=None
    )
    ray.nodes = lambda: [{"NodeID": node, "Alive": True} for node in ray.participants]
    ray.get_runtime_context = lambda: SimpleNamespace(get_node_id=lambda: ray.node)

    class Remote:
        def __init__(self, fn, strategy=None):
            self.fn, self.strategy = fn, strategy

        def options(self, *, scheduling_strategy):
            assert not scheduling_strategy.soft
            return Remote(self.fn, scheduling_strategy)

        def remote(self, volumes):
            return self.strategy.node_id, self.fn, volumes

    def remote(*, num_cpus):
        assert num_cpus == 0
        return Remote

    def get(pending, *, timeout):
        assert timeout == ray_cluster.RAY_WORKER_JOIN_TIMEOUT
        if ray.get_error:
            raise ray.get_error
        results = []
        for node, fn, volumes in pending:
            ray.node = node
            results.append(fn(volumes))
        return results

    ray.remote, ray.get = remote, get
    monkeypatch.setattr(
        modal.Volume,
        "from_id",
        lambda volume_id: SimpleNamespace(
            reload=lambda: ray.seen.append((ray.node, volume_id))
        ),
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(
        sys.modules,
        "ray.util.scheduling_strategies",
        SimpleNamespace(
            NodeAffinitySchedulingStrategy=lambda node_id, soft: SimpleNamespace(
                node_id=node_id, soft=soft
            )
        ),
    )
    return ray


def test_volume_barrier_refreshes_every_node(ray):
    ray_cluster.reload_volumes_on_nodes(["vo-test"], n_nodes=2)

    assert ray.seen == [("head", "vo-test"), ("worker", "vo-test")]


def test_volume_barrier_rejects_missing_node(ray):
    ray.participants = ["head"]

    with pytest.raises(RuntimeError, match="expected 2 live Ray nodes"):
        ray_cluster.reload_volumes_on_nodes([], n_nodes=2)
    assert ray.seen == []


@pytest.mark.parametrize(
    "failure", [RuntimeError("reload failed"), TimeoutError("node died")]
)
def test_volume_barrier_propagates_failure(ray, failure):
    ray.get_error = failure

    with pytest.raises(type(failure), match=str(failure)):
        ray_cluster.reload_volumes_on_nodes([], n_nodes=2)


def test_volume_barrier_rejects_wrong_participant(ray):
    ray.get_runtime_context = lambda: SimpleNamespace(get_node_id=lambda: "head")

    with pytest.raises(RuntimeError, match="not refreshed on every Ray node"):
        ray_cluster.reload_volumes_on_nodes([], n_nodes=2)
