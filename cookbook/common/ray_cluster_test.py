import os
from unittest.mock import patch

import pytest

from cookbook.common import ray_cluster


def test_interface_for_ip() -> None:
    with patch.object(
        ray_cluster,
        "_ipv4_interfaces",
        return_value=[("eth0", "172.20.0.2"), ("eth1", "10.100.0.2")],
    ):
        assert ray_cluster._interface_for_ip("10.100.0.2") == "eth1"


def test_interface_for_ip_rejects_missing_address() -> None:
    with (
        patch.object(
            ray_cluster,
            "_ipv4_interfaces",
            return_value=[("eth0", "172.20.0.2")],
        ),
        pytest.raises(RuntimeError, match="10.100.0.2"),
    ):
        ray_cluster._interface_for_ip("10.100.0.2")


def test_start_ray_node_pins_nvshmem_to_cluster_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", raising=False)
    with (
        patch.object(ray_cluster, "_interface_for_ip", return_value="eth1"),
        patch.object(ray_cluster, "start_ray_head") as start_head,
    ):
        ray_cluster.start_ray_node(
            0,
            "10.100.0.1",
            "10.100.0.1",
            n_nodes=2,
            ray_port=6379,
        )

    assert os.environ["NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"] == "=eth1"
    start_head.assert_called_once_with("10.100.0.1", 2, ray_port=6379)


def test_start_ray_node_preserves_nvshmem_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", "=eth0")
    with (
        patch.object(ray_cluster, "_interface_for_ip", return_value="eth1"),
        patch.object(ray_cluster, "start_ray_worker") as start_worker,
    ):
        ray_cluster.start_ray_node(
            1,
            "10.100.0.1",
            "10.100.0.2",
            n_nodes=2,
            ray_port=6379,
            extra_env={"NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": "=ib0"},
        )

    assert os.environ["NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"] == "=ib0"
    start_worker.assert_called_once_with("10.100.0.2", "10.100.0.1", ray_port=6379)
