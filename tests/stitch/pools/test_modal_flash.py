"""ModalFlashPool harness: the pure URL/host normalization helpers. The Modal calls
(discover / wake / scale / gateway) are lazy and validated e2e."""

from __future__ import annotations

from stitch.pools.base import Pool
from stitch.pools.modal_flash import (
    ModalFlashFleet,
    ModalFlashPool,
    _host,
    _normalize_url,
    _replica_urls,
)
from stitch.types import VersionRef


def test_replica_request_routes_through_the_pool_url() -> None:
    pool = ModalFlashPool("app", "Server")
    pool._upstream_url_cache = "https://pool.modal.run"

    url, headers = pool.replica_request(
        "https://h-ta-123.w.modal.host/", "/server_info"
    )

    assert url == "https://pool.modal.run/server_info"
    assert headers == {"modal-flash-upstream": "h-ta-123.w.modal.host:443"}

    _url, headers = pool.replica_request("https://h-ta-123.w.modal.host:8443", "/wake")
    assert headers == {"modal-flash-upstream": "h-ta-123.w.modal.host:8443"}


def test_base_pool_replica_request_is_direct() -> None:
    url, headers = Pool().replica_request("https://replica:8000/", "/wake")

    assert url == "https://replica:8000/wake"
    assert headers == {}


def test_replica_urls_filters_hostless_and_normalizes() -> None:
    containers = [{"host": "h1:8000"}, {}, {"host": "https://h2/"}]
    assert _replica_urls(containers) == ["https://h1:8000", "https://h2"]


def test_normalize_url_adds_scheme_and_strips_slash() -> None:
    assert _normalize_url("host:8000") == "https://host:8000"
    assert _normalize_url("http://host:8000/") == "http://host:8000"
    assert _normalize_url("https://host/") == "https://host"


def test_host_reads_dict_or_attr() -> None:
    assert _host({"host": "h1"}) == "h1"
    assert _host({}) is None

    class _Container:
        host = "h2"

    assert _host(_Container()) == "h2"
    assert _host(object()) is None


def test_fleet_preserves_pool_identity_for_direct_requests_and_wake() -> None:
    class _Pool:
        def __init__(self, name):
            self.name = name
            self.woken = []

        def discover_replicas(self):
            return [f"https://{self.name}-replica"]

        def replica_request(self, replica, path):
            return f"{self.name}:{replica}{path}", {"pool": self.name}

        def wake(self, replicas, ref):
            self.woken.append((replicas, ref))

    fleet = ModalFlashFleet("app", ["H100", "B300"], gateway_function="router")
    fleet._pools = {name: _Pool(name) for name in fleet.cls_names}

    replicas = fleet.discover_replicas()
    assert replicas == [
        "H100|https://H100-replica",
        "B300|https://B300-replica",
    ]
    assert fleet.replica_request(replicas[1], "/server_info") == (
        "B300:https://B300-replica/server_info",
        {"pool": "B300"},
    )

    ref = VersionRef("run", 4)
    fleet.wake(replicas, ref)
    assert fleet._pools["H100"].woken == [(["https://H100-replica"], ref)]
    assert fleet._pools["B300"].woken == [(["https://B300-replica"], ref)]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"modal_flash harness: {len(tests)} PASS")
