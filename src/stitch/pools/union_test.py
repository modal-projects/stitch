from stitch.pools.base import Pool
from stitch.pools.union import UnionPool
from stitch.versions import VersionRef


class FakePool(Pool):
    def __init__(self, name: str, urls: list[str]):
        self.name = name
        self.urls = urls
        self.woken: list[list[str]] = []
        self.scaled: list[tuple] = []

    def gateway_url(self) -> str:
        return f"http://{self.name}"

    def discover_replicas(self) -> list[str]:
        return list(self.urls)

    def wake(self, replicas, ref):
        self.woken.append(sorted(replicas))

    def scale(self, *, min=None, max=None):
        self.scaled.append((min, max))


REF = VersionRef("run", 1)


def test_discover_concatenates_and_dedupes():
    a = FakePool("a", ["http://r1", "http://r2"])
    b = FakePool("b", ["http://r2", "http://r3"])
    assert UnionPool([a, b]).discover_replicas() == ["http://r1", "http://r2", "http://r3"]


def test_gateway_is_first_member():
    assert UnionPool([FakePool("a", []), FakePool("b", [])]).gateway_url() == "http://a"


def test_wake_fans_out_by_ownership():
    a = FakePool("a", ["http://r1"])
    b = FakePool("b", ["http://r2"])
    UnionPool([a, b]).wake(["http://r1", "http://r2"], REF)
    assert a.woken == [["http://r1"]]
    assert b.woken == [["http://r2"]]


def test_wake_unclaimed_falls_back_to_first():
    a = FakePool("a", ["http://r1"])
    b = FakePool("b", [])
    UnionPool([a, b]).wake(["http://r1", "http://gone"], REF)
    assert a.woken == [["http://r1"], ["http://gone"]]


def test_scale_applies_to_every_member():
    a, b = FakePool("a", []), FakePool("b", [])
    UnionPool([a, b]).scale(min=1, max=2)
    assert a.scaled == [(1, 2)] and b.scaled == [(1, 2)]


def test_empty_union_rejected():
    try:
        UnionPool([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
