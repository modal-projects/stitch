"""``UnionPool`` — one logical pool over several member pools.

A pool is normally one deployment class, but a fleet can be deliberately split
across classes — e.g. one Modal Flash class per region for a geo-distributed
rollout pool, since a single class's containers can't be pinned to distinct
regions. The union presents the members as one pool: discovery concatenates,
wake fans out to the member that owns each replica, scale applies to every
member (per-member floors/caps). ``gateway_url`` is the first member's — the
union is meant to sit *behind* a router that dispatches per-replica, so the
front door is only a fallback.
"""

from __future__ import annotations

from stitch.pools.base import Pool
from stitch.versions import VersionRef


class UnionPool(Pool):
    def __init__(self, pools: list[Pool]) -> None:
        if not pools:
            raise ValueError("UnionPool needs at least one member pool")
        self.pools = list(pools)

    def gateway_url(self) -> str:
        return self.pools[0].gateway_url()

    def discover_replicas(self) -> list[str]:
        seen: dict[str, None] = {}
        for pool in self.pools:
            for url in pool.discover_replicas():
                seen.setdefault(url, None)
        return list(seen)

    def wake(self, replicas: list[str], ref: VersionRef) -> None:
        # Ownership is discovered live: each member wakes the subset it reports.
        # A replica no member claims (mid-churn) is woken via the first member —
        # ModalFlashPool.wake only POSTs to the URL, so any member can deliver it.
        remaining = list(replicas)
        for pool in self.pools:
            if not remaining:
                return
            mine = set(pool.discover_replicas()) & set(remaining)
            if mine:
                pool.wake(sorted(mine), ref)
                remaining = [u for u in remaining if u not in mine]
        if remaining:
            self.pools[0].wake(remaining, ref)

    def scale(self, *, min: int | None = None, max: int | None = None) -> None:
        for pool in self.pools:
            pool.scale(min=min, max=max)
