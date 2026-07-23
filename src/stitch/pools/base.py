"""The ``Pool`` port — a client to the running elastic replica pool.

Instances subclass this base: ``pools/modal_flash.py`` (Modal Flash). Add k8s as a new
subclass. Override ``gateway_url`` and ``discover_replicas``; ``wake`` and ``scale`` are
optional (their no-op defaults fall back to the replicas' own polling / load-autoscale).

The ``*_async`` variants serve callers running on an event loop (an async control plane,
``service.readiness``). Their defaults thread the sync implementation, so every subclass is
usable from async code as-is — but a pool whose client has a native async surface should
override them to ride it directly instead of parking a worker thread per call.
"""

from __future__ import annotations

import asyncio

from stitch.types import VersionRef


class Pool:
    """Reach, enumerate, and (optionally) nudge/scale the serving replicas — a client
    to a running pool, not its deployment."""

    def gateway_url(self) -> str:
        """The single URL rollout traffic is sent to (the pool's front door)."""
        raise NotImplementedError

    def discover_replicas(self) -> list[str]:
        """Base URLs of the currently-live replicas — a point-in-time snapshot of a *dynamic*
        pool (replicas join and leave as it autoscales), so the list can already be stale when
        it returns. Callers must tolerate that: ``wake`` is best-effort per URL, and a replica
        appearing or disappearing between discovery and use is expected, not an error."""
        raise NotImplementedError

    def wake(self, replicas: list[str], ref: VersionRef) -> None:
        """Nudge replicas to reconcile now. Optional — the default relies on their polling."""

    def scale(self, *, min: int | None = None, max: int | None = None) -> None:
        """Adjust the replica floor/cap. Optional — the default relies on load-autoscale."""

    async def gateway_url_async(self) -> str:
        """``gateway_url`` for async callers (default: the sync impl, off-loop)."""
        return await asyncio.to_thread(self.gateway_url)

    async def discover_replicas_async(self) -> list[str]:
        """``discover_replicas`` for async callers (default: the sync impl, off-loop)."""
        return await asyncio.to_thread(self.discover_replicas)

    async def wake_async(self, replicas: list[str], ref: VersionRef) -> None:
        """``wake`` for async callers (default: the sync impl, off-loop)."""
        await asyncio.to_thread(self.wake, replicas, ref)
