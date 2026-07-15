"""The ``Pool`` port — a client to the running elastic replica pool.

Instances subclass this base: ``pools/modal_flash.py`` (Modal Flash). Add k8s as a new
subclass. Override ``gateway_url`` and ``discover_replicas``; ``wake`` and ``scale`` are
optional (their no-op defaults fall back to the replicas' own polling / load-autoscale).
"""

from __future__ import annotations

from stitch.versions import VersionRef


class Pool:
    """Reach, enumerate, and (optionally) nudge/scale the serving replicas — a client
    to a running pool, not its deployment."""

    def gateway_url(self) -> str:
        """The single URL rollout traffic is sent to (the pool's front door)."""
        raise NotImplementedError

    def discover_replicas(self) -> list[str]:
        """Base URLs of the currently-live replicas."""
        raise NotImplementedError

    def wake(self, replicas: list[str], ref: VersionRef) -> None:
        """Nudge replicas to reconcile now. Optional — the default relies on their polling."""

    def scale(self, *, min: int | None = None, max: int | None = None) -> None:
        """Adjust the replica floor/cap. Optional — the default relies on load-autoscale."""
