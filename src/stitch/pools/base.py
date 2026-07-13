"""The ``Pool`` port — a client to the running elastic replica pool.

Instances: ``pools/modal_flash.py`` (Modal Flash). Add k8s as a new file.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from stitch.versions import VersionRef


@runtime_checkable
class Pool(Protocol):
    """Reach, enumerate, and scale the serving replicas — not their deployment.
    ``wake`` and ``scale`` are optional; polling and load-autoscale are the fallbacks."""

    def gateway_url(self) -> str: ...

    def discover_replicas(self) -> list[str]: ...

    def wake(self, replicas: list[str], ref: VersionRef) -> None: ...

    def scale(self, *, min: int | None = None, max: int | None = None) -> None: ...
