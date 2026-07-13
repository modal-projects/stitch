"""The ``Engine`` port — a client to one inference engine.

Instances: ``engines/sglang.py`` (sglang). Add vllm as a new file.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from stitch.versions import VersionConstraint, VersionManifest, VersionRef


@runtime_checkable
class Engine(Protocol):
    """Drives one engine and translates the version protocol; the heavy weight
    apply runs inside the engine, not here."""

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        """Bring the local checkpoint to ``manifest.ref``: seed from the nearest FULL
        anchor, then replay deltas forward. May run while the engine serves."""
        ...

    async def commit(self, ref: VersionRef) -> None:
        """Reload the staged checkpoint into the serving weights — the gate covers only this."""
        ...

    async def flush(self) -> None:
        """Evict cached state (KV / radix tree). Called before commit in quiesce mode."""
        ...

    async def pause(self) -> None:
        """Pause the scheduler in place (in_place commit); in-flight requests stay resident."""
        ...

    async def resume(self) -> None:
        """Resume the scheduler after a pause."""
        ...

    async def reset(self) -> None:
        """Reseed the local checkpoint to the engine's boot base."""
        ...

    async def applied_version(self) -> VersionRef | None: ...

    def stamp_request(
        self, request: dict[str, Any], constraint: VersionConstraint
    ) -> dict[str, Any]:
        """Stamp the version constraint onto a request (engine-specific, e.g. sglang's extra_key)."""
        ...

    def read_response_version(self, response: dict[str, Any]) -> VersionRef | None: ...

    def generate_url(self) -> str: ...
