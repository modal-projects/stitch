"""The ``Engine`` port — a client to one inference engine.

``engines/sglang.py`` is the working instance; ``engines/vllm.py`` sketches the vLLM
shape. Subclasses override the methods they use —
``initialize_update_destination`` and ``blocked_routes`` have safe defaults.
"""

from __future__ import annotations

from typing import Any

from stitch.types import VersionManifest, VersionRef


class Engine:
    """Drives one engine and translates the version protocol; the heavy weight apply
    runs inside the engine, not here."""

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        """Bring the local checkpoint to ``manifest.ref``: seed from the nearest FULL
        anchor, then apply deltas forward. May run while the engine serves."""
        raise NotImplementedError

    async def commit(
        self, manifest: VersionManifest, *, flush_cache: bool = False
    ) -> None:
        """Apply the staged checkpoint to the serving weights; the gate covers only this.
        ``flush_cache`` (a commit-policy decision the reconciler passes) evicts the engine's
        prefix/KV cache as part of the commit."""
        raise NotImplementedError

    async def flush_cache(self) -> None:
        """Evict the engine's prefix/KV cache — the standalone ``/flush_cache`` primitive.
        Commit-time flushing goes through ``commit(flush_cache=…)`` instead."""
        raise NotImplementedError

    async def pause(self) -> None:
        """Pause the scheduler in place (in_place commit); in-flight requests stay resident."""
        raise NotImplementedError

    async def resume(self) -> None:
        """Resume the scheduler after a pause."""
        raise NotImplementedError

    async def reset(self) -> None:
        """Restore the engine to its boot weights."""
        raise NotImplementedError

    async def initialize_update_destination(self) -> None:
        """Initialize engine state required before staging updates.

        This may run while the engine serves its boot weights. A reconciler waits
        for it before staging the first published update.
        """
        return

    def stamp_request(self, request: dict[str, Any], served: VersionRef) -> None:
        """Namespace a request to the version it's served on so requests from different
        versions can't share KV prefixes (engine-specific, e.g. sglang's extra_key).
        Mutates ``request`` in place."""
        raise NotImplementedError

    def stamp_response(
        self, response: dict[str, Any], served: VersionRef, current: VersionRef
    ) -> None:
        """Record which version generated a response, in the engine's response shape
        (e.g. sglang's meta_info vs OpenAI top-level). Mutates ``response`` in place."""
        raise NotImplementedError

    def base_url(self) -> str:
        """The engine's base HTTP URL — the proxy forwards to it, and the engine's own
        stage/commit calls target it."""
        raise NotImplementedError

    def blocked_routes(self) -> frozenset[str]:
        """Engine control routes the versioned proxy must never forward: a stray external
        call would mutate engine state behind the reconciler's back. Default: none."""
        return frozenset()
