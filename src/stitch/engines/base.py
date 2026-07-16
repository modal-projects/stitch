"""The ``Engine`` port — a client to one inference engine.

``engines/sglang.py`` is the working instance; ``engines/vllm.py`` sketches the vLLM
shape. Subclasses override the methods they use — ``prefetch`` and ``blocked_routes``
have safe defaults.
"""

from __future__ import annotations

from typing import Any

from stitch.versions import VersionManifest, VersionRef


class Engine:
    """Drives one engine and translates the version protocol; the heavy weight apply
    runs inside the engine, not here."""

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        """Bring the local checkpoint to ``manifest.ref``: seed from the nearest FULL
        anchor, then replay deltas forward. May run while the engine serves."""
        raise NotImplementedError

    async def commit(self, ref: VersionRef) -> None:
        """Reload the staged checkpoint into the serving weights — the gate covers only this."""
        raise NotImplementedError

    async def flush(self) -> None:
        """Evict cached state (KV / radix tree). Called before commit in quiesce mode."""
        raise NotImplementedError

    async def pause(self) -> None:
        """Pause the scheduler in place (in_place commit); in-flight requests stay resident."""
        raise NotImplementedError

    async def resume(self) -> None:
        """Resume the scheduler after a pause."""
        raise NotImplementedError

    async def reset(self) -> None:
        """Reseed the local checkpoint to the engine's boot base."""
        raise NotImplementedError

    async def prefetch(self) -> None:
        """Optional: seed the host-local checkpoint from the engine's base ahead of the first
        stage(), so stage only applies the delta instead of copying the full base off the
        critical path. Default no-op — an engine with no host-local checkpoint needs nothing."""
        return

    def stamp_request(self, request: dict[str, Any], served: VersionRef) -> None:
        """Namespace a request to the version it's served on so requests from different
        versions can't share KV prefixes (engine-specific, e.g. sglang's extra_key).
        Mutates ``request`` in place."""
        raise NotImplementedError

    def stamp_response(self, response: dict[str, Any], served: VersionRef, current: VersionRef) -> None:
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
