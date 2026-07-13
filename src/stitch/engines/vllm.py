"""``VLLMEngine`` — the ``Engine`` instance for a vLLM server. **TODO: not implemented.**

Scaffold only. ``engines/sglang.py`` is the reference implementation; a vLLM engine slots
in behind the same ``Engine`` port with zero core changes. The work is mapping stage/commit
onto vLLM's weight-update surface, which differs from sglang's ``/pull_weights`` +
``/update_weights_from_disk`` — vLLM has no built-in disk-delta receiver, so the host-side
apply (walk to the nearest anchor, replay xor/overwrite deltas, checksum-verify) has to live
either in a worker extension or a small sidecar step before the reload. Each method below
notes what its vLLM equivalent should do; fill them in when we bring a vLLM pool up.
"""

from __future__ import annotations

from typing import Any

from stitch.engines.base import Engine
from stitch.versions import VersionManifest, VersionRef


class VLLMEngine(Engine):
    def __init__(
        self,
        base_url: str,
        local_checkpoint_dir: str,
        *,
        control_timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.local_checkpoint_dir = local_checkpoint_dir
        self._control_timeout = control_timeout

    def base_url(self) -> str:
        return self._base_url

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        # TODO: bring the host-local checkpoint to manifest.ref — seed from the nearest FULL
        # anchor, replay deltas forward, checksum-verify. vLLM lacks sglang's /pull_weights, so
        # this apply must be provided (worker extension or a sidecar copy+decode) before commit.
        raise NotImplementedError("VLLMEngine.stage: TODO")

    async def commit(self, ref: VersionRef) -> None:
        # TODO: reload the staged checkpoint into the serving weights (the gate covers only this)
        # — e.g. vLLM collective_rpc into a WorkerExtension that runs load_weights, or vLLM's
        # runtime weight-update API. Must reproduce an initial load for the served quant format.
        raise NotImplementedError("VLLMEngine.commit: TODO")

    async def flush(self) -> None:
        # TODO: evict KV / prefix cache before a quiesce reload.
        raise NotImplementedError("VLLMEngine.flush: TODO")

    async def pause(self) -> None:
        # TODO: pause the scheduler in place (in_place commit), keeping in-flight requests resident.
        raise NotImplementedError("VLLMEngine.pause: TODO")

    async def resume(self) -> None:
        # TODO: resume the scheduler after a pause.
        raise NotImplementedError("VLLMEngine.resume: TODO")

    async def reset(self) -> None:
        # TODO: wipe local_checkpoint_dir so the next stage reseeds from the engine's boot base.
        raise NotImplementedError("VLLMEngine.reset: TODO")

    # prefetch() intentionally inherits the Engine no-op default until stage() exists.

    def stamp_request(self, request: dict[str, Any], served: VersionRef) -> None:
        # TODO: namespace the request to the served version for KV isolation (vLLM equivalent of
        # sglang's extra_key — e.g. a prefix-cache salt / request tag). Mutates request in place.
        raise NotImplementedError("VLLMEngine.stamp_request: TODO")

    def stamp_response(self, response: dict[str, Any], served: VersionRef, current: VersionRef) -> None:
        # TODO: record which version generated the response, in vLLM's response shape.
        raise NotImplementedError("VLLMEngine.stamp_response: TODO")
