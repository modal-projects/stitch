"""``SGLangEngine`` — the ``Engine`` instance for a single sglang server.

``stage`` always asks sglang to reconstruct and checksum a canonical host-local
checkpoint. In ``disk`` mode, ``commit`` reloads that checkpoint through the ordinary
model loader. In ``host_runtime`` mode, staging also advances one persistent pinned
CPU image in final runtime layout, and commit performs only its full CPU-to-GPU copy.
This client drives those endpoints and translates the version protocol; it holds no
host-side decoder and imports no trainer package.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any, Literal

from stitch.engines.base import Engine
from stitch.types import VersionManifest, VersionRef

WeightUpdateMode = Literal["disk", "host_runtime"]


class SGLangEngine(Engine):
    def __init__(
        self,
        base_url: str,
        local_checkpoint_dir: str,
        *,
        weight_update_mode: WeightUpdateMode = "disk",
        control_timeout: float = 120.0,
        reload_timeout: float = 600.0,
    ) -> None:
        if weight_update_mode not in ("disk", "host_runtime"):
            raise ValueError(f"unsupported SGLang weight update mode: {weight_update_mode!r}")
        self._base_url = base_url.rstrip("/")
        self.local_checkpoint_dir = local_checkpoint_dir
        self.weight_update_mode = weight_update_mode
        self._control_timeout = control_timeout
        self._reload_timeout = reload_timeout

    def base_url(self) -> str:
        return self._base_url

    def blocked_routes(self) -> frozenset[str]:
        return frozenset({
            "update_weights_from_disk", "update_weights_from_distributed",
            "update_weights_from_tensor", "update_weights_from_prepared",
            "pull_weights", "flush_cache",
            "pause_generation", "continue_generation", "abort_request",
        })

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        # /pull_weights walks the weight_v* dirs under source_dir's parent.
        await self._post(
            "/pull_weights",
            {
                "local_checkpoint_dir": self.local_checkpoint_dir,
                "source_dir": str(Path(source_dir).parent),
                "target_version": manifest.ref.version,
                "prepare": "runtime" if self.weight_update_mode == "host_runtime" else "checkpoint",
            },
            timeout=self._reload_timeout,
            action="weight pull",
        )

    async def prefetch(self) -> None:
        # target_version=0 seeds base with no deltas applied; source_dir is unused for the seed.
        await self._post(
            "/pull_weights",
            {
                "local_checkpoint_dir": self.local_checkpoint_dir,
                "source_dir": self.local_checkpoint_dir,
                "target_version": 0,
                # A base seed is a checkpoint operation. On cold start the host
                # image was captured from the initial GPU model; on a run reset
                # the disk reload below explicitly recaptures it.
                "prepare": "checkpoint",
            },
            timeout=self._reload_timeout,
            action="base prefetch",
        )

    async def commit(
        self, ref: VersionRef, *, flush_cache: bool = False, weight_names: list[str] | None = None
    ) -> None:
        # SGLang reloads are deliberately dense. A checkpoint delta may be
        # element-wise sparse while touching every runtime tensor, and fused or
        # derived runtime storages do not have a safe one-to-one correspondence
        # with checkpoint tensor names. Keep weight_names in the shared Engine
        # interface for other backends, but never narrow an SGLang commit with it.
        del weight_names
        if self.weight_update_mode == "host_runtime":
            await self._post(
                "/update_weights_from_prepared",
                {"weight_version": str(ref.version), "flush_cache": flush_cache},
                timeout=self._reload_timeout,
                action="prepared weight update",
            )
        else:
            payload: dict[str, Any] = {
                "model_path": self.local_checkpoint_dir,
                "weight_version": str(ref.version),
                "flush_cache": flush_cache,
            }
            await self._post(
                "/update_weights_from_disk",
                payload,
                timeout=self._reload_timeout,
                action="disk weight update",
            )

    async def flush_cache(self) -> None:
        await self._get("/flush_cache", ok=(200, 404))

    async def pause(self) -> None:
        await self._post("/pause_generation", {"mode": "in_place"}, timeout=self._control_timeout)

    async def resume(self) -> None:
        await self._post("/continue_generation", {}, timeout=self._control_timeout)

    async def reset(self) -> None:
        # Wipe + reseed base so a run switch to v0 serves base, not the prior run's weights
        # under the new (run, 0) identity (stitch#32).
        await asyncio.to_thread(shutil.rmtree, self.local_checkpoint_dir, ignore_errors=True)
        await self.prefetch()
        await self._post(
            "/update_weights_from_disk",
            {
                "model_path": self.local_checkpoint_dir,
                "weight_version": "0",
                "flush_cache": False,
                "refresh_host_runtime": self.weight_update_mode == "host_runtime",
            },
            timeout=self._reload_timeout,
            action="reset reload to base",
        )

    def stamp_request(self, request: dict[str, Any], served: VersionRef) -> None:
        user = request.get("extra_key")
        if isinstance(user, list):
            request["extra_key"] = [self._extra_key(served, k) for k in user]
        else:
            request["extra_key"] = self._extra_key(served, user)

    def stamp_response(self, response: dict[str, Any], served: VersionRef, current: VersionRef) -> None:
        meta = response.get("meta_info")
        if isinstance(meta, dict):  # sglang /generate carries attribution in meta_info
            meta["weight_version"] = str(served.version)
            meta["weight_version_start"] = served.version
            meta["weight_version_end"] = current.version
        else:  # OpenAI-style routes at the top level
            response["weight_version_start"] = served.version
            response["weight_version_end"] = current.version

    def _extra_key(self, served: VersionRef, user: str | None) -> str:
        # Namespace the KV cache by version+run so radix prefixes aren't shared across versions.
        run = f"{served.run_id}/" if served.run_id else ""
        return f"wv{served.version};{run}{user or ''}"

    async def _post(self, path: str, payload: dict[str, Any], *, timeout: float | None, action: str | None = None) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(f"{self._base_url}{path}", json=payload)
        _raise_for_engine(resp, action or path)

    async def _get(self, path: str, *, ok: tuple[int, ...] = (200,)) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=self._control_timeout, trust_env=False) as client:
            resp = await client.get(f"{self._base_url}{path}")
        if resp.status_code not in ok:
            _raise_for_engine(resp, path)


def _raise_for_engine(resp: Any, action: str) -> None:
    # sglang puts the real error in the JSON body on 4xx — read it before the status.
    try:
        data = resp.json()
        if not isinstance(data, dict):
            data = {"message": data}
    except ValueError:
        data = {"message": resp.text}
    if resp.status_code != 200 or data.get("success") is False:
        raise RuntimeError(f"sglang rejected {action} (HTTP {resp.status_code}): {data.get('message', data)}")
