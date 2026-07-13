"""``SGLangEngine`` — the ``Engine`` instance for a single sglang server.

The weight apply lives inside sglang (``weight_sync/local_checkpoint``): ``stage``
POSTs ``/pull_weights``, which chain-replays deltas from the applied checkpoint with
per-tensor checksum verification (reseeding from base on corruption), and ``commit``
reloads the materialized checkpoint via ``/update_weights_from_disk``. This client
only drives those endpoints and translates the version protocol — it holds no
host-side decoder and imports no trainer package.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from stitch.versions import VersionManifest, VersionRef

_EXTRA_KEY_DELIM = ";"


class SGLangEngine:
    def __init__(self, upstream_url: str, local_checkpoint_dir: str, *, control_timeout: float = 120.0) -> None:
        self._upstream = upstream_url.rstrip("/")
        self.local_checkpoint_dir = local_checkpoint_dir
        self._control_timeout = control_timeout

    def upstream_url(self) -> str:
        return self._upstream

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        # source_dir is the target version's dir; its parent is the root of weight_v*
        # dirs the pull walks. Disk-only, so it runs while the engine serves.
        await self._post(
            "/pull_weights",
            {
                "local_checkpoint_dir": self.local_checkpoint_dir,
                "source_dir": str(Path(source_dir).parent),
                "target_version": manifest.ref.version,
            },
            timeout=None,
            action="weight pull",
        )

    async def commit(self, ref: VersionRef) -> None:
        # flush_cache stays off here: the reconciler flushes (quiesce) while drained,
        # and the engine's own post-apply flush hard-asserts — killing the scheduler —
        # if a request slipped in.
        await self._post(
            "/update_weights_from_disk",
            {"model_path": self.local_checkpoint_dir, "weight_version": str(ref.version), "flush_cache": False},
            timeout=None,
            action="weight update",
        )

    async def flush(self) -> None:
        await self._get("/flush_cache", ok=(200, 404))

    async def pause(self) -> None:
        await self._post("/pause_generation", {"mode": "in_place"}, timeout=self._control_timeout)

    async def resume(self) -> None:
        await self._post("/continue_generation", {}, timeout=self._control_timeout)

    async def reset(self) -> None:
        # Wipe the local checkpoint so the next pull reseeds from the engine's base
        # rather than chaining a new run's deltas onto the old run's bytes.
        await asyncio.to_thread(shutil.rmtree, self.local_checkpoint_dir, ignore_errors=True)

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
        # Namespace the KV cache by version (and run, so two runs that both restart at
        # v1 stay distinct): requests on different versions can't share radix prefixes.
        run = f"{served.run_id}/" if served.run_id else ""
        return f"wv{served.version}{_EXTRA_KEY_DELIM}{run}{user or ''}"

    async def _post(self, path: str, payload: dict[str, Any], *, timeout: float | None, action: str | None = None) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(f"{self._upstream}{path}", json=payload)
        _raise_for_engine(resp, action or path)

    async def _get(self, path: str, *, ok: tuple[int, ...] = (200,)) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=self._control_timeout, trust_env=False) as client:
            resp = await client.get(f"{self._upstream}{path}")
        if resp.status_code not in ok:
            _raise_for_engine(resp, path)


def _raise_for_engine(resp: Any, action: str) -> None:
    # A failed control call comes back as HTTP 4xx with the engine's traceback in the
    # JSON body — read the body before the status so the real error isn't lost.
    try:
        data = resp.json()
        if not isinstance(data, dict):
            data = {"message": data}
    except ValueError:
        data = {"message": resp.text}
    if resp.status_code != 200 or data.get("success") is False:
        raise RuntimeError(f"sglang rejected {action} (HTTP {resp.status_code}): {data.get('message', data)}")
