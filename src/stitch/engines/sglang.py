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

from stitch.engines.base import Engine
from stitch.versions import VersionManifest, VersionRef


class SGLangEngine(Engine):
    def __init__(
        self,
        base_url: str,
        local_checkpoint_dir: str,
        *,
        control_timeout: float = 120.0,
        flush_cache_on_commit: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.local_checkpoint_dir = local_checkpoint_dir
        self._control_timeout = control_timeout
        self._flush_cache_on_commit = flush_cache_on_commit

    def base_url(self) -> str:
        return self._base_url

    def blocked_routes(self) -> frozenset[str]:
        # sglang's weight-update + scheduler-control endpoints; the sidecar drives these
        # directly, so external rollout traffic must never reach them.
        return frozenset({
            "update_weights_from_disk", "update_weights_from_distributed",
            "update_weights_from_tensor", "pull_weights", "flush_cache",
            "pause_generation", "continue_generation", "abort_request",
        })

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

    async def prefetch(self) -> None:
        # Seed the host-local checkpoint from the served base now (target_version=0): the
        # receiver copies its own model_path into local_checkpoint_dir and applies no deltas,
        # so the first real stage() only applies the delta rather than paying the full base
        # copy. Disk-only (serves throughout) and flock-serialized + idempotent with a
        # concurrent stage. source_dir is unused for the base seed (a placeholder here).
        await self._post(
            "/pull_weights",
            {"local_checkpoint_dir": self.local_checkpoint_dir, "source_dir": self.local_checkpoint_dir, "target_version": 0},
            timeout=None,
            action="base prefetch",
        )

    async def commit(self, ref: VersionRef) -> None:
        # flush_cache is configurable via flush_cache_on_commit. The reconciler already flush()es
        # before a quiesce reload and in_place deliberately keeps in-flight KV, so the default
        # is off — but whether the reload itself evicts the prefix/KV cache is a real per-run
        # choice (e.g. flush even in in_place), so a deployment can turn it on.
        await self._post(
            "/update_weights_from_disk",
            {"model_path": self.local_checkpoint_dir, "weight_version": str(ref.version),
             "flush_cache": self._flush_cache_on_commit},
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
        # Re-materialize base and reload it into the engine, so a run switch that lands at
        # v0 serves base on the GPU -- not the previous run's weights under the new
        # (run, 0) identity. Runs under the commit gate (engine paused). Ref: stitch#32.
        await asyncio.to_thread(shutil.rmtree, self.local_checkpoint_dir, ignore_errors=True)
        await self.prefetch()  # reseed base (target_version=0) into the wiped checkpoint
        await self._post(
            "/update_weights_from_disk",
            {"model_path": self.local_checkpoint_dir, "weight_version": "0",
             "flush_cache": self._flush_cache_on_commit},
            timeout=None,
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
        # Namespace the KV cache by version (and run, so two runs that both restart at
        # v1 stay distinct): requests on different versions can't share radix prefixes.
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
