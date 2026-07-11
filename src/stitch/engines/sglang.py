"""SGLang rollout engine adapters."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stitch.protocol import VersionManifest


logger = logging.getLogger(__name__)
EXTRA_KEY_DELIMITER = ";"


def parse_reload_timing(message: str) -> dict[str, float]:
    """Lift the engine's optional ``[reload timing] iter_wait=1.2s load=3.4s ...``
    success-message suffix (emitted by the instrumented modal-projects/sglang
    fork) into ``{"engine_iter_wait_s": 1.2, ...}``. Empty on engines without
    the instrumentation, so callers can treat the result as best-effort."""
    match = re.search(r"\[reload timing\]([^\[\]]*)", message)
    if match is None:
        return {}
    return {
        f"engine_{key}_s": float(value)
        for key, value in re.findall(r"(\w+)=([0-9.]+)s", match.group(1))
    }


def _response_payload(resp: Any) -> dict[str, Any]:
    """The engine's JSON body when there is one, else the raw text under
    ``message`` — error responses must never lose the engine's explanation."""
    try:
        data = resp.json()
    except ValueError:
        return {"message": resp.text}
    return data if isinstance(data, dict) else {"message": data}


def compose_extra_key(
    version: int, user_extra_key: str | None = None, run_id: str | None = None
) -> str:
    """Compose a weight-version-namespaced SGLang radix-cache ``extra_key``,
    e.g. ``wv7;my-key`` or ``wv1;run-a/my-key``.

    The version prefix namespaces the KV cache per weight version; folding
    ``run_id`` into the key content keeps two runs that both restart version
    numbering at 1 in distinct namespaces, so a stale cross-run request can
    never reuse another run's same-numbered KV.
    """
    run_segment = f"{run_id}/" if run_id else ""
    return f"wv{int(version)}{EXTRA_KEY_DELIMITER}{run_segment}{user_extra_key or ''}"


@dataclass
class SGLangDiskDeltaAdapter:
    """Syncs disk-delta weight versions into one local SGLang server via the
    engine's ``/pull_weights`` endpoint.

    The delta apply lives in the ENGINE (sglang ``weight_sync/local_checkpoint``):
    ``stage_manifest`` POSTs ``/pull_weights``, which materializes the host-local
    checkpoint — seeded from the engine's own ``model_path``, chain-replayed with
    per-tensor checksum verification, and self-recovering (a torn or corrupt
    local state is reseeded from base and replayed, never re-patched). Multi-
    version catch-up is native: one pull chains every delta up to the target.
    ``commit_manifest`` then reloads the materialized checkpoint through the
    ordinary ``update_weights_from_disk`` path.

    The sidecar holds no host-side decoder — no trainer package (slime/miles)
    import at all. Cross-host object-store visibility is the engine's concern
    too (``--custom-pull-weights-pre-read-hook`` on the server).
    """

    upstream_url: str
    local_checkpoint_dir: str
    backend: str = "disk_delta"

    def __post_init__(self) -> None:
        self.upstream_url = self.upstream_url.rstrip("/")

    async def reset(self) -> None:
        """Discard the local checkpoint so the next pull reseeds from scratch.

        Used on a run switch: the new run's chain forks at base, and version
        numbers restart, so the engine-side pull cannot be allowed to chain the
        new run's deltas onto the old run's bytes. Wiping the directory makes
        the next ``/pull_weights`` take the fresh-host path (full seed from the
        engine's base, then the new chain). The sync manager invokes this under
        the commit gate (engine paused), so no reload races the wipe.
        """
        import shutil

        await asyncio.to_thread(shutil.rmtree, self.local_checkpoint_dir, ignore_errors=True)

    async def flush_cache(self) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            resp = await client.get(f"{self.upstream_url}/flush_cache")
            if resp.status_code not in (200, 404):
                resp.raise_for_status()

    async def pause_generation(self) -> None:
        """Pause the scheduler loop in place: in-flight requests stay resident
        and resume decoding on their existing KV after continue_generation."""
        import httpx

        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            resp = await client.post(f"{self.upstream_url}/pause_generation", json={"mode": "in_place"})
            resp.raise_for_status()

    async def continue_generation(self) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            resp = await client.post(f"{self.upstream_url}/continue_generation", json={})
            resp.raise_for_status()

    async def stage_manifest(self, manifest: VersionManifest, version_path: str) -> dict[str, Any] | None:
        """Bring the local checkpoint up to this version via the engine's
        ``/pull_weights`` (chain-replayed from whatever is applied, per-tensor
        checksum-verified, reseed-on-corruption).

        Safe to run while the engine serves: the pull is disk-only (weights live
        on the GPU), so the sync manager calls this BEFORE closing the commit
        gate. version_path is the published version dir; its parent is the root
        of weight_v* dirs the pull walks.
        """
        import httpx

        delta_root = str(Path(version_path).parent)
        _t_apply = time.perf_counter()
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            resp = await client.post(
                f"{self.upstream_url}/pull_weights",
                json={
                    "local_checkpoint_dir": self.local_checkpoint_dir,
                    "source_dir": delta_root,
                    "target_version": int(manifest.version),
                },
            )
            # A failed pull comes back as HTTP 400 with the engine's traceback
            # in the JSON body — read it before checking status or the actual
            # error is lost behind a bare status code.
            data = _response_payload(resp)
            if resp.status_code != 200 or data.get("success") is False:
                raise RuntimeError(
                    f"SGLang rejected weight pull (HTTP {resp.status_code}): {data.get('message', data)}"
                )
        elapsed = time.perf_counter() - _t_apply
        logger.info("[apply timing] v=%s engine_pull=%.2fs", manifest.version, elapsed)
        return {"engine_pull_s": round(elapsed, 3)}

    async def commit_manifest(
        self,
        manifest: VersionManifest,
        version_path: str,
        weight_names: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Reload the staged local checkpoint into the engine. The sync
        manager calls this under the commit gate (engine paused in in_place
        mode), after :meth:`stage_manifest` has materialized the version.

        ``weight_names`` is the tail's union of touched tensor names; an
        engine with the partial-reload load plan uses it to reload only the
        touched modules (O(delta)) and silently full-reloads otherwise. The
        pinned engine has no partial reload (full reloads re-derive quantized
        kernel state correctly; partial needs a per-module restore design), so
        the names are withheld unless ``STITCH_PARTIAL_RELOAD=1`` opts in."""
        import os

        import httpx

        _t_reload = time.perf_counter()
        payload: dict[str, Any] = {
            "model_path": self.local_checkpoint_dir,
            "weight_version": str(manifest.version),
            # The sync manager flushes via GET /flush_cache while quiesced.
            # The engine-side post-apply flush hard-asserts on failure
            # (killing the scheduler process) if any request slipped in, so
            # it must stay disabled here.
            "flush_cache": False,
        }
        if weight_names and os.environ.get("STITCH_PARTIAL_RELOAD", "0") == "1":
            payload["weight_names"] = list(weight_names)
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            resp = await client.post(f"{self.upstream_url}/update_weights_from_disk", json=payload)
            # Failures come back as HTTP 400 with the engine's error in the
            # JSON body — read it before checking status.
            data = _response_payload(resp)
            if resp.status_code != 200 or data.get("success") is False:
                raise RuntimeError(
                    f"SGLang rejected weight update (HTTP {resp.status_code}): {data.get('message', data)}"
                )
        elapsed = time.perf_counter() - _t_reload
        logger.info("[apply timing] v=%s engine_reload=%.2fs", manifest.version, elapsed)
        detail: dict[str, Any] = {"engine_reload_s": round(elapsed, 3)}
        detail.update(parse_reload_timing(str(data.get("message") or "")))
        return detail
