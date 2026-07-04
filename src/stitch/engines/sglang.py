"""SGLang rollout engine adapters."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from stitch.protocol import VersionManifest


logger = logging.getLogger(__name__)
EXTRA_KEY_DELIMITER = ";"


def compose_extra_key(
    version: int, user_extra_key: str | None = None, run_id: str | None = None
) -> str:
    """Compose a weight-version-namespaced SGLang ``extra_key``.

    The version segment sits at a fixed position (the prefix) and is
    delimiter-terminated, so it parses unambiguously regardless of the user
    key's content. sglang appends ``lora_id`` to ``extra_key`` with no
    delimiter, so the version must never be parsed from the right.
    Examples: ``wv7;`` (no user key), ``wv7;my-key``. This is an SGLang
    radix-cache namespacing concern, not part of the engine-neutral protocol.

    ``run_id`` is folded into the key *content* (after the delimiter), so two
    runs that both restart version numbering at 1 get distinct radix namespaces
    (``wv1;run-a/`` vs ``wv1;run-b/``) and a stale cross-run request can never
    reuse another run's same-numbered KV. ``run_id=None`` keeps the bare form.
    """
    run_segment = f"{run_id}/" if run_id else ""
    return f"wv{int(version)}{EXTRA_KEY_DELIMITER}{run_segment}{user_extra_key or ''}"


@dataclass
class SGLangDiskDeltaAdapter:
    """Applies disk-delta weight versions to one local SGLang server.

    The delta is applied *host-side*: slime's ``disk_delta`` patches a local
    full HF checkpoint in place (chain-replayed from the base, per-tensor
    checksum-verified, with a base-version precondition), and the engine then
    reloads that checkpoint through the ordinary ``update_weights_from_disk``
    path. The engine carries no delta receiver, so no ``load_format`` / ``files``
    delta payload is sent — that is all the new disk-delta slime branch needs.

    ``apply_deltas`` / ``init_local_checkpoint`` are injectable so the adapter is
    testable without slime/numpy installed; by default they bind lazily to
    ``slime.utils.disk_delta``.
    """

    upstream_url: str
    local_checkpoint_dir: str
    base_checkpoint_dir: str
    backend: str = "disk_delta"
    apply_deltas: Callable[[str, str, int], None] | None = None
    init_local_checkpoint: Callable[[str, str], None] | None = None

    def __post_init__(self) -> None:
        self.upstream_url = self.upstream_url.rstrip("/")

    def _apply_deltas(self) -> Callable[[str, str, int], None]:
        if self.apply_deltas is not None:
            return self.apply_deltas
        from slime.utils.disk_delta import apply_deltas

        return apply_deltas

    def _init_local_checkpoint(self) -> Callable[[str, str], None]:
        if self.init_local_checkpoint is not None:
            return self.init_local_checkpoint
        from slime.utils.disk_delta import init_local_checkpoint

        return init_local_checkpoint

    async def prepare(self) -> None:
        """Materialize the host-local full checkpoint from the base once
        (idempotent) so later deltas apply on top of it in place. Run at startup;
        blocking copy, so it is offloaded to a thread."""
        await asyncio.to_thread(
            self._init_local_checkpoint(), self.local_checkpoint_dir, self.base_checkpoint_dir
        )

    async def reset(self) -> None:
        """Discard the local checkpoint and re-materialize the base from scratch.

        Used on a run switch: the new run's chain forks at base, so the locally
        patched checkpoint must be thrown away before replaying the new chain. The
        sync manager invokes this under the commit gate (engine paused), so no
        request decodes across the wipe; ``prepare`` then rebuilds a complete base
        (a partial wipe from a crash is re-seeded by ``init_local_checkpoint``).
        """
        import shutil

        await asyncio.to_thread(shutil.rmtree, self.local_checkpoint_dir, ignore_errors=True)
        await self.prepare()

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

    async def apply_manifest(self, manifest: VersionManifest, version_path: str) -> None:
        import httpx

        # Bring the local checkpoint up to this version host-side (apply_deltas
        # chain-replays from whatever is applied, verifying each base_version),
        # then reload the full local checkpoint. version_path is the published
        # version dir; its parent is the root of weight_v* dirs apply_deltas walks.
        delta_root = str(Path(version_path).parent)
        _t_apply = time.perf_counter()
        await asyncio.to_thread(
            self._apply_deltas(), self.local_checkpoint_dir, delta_root, int(manifest.version)
        )
        logger.info(
            "[apply timing] v=%s host_delta_apply=%.2fs", manifest.version, time.perf_counter() - _t_apply
        )

        _t_reload = time.perf_counter()
        payload = {
            "model_path": self.local_checkpoint_dir,
            "weight_version": str(manifest.version),
            # The sync manager flushes via GET /flush_cache while quiesced.
            # The engine-side post-apply flush hard-asserts on failure
            # (killing the scheduler process) if any request slipped in, so
            # it must stay disabled here.
            "flush_cache": False,
        }
        async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
            resp = await client.post(f"{self.upstream_url}/update_weights_from_disk", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") is False:
                raise RuntimeError(f"SGLang rejected weight update: {data}")
        logger.info(
            "[apply timing] v=%s engine_reload=%.2fs", manifest.version, time.perf_counter() - _t_reload
        )
