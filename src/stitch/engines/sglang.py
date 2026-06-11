"""SGLang rollout engine adapters."""

from __future__ import annotations

from dataclasses import dataclass

from stitch.protocol import VersionManifest


@dataclass
class SGLangDiskDeltaAdapter:
    upstream_url: str
    backend: str = "sparse_delta"

    def __post_init__(self) -> None:
        self.upstream_url = self.upstream_url.rstrip("/")

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

        files = manifest.transition_artifact_paths()
        if not files:
            return

        payload = {
            "model_path": version_path,
            "files": files,
            "load_format": manifest.load_format,
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
