"""``Engine`` adapter for SGLang's staged checkpoint updates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from stitch.engines.base import Engine
from stitch.types import VersionKind, VersionManifest, VersionRef


class SGLangEngine(Engine):
    def __init__(
        self,
        base_url: str,
        base_checkpoint_dir: str,
        local_checkpoint_dir: str | None = None,
        *,
        delta_update_mode: Literal["disk", "cpu"] = "disk",
        disk_load_format: str = "auto",
        control_timeout: float = 120.0,
        weight_update_timeout: float = 600.0,
    ) -> None:
        if delta_update_mode not in ("disk", "cpu"):
            raise ValueError(
                "delta_update_mode must be either 'disk' or 'cpu', "
                f"got {delta_update_mode!r}"
            )
        if delta_update_mode == "disk" and not local_checkpoint_dir:
            raise ValueError("disk delta update mode requires local_checkpoint_dir")
        self._base_url = base_url.rstrip("/")
        self.base_checkpoint_dir = base_checkpoint_dir
        self.local_checkpoint_dir = local_checkpoint_dir
        self.delta_update_mode = delta_update_mode
        self.disk_load_format = disk_load_format
        self._control_timeout = control_timeout
        self._weight_update_timeout = weight_update_timeout

    def base_url(self) -> str:
        return self._base_url

    def blocked_routes(self) -> frozenset[str]:
        return frozenset(
            {
                "update_weights_from_disk",
                "update_weights_from_cpu",
                "update_weights_from_distributed",
                "update_weights_from_tensor",
                "stage_weight_update",
                "flush_cache",
                "pause_generation",
                "continue_generation",
                "abort_request",
            }
        )

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        await self._stage_weight_update(
            checkpoint_source_dir=str(Path(source_dir).parent),
            target_version=manifest.ref.version,
            destination=self._destination_for(manifest),
        )

    async def initialize_update_destination(self) -> None:
        await self._stage_weight_update(
            checkpoint_source_dir=None,
            target_version=0,
            destination=self.delta_update_mode,
        )

    async def commit(
        self,
        manifest: VersionManifest,
        *,
        flush_cache: bool = False,
    ) -> None:
        if self._destination_for(manifest) == "cpu":
            path = "/update_weights_from_cpu"
            payload: dict[str, Any] = {
                "target_version": manifest.ref.version,
                "flush_cache": flush_cache,
            }
        else:
            assert self.local_checkpoint_dir is not None
            path = "/update_weights_from_disk"
            payload = {
                "model_path": self.local_checkpoint_dir,
                "load_format": self.disk_load_format,
                "weight_version": str(manifest.ref.version),
                "flush_cache": flush_cache,
            }
        await self._post(
            path,
            payload,
            timeout=self._weight_update_timeout,
            action="weight update",
        )

    async def flush_cache(self) -> None:
        await self._get("/flush_cache", ok=(200, 404))

    async def pause(self) -> None:
        await self._post(
            "/pause_generation", {"mode": "in_place"}, timeout=self._control_timeout
        )

    async def resume(self) -> None:
        await self._post("/continue_generation", {}, timeout=self._control_timeout)

    async def reset(self) -> None:
        if self.delta_update_mode == "cpu":
            raise RuntimeError(
                "CPU delta update mode cannot reset a live engine to version 0; "
                "start a fresh rollout replica for a new run"
            )
        assert self.local_checkpoint_dir is not None
        await self._stage_weight_update(
            checkpoint_source_dir=None,
            target_version=0,
            destination="disk",
        )
        await self._post(
            "/update_weights_from_disk",
            {
                "model_path": self.local_checkpoint_dir,
                "load_format": self.disk_load_format,
                "weight_version": "0",
                "flush_cache": False,
            },
            timeout=self._weight_update_timeout,
            action="reset weights to base",
        )

    def _destination_for(
        self,
        manifest: VersionManifest,
    ) -> Literal["disk", "cpu"]:
        if manifest.kind is VersionKind.FULL:
            if self.delta_update_mode == "cpu":
                raise ValueError(
                    "CPU delta update mode accepts delta manifests only; "
                    "use disk mode to publish full checkpoints"
                )
            return "disk"
        return self.delta_update_mode

    async def _stage_weight_update(
        self,
        *,
        checkpoint_source_dir: str | None,
        target_version: int,
        destination: Literal["disk", "cpu"],
    ) -> None:
        payload: dict[str, Any] = {
            "base_checkpoint_dir": self.base_checkpoint_dir,
            "target_version": target_version,
            "destination": destination,
        }
        if checkpoint_source_dir is not None:
            payload["checkpoint_source_dir"] = checkpoint_source_dir
        if destination == "disk":
            assert self.local_checkpoint_dir is not None
            payload["local_checkpoint_dir"] = self.local_checkpoint_dir
        await self._post(
            "/stage_weight_update",
            payload,
            timeout=self._weight_update_timeout,
            action="weight staging",
        )

    def stamp_request(self, request: dict[str, Any], served: VersionRef) -> None:
        user = request.get("extra_key")
        if isinstance(user, list):
            request["extra_key"] = [self._extra_key(served, k) for k in user]
        else:
            request["extra_key"] = self._extra_key(served, user)

    def stamp_response(
        self, response: dict[str, Any], served: VersionRef, current: VersionRef
    ) -> None:
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

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float | None,
        action: str | None = None,
    ) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(f"{self._base_url}{path}", json=payload)
        _raise_for_engine(resp, action or path)

    async def _get(self, path: str, *, ok: tuple[int, ...] = (200,)) -> None:
        import httpx

        async with httpx.AsyncClient(
            timeout=self._control_timeout, trust_env=False
        ) as client:
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
        raise RuntimeError(
            f"sglang rejected {action} (HTTP {resp.status_code}): {data.get('message', data)}"
        )
