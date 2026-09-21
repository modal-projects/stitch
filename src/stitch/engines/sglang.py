"""``Engine`` adapter for SGLang's staged checkpoint updates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from stitch.engines.base import Engine, EngineHealth, EngineHealthStatus
from stitch.errors import UnrecoverableEngineError
from stitch.types import VersionKind, VersionManifest, VersionRef


class SGLangEngine(Engine):
    def __init__(
        self,
        base_url: str,
        local_checkpoint_dir: str | None = None,
        *,
        delta_update_mode: Literal["disk", "cpu"] = "disk",
        control_timeout: float = 120.0,
        health_timeout: float = 5.0,
        weight_staging_timeout: float = 3600.0,
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
        self.local_checkpoint_dir = local_checkpoint_dir
        self.delta_update_mode = delta_update_mode
        self._control_timeout = control_timeout
        self._health_timeout = health_timeout
        self._weight_staging_timeout = weight_staging_timeout
        self._weight_update_timeout = weight_update_timeout

    def base_url(self) -> str:
        return self._base_url

    def blocked_routes(self) -> frozenset[str]:
        return frozenset(
            {
                "prepare_weight_update",
                "commit_weight_update",
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

    async def check_health(self) -> EngineHealth:
        """Require scheduler progress through SGLang's generation health probe."""
        import httpx

        try:
            async with httpx.AsyncClient(
                timeout=self._health_timeout, trust_env=False
            ) as client:
                response = await client.get(f"{self._base_url}/health")
        except httpx.ConnectError as exc:
            return EngineHealth(EngineHealthStatus.UNREACHABLE, str(exc))
        except httpx.RequestError as exc:
            return EngineHealth(
                EngineHealthStatus.UNRESPONSIVE,
                f"{type(exc).__name__}: {exc}",
            )
        if response.status_code == 200:
            return EngineHealth(EngineHealthStatus.HEALTHY)
        return EngineHealth(
            EngineHealthStatus.UNRESPONSIVE,
            f"health endpoint returned HTTP {response.status_code}",
        )

    async def stage(self, manifest: VersionManifest, source_dir: str) -> None:
        self._validate_manifest(manifest)
        await self._post(
            "/prepare_weight_update",
            {
                "checkpoint_source_dir": str(Path(source_dir).parent),
                "target_version": manifest.ref.version,
            },
            timeout=self._weight_staging_timeout,
            action="weight preparation",
        )

    async def initialize_update_destination(self, boot_version: int = 0) -> None:
        """Verify the staging destination SGLang built before reporting ready."""
        server_info = await self._get_json("/server_info")
        actual_mode = server_info.get("weight_update_staging")
        if actual_mode != self.delta_update_mode:
            raise RuntimeError(
                "sglang staging mode does not match the sidecar: "
                f"expected {self.delta_update_mode!r}, got {actual_mode!r}"
            )
        actual_dir = server_info.get("weight_update_local_checkpoint_dir")
        if actual_dir != self.local_checkpoint_dir:
            raise RuntimeError(
                "sglang staging checkpoint directory does not match the sidecar: "
                f"expected {self.local_checkpoint_dir!r}, got {actual_dir!r}"
            )

        model_info = await self._get_json("/model_info")
        try:
            actual_version = int(model_info["weight_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "sglang did not report a valid startup weight version"
            ) from exc
        if actual_version != boot_version:
            raise RuntimeError(
                "sglang startup weight version does not match the sidecar: "
                f"expected {boot_version}, got {actual_version}"
            )

    async def commit(
        self, manifest: VersionManifest, *, flush_cache: bool = False
    ) -> None:
        self._validate_manifest(manifest)
        await self._post(
            "/commit_weight_update",
            {
                "target_version": manifest.ref.version,
                "abort_all_requests": False,
                "torch_empty_cache": False,
                "flush_cache": flush_cache,
            },
            timeout=self._weight_update_timeout,
            action="weight commit",
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
        raise UnrecoverableEngineError(
            "staged weight updates are monotonic; start a fresh rollout replica "
            "to restore its boot checkpoint"
        )

    def _validate_manifest(self, manifest: VersionManifest) -> None:
        if manifest.kind is VersionKind.FULL and self.delta_update_mode == "cpu":
            raise ValueError(
                "CPU delta update mode accepts delta manifests only; "
                "use disk mode to publish full checkpoints"
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
            for choice in response.get("choices", []):
                meta = choice.get("meta_info")
                if isinstance(meta, dict):
                    meta["weight_version"] = str(served.version)
                    meta["weight_version_start"] = served.version
                    meta["weight_version_end"] = current.version

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

    async def _get_json(self, path: str) -> dict[str, Any]:
        resp = await self._get(path)
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"sglang returned a non-object response from {path}")
        return data

    async def _get(self, path: str, *, ok: tuple[int, ...] = (200,)) -> Any:
        import httpx

        async with httpx.AsyncClient(
            timeout=self._control_timeout, trust_env=False
        ) as client:
            resp = await client.get(f"{self._base_url}{path}")
        if resp.status_code not in ok:
            _raise_for_engine(resp, path)
        return resp


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
