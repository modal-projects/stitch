"""``ModalFlashPool`` — the ``Pool`` instance for a Modal Flash service.

Replicas are the Flash containers; the gateway is the Flash URL. This is a *client*
to a running pool — reach, enumerate, wake, scale — not the pool's deployment (that
is an example). Every Modal call is import-lazy, so the module loads without Modal.

The ``*_async`` overrides use the Modal SDK's native ``.aio()`` interface (every Modal
call is synchronicity-wrapped and awaitable), so async callers never park a worker
thread just to wait on Modal's own event loop.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

from stitch.pools.base import Pool
from stitch.types import VersionRef

logger = logging.getLogger(__name__)


class ModalFlashPool(Pool):
    def __init__(self, app_name: str, cls_name: str) -> None:
        self.app_name = app_name
        self.cls_name = cls_name
        self._upstream_url_cache: str | None = None

    def _server(self):
        import modal

        return modal.Server.from_name(self.app_name, self.cls_name)

    def _upstream_url(self) -> str:
        """The upstream class's own Flash URL, not an LB override. Stable for a
        deployed app, so it is cached for the client's lifetime."""
        if self._upstream_url_cache is None:
            self._upstream_url_cache = self._require_gateway(self._server().get_url())
        return self._upstream_url_cache

    def replica_request(self, replica: str, path: str) -> tuple[str, dict[str, str]]:
        # Direct container URLs sit behind relay auth, so a replica is addressed
        # through the pool URL. The edge keys upstreams by host:port.
        host = replica.split("://", 1)[-1].rstrip("/")
        if ":" not in host:
            host = f"{host}:443"
        return f"{self._upstream_url()}{path}", {"modal-flash-upstream": host}

    def gateway_url(self) -> str:
        return self._require_gateway(self._server().get_url())

    async def gateway_url_async(self) -> str:
        return self._require_gateway(await self._server().get_url.aio())

    def _require_gateway(self, url: str | None) -> str:
        if not url:
            raise RuntimeError(
                f"no gateway URL for {self.app_name}.{self.cls_name} — deploy the app first"
            )
        return str(url).rstrip("/")

    def discover_replicas(self) -> list[str]:
        return _replica_urls(list_flash_containers(self.app_name, self.cls_name))

    async def discover_replicas_async(self) -> list[str]:
        return _replica_urls(
            await list_flash_containers_async(self.app_name, self.cls_name)
        )

    def wake(self, replicas: list[str], ref: VersionRef) -> None:
        # Fan out (this is on the publish hot path); each replica re-reads the pointer, so no version in the body.
        if not replicas:
            return
        import httpx

        with httpx.Client(timeout=5.0, trust_env=False) as client:

            def wake_one(url: str) -> None:
                try:
                    target, headers = self.replica_request(url, "/wake")
                    client.post(target, headers=headers).raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "failed to wake %s for %s: %s", url, ref.identity, exc
                    )

            with ThreadPoolExecutor(max_workers=min(16, len(replicas))) as pool:
                list(pool.map(wake_one, replicas))

    async def wake_async(self, replicas: list[str], ref: VersionRef) -> None:
        if not replicas:
            return
        import httpx

        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:

            async def wake_one(url: str) -> None:
                try:
                    target, headers = await asyncio.to_thread(
                        self.replica_request, url, "/wake"
                    )
                    (await client.post(target, headers=headers)).raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "failed to wake %s for %s: %s", url, ref.identity, exc
                    )

            await asyncio.gather(*(wake_one(url) for url in replicas))

    def scale(self, *, min: int | None = None, max: int | None = None) -> None:
        kwargs: dict[str, int] = {}
        if min is not None:
            kwargs["min_containers"] = min
        if max is not None:
            kwargs["max_containers"] = max
        if kwargs:
            self._server().update_autoscaler(**kwargs)


async def _list_flash_containers_rpc(app_name: str, cls_name: str) -> list[Any]:
    from modal.client import _Client
    from modal.config import config
    from modal_proto import api_pb2

    client = await _Client.from_env()
    fn = await client.stub.FunctionGet(
        api_pb2.FunctionGetRequest(
            app_name=app_name,
            object_tag=cls_name,
            environment_name=config.get("environment") or "",
        )
    )
    response = await client.stub.FlashContainerList(
        api_pb2.FlashContainerListRequest(function_id=fn.function_id)
    )
    return list(response.containers)


@cache
def _flash_container_lister():
    from modal._utils.async_utils import synchronize_api

    return synchronize_api(_list_flash_containers_rpc)


def list_flash_containers(app_name: str, cls_name: str) -> list[Any]:
    """Return live containers for an ``@app.server`` function.

    Modal's experimental helper resolves ``<Class>.*``, while ``@app.server``
    registers the plain ``<Class>`` tag. Resolve that function first, then list
    its Flash containers by function id.
    """
    return _flash_container_lister()(app_name, cls_name)


async def list_flash_containers_async(app_name: str, cls_name: str) -> list[Any]:
    return await _flash_container_lister().aio(app_name, cls_name)


def _replica_urls(containers) -> list[str]:
    return [_normalize_url(h) for c in containers if (h := _host(c))]


def _host(container) -> str | None:
    if isinstance(container, dict):
        return container.get("host")
    return getattr(container, "host", None)


def _normalize_url(host: str) -> str:
    host = str(host).rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"https://{host}"
