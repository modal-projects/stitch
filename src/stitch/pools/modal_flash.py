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

from stitch.pools.base import Pool
from stitch.types import VersionRef

logger = logging.getLogger(__name__)


class ModalFlashPool(Pool):
    def __init__(self, app_name: str, cls_name: str) -> None:
        self.app_name = app_name
        self.cls_name = cls_name

    def _cls(self):
        import modal

        try:
            return modal.Cls.from_name(self.app_name, self.cls_name)
        except Exception as exc:  # NotFoundError etc.
            raise RuntimeError(
                f"cannot resolve {self.app_name}.{self.cls_name} — is the Server pool deployed?"
            ) from exc

    def gateway_url(self) -> str:
        return self._pick_gateway(self._cls()._experimental_get_flash_urls())

    async def gateway_url_async(self) -> str:
        return self._pick_gateway(await self._cls()._experimental_get_flash_urls.aio())

    def _pick_gateway(self, urls) -> str:
        if not urls:
            raise RuntimeError(
                f"no Flash gateway URL for {self.app_name}.{self.cls_name} — deploy the app first"
            )
        return str(urls[0]).rstrip("/")

    def discover_replicas(self) -> list[str]:
        import modal.experimental

        self._cls()  # resolve first: clear error if not deployed
        return _replica_urls(modal.experimental.flash_get_containers(self.app_name, self.cls_name))

    async def discover_replicas_async(self) -> list[str]:
        import modal.experimental

        self._cls()  # resolve first: clear error if not deployed
        return _replica_urls(await modal.experimental.flash_get_containers.aio(self.app_name, self.cls_name))

    def wake(self, replicas: list[str], ref: VersionRef) -> None:
        # Fan out (this is on the publish hot path); each replica re-reads the pointer, so no version in the body.
        if not replicas:
            return
        import httpx

        with httpx.Client(timeout=5.0, trust_env=False) as client:

            def wake_one(url: str) -> None:
                try:
                    client.post(f"{url}/wake").raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed to wake %s for %s: %s", url, ref.identity, exc)

            with ThreadPoolExecutor(max_workers=min(16, len(replicas))) as pool:
                list(pool.map(wake_one, replicas))

    async def wake_async(self, replicas: list[str], ref: VersionRef) -> None:
        if not replicas:
            return
        import httpx

        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:

            async def wake_one(url: str) -> None:
                try:
                    (await client.post(f"{url}/wake")).raise_for_status()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("failed to wake %s for %s: %s", url, ref.identity, exc)

            await asyncio.gather(*(wake_one(url) for url in replicas))

    def scale(self, *, min: int | None = None, max: int | None = None) -> None:
        fn = self._cls()._get_class_service_function()
        kwargs: dict[str, int] = {}
        if min is not None:
            kwargs["min_containers"] = min
        if max is not None:
            kwargs["max_containers"] = max
        if kwargs:
            fn.update_autoscaler(**kwargs)


def _replica_urls(containers) -> list[str]:
    return [_normalize_url(h) for c in containers if (h := _host(c))]


def _host(container) -> str | None:
    if isinstance(container, dict):
        return container.get("host")
    return getattr(container, "host", None)


def _normalize_url(host: str) -> str:
    host = str(host).rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"https://{host}"
