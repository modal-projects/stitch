"""``ModalFlashPool`` — the ``Pool`` instance for a Modal Flash service.

Replicas are the Flash containers; the gateway is the Flash URL. This is a *client*
to a running pool — reach, enumerate, wake, scale — not the pool's deployment (that
is an example). Every Modal call is import-lazy, so the module loads without Modal.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from stitch.pools.base import Pool
from stitch.versions import VersionRef

logger = logging.getLogger(__name__)


class ModalFlashPool(Pool):
    def __init__(self, app_name: str, cls_name: str) -> None:
        self.app_name = app_name
        self.cls_name = cls_name

    def gateway_url(self) -> str:
        import modal

        cls = modal.Cls.from_name(self.app_name, self.cls_name)
        urls = cls._experimental_get_flash_urls()
        if not urls:
            raise RuntimeError(
                f"no Flash gateway URL for {self.app_name}.{self.cls_name} — deploy the app first"
            )
        return str(urls[0]).rstrip("/")

    def discover_replicas(self) -> list[str]:
        import modal
        import modal.experimental

        modal.Cls.from_name(self.app_name, self.cls_name)  # client-side resolve side effect
        containers = modal.experimental.flash_get_containers(self.app_name, self.cls_name)
        return [_normalize_url(h) for c in containers if (h := _host(c))]

    def wake(self, replicas: list[str], ref: VersionRef) -> None:
        # Kick each replica to reconcile now; it re-reads the authoritative pointer,
        # so no target version travels in the body. Runs in the trainer's publish hot
        # path, so fan out over one shared client instead of a serial round-trip each.
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

    def scale(self, *, min: int | None = None, max: int | None = None) -> None:
        import modal

        fn = modal.Cls.from_name(self.app_name, self.cls_name)._get_class_service_function()
        kwargs: dict[str, int] = {}
        if min is not None:
            kwargs["min_containers"] = min
        if max is not None:
            kwargs["max_containers"] = max
        if kwargs:
            fn.update_autoscaler(**kwargs)


def _host(container) -> str | None:
    if isinstance(container, dict):
        return container.get("host")
    return getattr(container, "host", None)


def _normalize_url(host: str) -> str:
    host = str(host).rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"https://{host}"
